#!/usr/bin/env python3
"""
Build a cropped foot / lower-leg dataset from HALPE-26 YOLO pose labels.

For each image+label pair in the Stage-A dataset, this script:
  1. Locates the ankle keypoints (indices 15, 16) of each person.
  2. Builds a square crop centred on the mid-ankle, expanded downward to
     capture toes, clamped to image boundaries.
  3. Transforms the YOLO-normalised keypoints into the crop coordinate system.
  4. Writes the crop image + transformed label in Ultralytics pose format.
  5. Generates a ``foot_crops.yaml`` dataset YAML with ``flip_idx`` and
     ``kpt_shape: [26, 3]``.

The resulting dataset has the same HALPE-26 keypoint order — upper-body
keypoints that fall outside the crop are written with visibility 0.

Usage
-----
::

    python scripts/build_foot_crops.py \\
        --stage-a-dir data/halpe26_yolo_pose \\
        --output-dir  data/halpe26_foot_crops \\
        --margin 0.4 --min-crop 192 --max-crop 384
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

LOG = logging.getLogger(__name__)

NUM_KEYPOINTS = 26
YOLO_VALUES = 1 + 4 + NUM_KEYPOINTS * 3  # 83

L_ANKLE, R_ANKLE = 15, 16
FOOT_KPT_INDICES = [15, 16, 20, 21, 22, 23, 24, 25]

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
            17, 18, 19, 21, 20, 23, 22, 25, 24]


def compute_crop_box(
    kpts_abs: np.ndarray,
    img_w: int,
    img_h: int,
    margin_frac: float,
    min_crop: int,
    max_crop: int,
) -> tuple[int, int, int, int] | None:
    """Return (x1, y1, x2, y2) crop box or None if no ankle visible."""
    ankles = []
    for idx in [L_ANKLE, R_ANKLE]:
        x, y, v = kpts_abs[idx]
        if v > 0 and x > 0 and y > 0:
            ankles.append((x, y))
    if not ankles:
        return None

    cx = np.mean([a[0] for a in ankles])
    cy = np.mean([a[1] for a in ankles])

    foot_pts = []
    for idx in FOOT_KPT_INDICES:
        x, y, v = kpts_abs[idx]
        if v > 0 and x > 0 and y > 0:
            foot_pts.append((x, y))
    if not foot_pts:
        foot_pts = ankles

    xs = [p[0] for p in foot_pts]
    ys = [p[1] for p in foot_pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 50)
    half = int(span * (1 + margin_frac) / 2)
    half = max(half, min_crop // 2)
    half = min(half, max_crop // 2)

    # Shift centre down to capture toes
    cy_shifted = cy + half * 0.15

    x1 = int(cx - half)
    y1 = int(cy_shifted - half)
    x2 = int(cx + half)
    y2 = int(cy_shifted + half)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return x1, y1, x2, y2


def transform_label(
    parts: list[float],
    crop_box: tuple[int, int, int, int],
    orig_w: int,
    orig_h: int,
) -> str | None:
    """Transform one YOLO pose label line into crop coordinates."""
    x1, y1, x2, y2 = crop_box
    cw = x2 - x1
    ch = y2 - y1

    cls_id = int(parts[0])
    # Bbox in original image (absolute)
    bcx = parts[1] * orig_w
    bcy = parts[2] * orig_h
    bw = parts[3] * orig_w
    bh = parts[4] * orig_h

    # Transform bbox into crop
    bcx_c = (bcx - x1) / cw
    bcy_c = (bcy - y1) / ch
    bw_c = bw / cw
    bh_c = bh / ch

    # Clamp bbox
    bcx_c = max(0, min(1, bcx_c))
    bcy_c = max(0, min(1, bcy_c))
    bw_c = max(0, min(1, bw_c))
    bh_c = max(0, min(1, bh_c))

    if bw_c < 0.01 or bh_c < 0.01:
        return None

    out = [cls_id, bcx_c, bcy_c, bw_c, bh_c]
    for i in range(NUM_KEYPOINTS):
        j = 5 + i * 3
        kx = parts[j] * orig_w
        ky = parts[j + 1] * orig_h
        kv = parts[j + 2]

        kx_c = (kx - x1) / cw
        ky_c = (ky - y1) / ch

        if kv > 0 and 0 <= kx_c <= 1 and 0 <= ky_c <= 1:
            out.extend([kx_c, ky_c, kv])
        else:
            out.extend([0.0, 0.0, 0.0])

    return " ".join(f"{v:.6g}" for v in out)


def process_split(
    stage_a_dir: Path,
    output_dir: Path,
    split: str,
    margin_frac: float,
    min_crop: int,
    max_crop: int,
) -> dict:
    images_src = stage_a_dir / "images" / split
    labels_src = stage_a_dir / "labels" / split
    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "crops": 0, "skipped_no_ankle": 0, "skipped_no_image": 0}

    if not labels_src.exists():
        LOG.warning("Labels dir not found: %s", labels_src)
        return stats

    for lbl_path in sorted(labels_src.glob("*.txt")):
        stem = lbl_path.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_src / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            stats["skipped_no_image"] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            stats["skipped_no_image"] += 1
            continue
        img_h, img_w = img.shape[:2]
        stats["images"] += 1

        with open(lbl_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for line_idx, line in enumerate(lines):
            parts = [float(x) for x in line.split()]
            if len(parts) != YOLO_VALUES:
                continue

            kpts_abs = np.zeros((NUM_KEYPOINTS, 3))
            for k in range(NUM_KEYPOINTS):
                j = 5 + k * 3
                kpts_abs[k, 0] = parts[j] * img_w
                kpts_abs[k, 1] = parts[j + 1] * img_h
                kpts_abs[k, 2] = parts[j + 2]

            crop_box = compute_crop_box(kpts_abs, img_w, img_h, margin_frac, min_crop, max_crop)
            if crop_box is None:
                stats["skipped_no_ankle"] += 1
                continue

            x1, y1, x2, y2 = crop_box
            crop_img = img[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue

            new_label = transform_label(parts, crop_box, img_w, img_h)
            if new_label is None:
                continue

            crop_name = f"{stem}_p{line_idx}"
            cv2.imwrite(str(images_out / f"{crop_name}.jpg"), crop_img)
            with open(labels_out / f"{crop_name}.txt", "w") as f:
                f.write(new_label + "\n")
            stats["crops"] += 1

    return stats


def write_dataset_yaml(output_dir: Path) -> None:
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [NUM_KEYPOINTS, 3],
        "flip_idx": FLIP_IDX,
        "names": {0: "person"},
        "nc": 1,
    }
    yaml_path = output_dir / "foot_crops.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    LOG.info("Wrote %s", yaml_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build foot/lower-leg crop dataset from HALPE-26 YOLO labels."
    )
    parser.add_argument("--stage-a-dir", type=Path, required=True,
                        help="Stage A YOLO pose dataset root (images/ + labels/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for cropped dataset")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="Margin fraction around ankle span (default 0.4)")
    parser.add_argument("--min-crop", type=int, default=192,
                        help="Minimum crop side in pixels (default 192)")
    parser.add_argument("--max-crop", type=int, default=384,
                        help="Maximum crop side in pixels (default 384)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_stats: dict[str, int] = {}
    for split in ("train", "val"):
        stats = process_split(
            args.stage_a_dir, output_dir, split,
            args.margin, args.min_crop, args.max_crop,
        )
        LOG.info("[%s] %s", split, stats)
        for k, v in stats.items():
            total_stats[k] = total_stats.get(k, 0) + v

    write_dataset_yaml(output_dir)

    LOG.info("Total: %s", total_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
