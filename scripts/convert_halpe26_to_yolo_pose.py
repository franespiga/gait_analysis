#!/usr/bin/env python3
"""
Convert HALPE-26 dataset (COCO-style JSON with 26 keypoints) to Ultralytics YOLO pose format.

Assumptions:
- Input has images/train, images/val and annotations/halpe26/*.json with 26 keypoints per person.
- COCO bbox format: [x, y, width, height]; keypoints: flat list of 78 values (26 * [x, y, v]).
- Only category "person" (category_id resolved from JSON) is converted.
- Output is one .txt label file per image, one line per person: class cx cy w h kpt1_x kpt1_y kpt1_v ... kpt26_v (83 values).
- Normalized coordinates in [0, 1]; visibility preserved as in COCO (0, 1, 2).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

LOG = logging.getLogger(__name__)

NUM_KEYPOINTS = 26
KEYPOINT_VALUES = NUM_KEYPOINTS * 3  # 78
YOLO_POSE_VALUES_PER_LINE = 1 + 4 + KEYPOINT_VALUES  # class + bbox + keypoints = 83
PERSON_CLASS_ID = 0

# Default annotation filenames under dataset_root/annotations/halpe26/
DEFAULT_TRAIN_JSON = "halpe_train_v1.json"
DEFAULT_VAL_JSON = "halpe_val_v1.json"

VISUALIZE_SAMPLES_SCRIPT = '''#!/usr/bin/env python3
"""Quick QA: draw a few random YOLO pose labels on images. Run from this directory."""
from pathlib import Path
import random
import sys

try:
    import cv2
except ImportError:
    print("Install opencv-python to run visualization.", file=sys.stderr)
    sys.exit(1)

def main():
    images_dir = Path("images/train")
    labels_dir = Path("labels/train")
    if not images_dir.exists():
        images_dir = Path("images/val")
        labels_dir = Path("labels/val")
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not image_files:
        print("No images in images/train or images/val")
        return
    sample = random.sample(image_files, min(5, len(image_files)))
    for imp in sample:
        lbl_path = labels_dir / (imp.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(imp))
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # class cx cy w h then 26*3 keypoints
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for i in range(26):
                    j = 5 + i * 3
                    if j + 2 < len(parts):
                        kx = float(parts[j])
                        ky = float(parts[j+1])
                        v = float(parts[j+2])
                        if v > 0:
                            px, py = int(kx * w), int(ky * h)
                            cv2.circle(img, (px, py), 4, (0, 0, 255), -1)
        cv2.imshow(imp.name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def get_person_category_id(categories: list[dict[str, Any]]) -> int | None:
    """Return category id for 'person' or first category with keypoints."""
    for cat in categories:
        if cat.get("name") == "person":
            return int(cat["id"])
    for cat in categories:
        if cat.get("keypoints"):
            return int(cat["id"])
    return None


def clamp_norm(value: float, name: str, log_fn: Any) -> float:
    """Clamp to [0, 1] and log if changed."""
    if value < 0:
        log_fn("Clamping %s from %s to 0", name, value)
        return 0.0
    if value > 1:
        log_fn("Clamping %s from %s to 1", name, value)
        return 1.0
    return value


def count_visible_keypoints(keypoints: list[float]) -> int:
    """Count keypoints with visibility > 0 (COCO: 0=not labeled, 1=labeled, 2=visible)."""
    n = 0
    for i in range(2, len(keypoints), 3):
        if i < len(keypoints) and keypoints[i] > 0:
            n += 1
    return n


def coco_ann_to_yolo_line(
    ann: dict[str, Any],
    img_w: int,
    img_h: int,
    min_visible_kpts: int,
    report: dict[str, Any],
) -> str | None:
    """
    Convert one COCO person annotation to one YOLO pose line.
    Returns None if annotation should be skipped; otherwise returns the line string.
    """
    keypoints = ann.get("keypoints", [])
    if not isinstance(keypoints, list) or len(keypoints) < KEYPOINT_VALUES:
        report["skip_reasons"]["fewer_than_78_keypoints"] = report["skip_reasons"].get("fewer_than_78_keypoints", 0) + 1
        return None
    kp = keypoints[:KEYPOINT_VALUES]
    if count_visible_keypoints(kp) < min_visible_kpts:
        report["skip_reasons"]["below_min_visible_kpts"] = report["skip_reasons"].get("below_min_visible_kpts", 0) + 1
        return None
    bbox = ann.get("bbox", [])
    if len(bbox) < 4:
        report["skip_reasons"]["invalid_bbox"] = report["skip_reasons"].get("invalid_bbox", 0) + 1
        return None
    x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    if img_w <= 0 or img_h <= 0:
        report["skip_reasons"]["invalid_image_size"] = report["skip_reasons"].get("invalid_image_size", 0) + 1
        return None
    if bw <= 0 or bh <= 0:
        report["skip_reasons"]["degenerate_bbox"] = report["skip_reasons"].get("degenerate_bbox", 0) + 1
        return None
    cx = x + bw / 2
    cy = y + bh / 2
    cx_n = clamp_norm(cx / img_w, "cx", LOG.debug)
    cy_n = clamp_norm(cy / img_h, "cy", LOG.debug)
    bw_n = clamp_norm(bw / img_w, "w", LOG.debug)
    bh_n = clamp_norm(bh / img_h, "h", LOG.debug)
    parts = [PERSON_CLASS_ID, cx_n, cy_n, bw_n, bh_n]
    for i in range(NUM_KEYPOINTS):
        xi = kp[i * 3] / img_w if kp[i * 3] > 0 else 0.0
        yi = kp[i * 3 + 1] / img_h if kp[i * 3 + 1] > 0 else 0.0
        vi = kp[i * 3 + 2] if (i * 3 + 2) < len(kp) else 0.0
        xi = clamp_norm(xi, f"kpt{i}_x", LOG.debug)
        yi = clamp_norm(yi, f"kpt{i}_y", LOG.debug)
        parts.extend([xi, yi, vi])
    if len(parts) != YOLO_POSE_VALUES_PER_LINE:
        report["skip_reasons"]["wrong_line_length"] = report["skip_reasons"].get("wrong_line_length", 0) + 1
        return None
    return " ".join(f"{p:.6g}" for p in parts)


def process_split(
    dataset_root: Path,
    output_dir: Path,
    json_path: Path,
    split: str,
    person_cid: int,
    min_visible_kpts: int,
    copy_images: bool,
    check_images: bool,
    report: dict[str, Any],
    force: bool,
) -> tuple[int, int, int, int, list[dict[str, Any]]]:
    """
    Process one split (train or val): copy/symlink images, write YOLO labels, build manifest rows.
    Returns (images_processed, labels_written, total_persons, skipped_persons, manifest_rows).
    """
    images_dir_src = dataset_root / "images" / split
    labels_dir_out = output_dir / "labels" / split
    images_dir_out = output_dir / "images" / split
    labels_dir_out.mkdir(parents=True, exist_ok=True)
    images_dir_out.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        LOG.warning("JSON not found: %s", json_path)
        return 0, 0, 0, 0, []

    with open(json_path) as f:
        data = json.load(f)
    images = {im["id"]: im for im in data.get("images", [])}
    annotations = data.get("annotations", [])
    ann_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        if ann.get("category_id") != person_cid:
            continue
        image_id = ann["image_id"]
        ann_by_image.setdefault(image_id, []).append(ann)

    manifest_rows: list[dict[str, Any]] = []
    images_processed = 0
    labels_written = 0
    total_persons = 0
    skipped_persons = 0

    for image_id, im in images.items():
        img_w = int(im.get("width", 0))
        img_h = int(im.get("height", 0))
        file_name = im.get("file_name", "")
        if not file_name:
            report["skip_reasons"]["missing_file_name"] = report["skip_reasons"].get("missing_file_name", 0) + 1
            continue
        src_image = images_dir_src / file_name
        if check_images and not src_image.exists():
            LOG.warning("Image not found: %s", src_image)
            report["skip_reasons"]["image_not_found"] = report["skip_reasons"].get("image_not_found", 0) + 1
            continue
        out_image = images_dir_out / Path(file_name).name
        if not src_image.exists():
            continue
        if not force and out_image.exists():
            pass
        elif copy_images:
            shutil.copy2(src_image, out_image)
        else:
            try:
                if out_image.exists():
                    out_image.unlink()
                os.symlink(src_image.resolve(), out_image)
            except OSError:
                shutil.copy2(src_image, out_image)
        images_processed += 1
        anns = ann_by_image.get(image_id, [])
        lines: list[str] = []
        kept = 0
        for ann in anns:
            total_persons += 1
            line = coco_ann_to_yolo_line(ann, img_w, img_h, min_visible_kpts, report)
            if line is None:
                skipped_persons += 1
                continue
            lines.append(line)
            kept += 1
        label_path = labels_dir_out / (Path(file_name).stem + ".txt")
        if lines:
            with open(label_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            labels_written += 1
        manifest_rows.append({
            "image_path": str(out_image),
            "label_path": str(label_path),
            "width": img_w,
            "height": img_h,
            "num_persons": len(anns),
            "kept": kept,
            "skipped": len(anns) - kept,
        })

    return images_processed, labels_written, total_persons, skipped_persons, manifest_rows


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label_path", "width", "height", "num_persons", "kept", "skipped"])
        w.writeheader()
        w.writerows(rows)


def write_yaml(output_dir: Path, yaml_path: Path) -> None:
    """Write halpe26-pose.yaml for Ultralytics. path is relative to yaml location."""
    # Left-right swap indices for horizontal flip augmentation (HALPE-26 order).
    # Centre keypoints map to themselves; symmetric pairs swap.
    # 0 Nose, 1 LEye↔2 REye, 3 LEar↔4 REar, 5 LSh↔6 RSh, 7 LEl↔8 REl,
    # 9 LWr↔10 RWr, 11 LHip↔12 RHip, 13 LKn↔14 RKn, 15 LAn↔16 RAn,
    # 17 Head, 18 Neck, 19 Hip, 20 LBigToe↔21 RBigToe,
    # 22 LSmallToe↔23 RSmallToe, 24 LHeel↔25 RHeel
    flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
                17, 18, 19, 21, 20, 23, 22, 25, 24]
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [NUM_KEYPOINTS, 3],
        "flip_idx": flip_idx,
        "names": {0: "person"},
        "nc": 1,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def run_qa(output_dir: Path, report: dict[str, Any]) -> None:
    """Verify label files: 83 values per line, coords in [0,1], and collect stats."""
    for split in ("train", "val"):
        labels_dir = output_dir / "labels" / split
        if not labels_dir.exists():
            continue
        images_ok = 0
        labels_ok = 0
        empty_images = 0
        total_instances = 0
        bad_lines = 0
        bad_coords = 0
        for lbl_path in labels_dir.glob("*.txt"):
            img_stem = lbl_path.stem
            img_candidates = list((output_dir / "images" / split).glob(f"{img_stem}.*"))
            if not img_candidates:
                continue
            images_ok += 1
            with open(lbl_path) as f:
                lines = f.readlines()
            if not lines:
                empty_images += 1
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                total_instances += 1
                parts = line.split()
                if len(parts) != YOLO_POSE_VALUES_PER_LINE:
                    bad_lines += 1
                    continue
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    bad_lines += 1
                    continue
                for i in (1, 2, 3, 4):  # cx cy w h
                    if not (0 <= vals[i] <= 1):
                        bad_coords += 1
                        break
                for i in range(26):
                    j = 5 + i * 3
                    if not (0 <= vals[j] <= 1 and 0 <= vals[j + 1] <= 1):
                        bad_coords += 1
                        break
                labels_ok += 1
        report["qa"] = report.get("qa", {})
        report["qa"][split] = {
            "images_with_label_file": images_ok,
            "label_lines_ok": labels_ok,
            "empty_label_files": empty_images,
            "total_person_instances": total_instances,
            "lines_bad_length": bad_lines,
            "lines_bad_coords": bad_coords,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert HALPE-26 COCO JSON to Ultralytics YOLO pose format.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="HALPE-26 dataset root (images/train, images/val, annotations/halpe26)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for YOLO pose dataset")
    parser.add_argument("--train-json", type=Path, default=None, help="Train annotation JSON (default: dataset-root/annotations/halpe26/halpe_train_v1.json)")
    parser.add_argument("--val-json", type=Path, default=None, help="Val annotation JSON (default: dataset-root/annotations/halpe26/halpe_val_v1.json)")
    parser.add_argument("--copy-images", action="store_true", help="Copy images; default is to symlink when possible")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--generate-yaml", action="store_true", default=True, help="Generate halpe26-pose.yaml (default: True)")
    parser.add_argument("--no-generate-yaml", action="store_false", dest="generate_yaml", help="Do not generate YAML")
    parser.add_argument("--check-images", action="store_true", help="Verify each image file exists before processing")
    parser.add_argument("--min-visible-kpts", type=int, default=1, help="Minimum visible keypoints per person to keep (default: 1)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_root.exists():
        LOG.error("Dataset root does not exist: %s", dataset_root)
        return 1

    halpe26 = dataset_root / "annotations" / "halpe26"
    train_json = args.train_json or halpe26 / DEFAULT_TRAIN_JSON
    val_json = args.val_json or halpe26 / DEFAULT_VAL_JSON
    if not train_json.is_absolute():
        train_json = dataset_root / train_json
    if not val_json.is_absolute():
        val_json = dataset_root / val_json

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "skip_reasons": {},
        "train": {},
        "val": {},
        "summary": {},
    }

    person_cid: int | None = None
    for jpath in (train_json, val_json):
        if not jpath.exists():
            continue
        with open(jpath) as f:
            data = json.load(f)
        cid = get_person_category_id(data.get("categories", []))
        if cid is not None:
            person_cid = cid
            break
    if person_cid is None:
        person_cid = 1
        LOG.warning("Could not resolve 'person' category id; using 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_imgs = 0
    total_labels = 0
    total_persons = 0
    total_skipped = 0

    for split, jpath in [("train", train_json), ("val", val_json)]:
        n_imgs, n_lbl, n_per, n_skip, manifest_rows = process_split(
            dataset_root,
            output_dir,
            jpath,
            split,
            person_cid,
            args.min_visible_kpts,
            args.copy_images,
            args.check_images,
            report,
            args.force,
        )
        total_imgs += n_imgs
        total_labels += n_lbl
        total_persons += n_per
        total_skipped += n_skip
        report[split] = {"images_processed": n_imgs, "labels_written": n_lbl, "persons": n_per, "skipped": n_skip}
        write_manifest(manifest_rows, output_dir / "manifests" / f"{split}_manifest.csv")

    run_qa(output_dir, report)
    report["summary"] = {
        "images_processed": total_imgs,
        "labels_written": total_labels,
        "total_person_instances": total_persons,
        "skipped_person_instances": total_skipped,
    }

    if args.generate_yaml:
        write_yaml(output_dir, output_dir / "halpe26-pose.yaml")
        LOG.info("Wrote %s", output_dir / "halpe26-pose.yaml")

    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "visualize_samples.py", "w") as f:
        f.write(VISUALIZE_SAMPLES_SCRIPT)
    LOG.info("Wrote %s", output_dir / "visualize_samples.py")

    with open(output_dir / "conversion_report.json", "w") as f:
        json.dump(report, f, indent=2)
    LOG.info("Wrote %s", output_dir / "conversion_report.json")

    LOG.info("Summary: images=%d labels=%d persons=%d skipped=%d", total_imgs, total_labels, total_persons, total_skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
