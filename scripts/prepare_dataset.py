#!/usr/bin/env python3
"""
Prepare dataset for training a YOLO pose model with foot keypoints.

- Download/prepare dataset or accept user-provided dataset path.
- Convert annotations to Ultralytics keypoint format (one .txt per image:
  class_id x1 y1 v1 x2 y2 v2 ... with normalized 0-1 coordinates).
- Output dataset YAML for training.

Keypoint schema (default): L_HEEL, L_BIG_TOE, L_SMALL_TOE, R_HEEL, R_BIG_TOE, R_SMALL_TOE (6 points).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Optional

import yaml


# Default 6 keypoints for gait
KEYPOINT_NAMES = [
    "L_HEEL", "L_BIG_TOE", "L_SMALL_TOE",
    "R_HEEL", "R_BIG_TOE", "R_SMALL_TOE",
]
N_KEYPOINTS = 6


def load_schema_yaml(schema_path: Optional[Path]) -> list[str]:
    """Load keypoint names from keypoint schema YAML (e.g. config/keypoint_schema_cmu.yaml) if present."""
    if schema_path and schema_path.exists():
        with open(schema_path) as f:
            data = yaml.safe_load(f)
        if data and "keypoints" in data:
            return list(data["keypoints"])
    return KEYPOINT_NAMES


def coco_wholebody_to_ultralytics(
    coco_annot_path: Path,
    images_dir: Path,
    output_dir: Path,
    keypoint_names: list[str],
    min_visibility: float = 0.0,
) -> int:
    """
    Convert COCO-WholeBody (or COCO format with keypoints) to Ultralytics pose format.

    Ultralytics format per line: class_id x1 y1 v1 x2 y2 v2 ... (normalized 0-1).
    v is 2 if visible, 1 if occluded, 0 if not labeled.
    """
    with open(coco_annot_path) as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco.get("images", [])}
    # Build keypoint index mapping from category; assume one category with keypoints
    categories = coco.get("categories", [])
    kpt_names_coco = []
    for cat in categories:
        if "keypoints" in cat:
            kpt_names_coco = cat["keypoints"]
            break
    if not kpt_names_coco:
        return 0
    # Map our keypoint names to COCO indices (by name match)
    name_to_idx = {n: i for i, n in enumerate(kpt_names_coco)}
    our_indices = []
    for n in keypoint_names:
        idx = name_to_idx.get(n, name_to_idx.get(n.replace("_", "")))
        if idx is not None:
            our_indices.append(idx)
        else:
            our_indices.append(-1)
    # Collect all annotations and build label rows (no write)
    rows: list[tuple[dict, list[float], str]] = []
    for ann in coco.get("annotations", []):
        keypoints = ann.get("keypoints", [])
        if not keypoints:
            continue
        image_id = ann["image_id"]
        if image_id not in images:
            continue
        im = images[image_id]
        w, h = im.get("width", 1), im.get("height", 1)
        if w <= 0 or h <= 0:
            continue
        bbox = ann.get("bbox", [0, 0, w, h])
        x_c = bbox[0] + bbox[2] / 2
        y_c = bbox[1] + bbox[3] / 2
        bw, bh = bbox[2], bbox[3]
        x_c_n = x_c / w
        y_c_n = y_c / h
        bw_n = bw / w
        bh_n = bh / h
        parts = [0, x_c_n, y_c_n, 2, bw_n, bh_n]
        for ki in our_indices:
            if ki >= 0 and ki * 3 + 2 < len(keypoints):
                x = keypoints[ki * 3] / w if keypoints[ki * 3] > 0 else 0
                y = keypoints[ki * 3 + 1] / h if keypoints[ki * 3 + 1] > 0 else 0
                v = 2 if keypoints[ki * 3 + 2] >= min_visibility else 0
            else:
                x, y, v = 0, 0, 0
            parts.extend([x, y, v])
        out_name = Path(im["file_name"]).stem + ".txt"
        rows.append((im, parts, out_name))
    return rows


def _write_rows_to_dir(rows: list[tuple[dict, list[float], str]], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for im, parts, out_name in rows:
        with open(output_dir / out_name, "w") as f:
            f.write(" ".join(f"{x:.6g}" for x in parts) + "\n")
    return len(rows)


def coco_wholebody_to_ultralytics_with_split(
    coco_annot_path: Path,
    images_dir: Path,
    train_labels_dir: Path,
    val_labels_dir: Path,
    keypoint_names: list[str],
    train_fraction: float = 0.9,
    min_visibility: float = 0.0,
    seed: int = 42,
) -> tuple[int, int]:
    """
    Convert COCO-WholeBody to Ultralytics format and split into train/val.
    Returns (num_train, num_val).
    """
    rows = coco_wholebody_to_ultralytics(
        coco_annot_path, images_dir, train_labels_dir, keypoint_names, min_visibility
    )
    if not rows:
        return 0, 0
    random.seed(seed)
    random.shuffle(rows)
    n = len(rows)
    n_train = max(1, int(n * train_fraction))
    train_rows, val_rows = rows[:n_train], rows[n_train:]
    n_t = _write_rows_to_dir(train_rows, train_labels_dir)
    n_v = _write_rows_to_dir(val_rows, val_labels_dir)
    return n_t, n_v


def create_dataset_yaml(
    output_root: Path,
    train_images: Path,
    val_images: Path,
    train_labels: Path,
    val_labels: Path,
    keypoint_names: list[str],
    path_type: str = "relative",
) -> Path:
    """Write dataset YAML for Ultralytics and return path."""
    if path_type == "absolute":
        train_im = str(train_images.resolve())
        val_im = str(val_images.resolve())
    else:
        train_im = str(train_images)
        val_im = str(val_images)
    data = {
        "path": str(output_root.resolve()),
        "train": train_im,
        "val": val_im,
        "kpt_shape": [len(keypoint_names), 3],
        "flip_idx": [3, 4, 5, 0, 1, 2],  # L<->R for 6 keypoints
        "names": {i: n for i, n in enumerate(keypoint_names)},
        "nc": 1,
    }
    yaml_path = output_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare foot keypoint dataset for YOLO pose training.")
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/foot_pose"), help="Root directory for dataset")
    ap.add_argument("--coco-annot", type=Path, default=None, help="Path to COCO-format annotations JSON")
    ap.add_argument("--images-dir", type=Path, default=None, help="Path to images (if different from dataset_dir/images)")
    ap.add_argument("--schema", type=Path, default=Path("config/keypoint_schema_cmu.yaml"), help="Keypoint schema YAML (CMU: keypoint_schema_cmu.yaml)")
    ap.add_argument("--output-dir", type=Path, default=None, help="Output for labels (default: dataset_dir/labels/train, val)")
    ap.add_argument("--train-split", type=float, default=0.9, help="Train fraction when splitting")
    ap.add_argument("--min-visibility", type=float, default=0.0, help="Min keypoint visibility to include")
    args = ap.parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir or dataset_dir
    schema_path = args.schema
    keypoint_names = load_schema_yaml(schema_path)
    if args.coco_annot and args.coco_annot.exists():
        images_dir = args.images_dir or (dataset_dir / "images")
        train_labels = output_dir / "labels" / "train"
        val_labels = output_dir / "labels" / "val"
        rows = coco_wholebody_to_ultralytics(
            args.coco_annot,
            images_dir,
            train_labels,  # used only as placeholder for path in converter
            keypoint_names,
            min_visibility=args.min_visibility,
        )
        if not rows:
            print("No annotations converted.")
        else:
            random.seed(42)
            random.shuffle(rows)
            n = len(rows)
            n_train = max(1, int(n * args.train_split))
            train_rows, val_rows = rows[:n_train], rows[n_train:]
            n_t = _write_rows_to_dir(train_rows, train_labels)
            n_v = _write_rows_to_dir(val_rows, val_labels)
            print(f"Train labels: {n_t}, Val labels: {n_v}")
        train_images = dataset_dir / "images"
        val_images = dataset_dir / "images"
        yaml_path = create_dataset_yaml(output_dir, train_images, val_images, train_labels, val_labels, keypoint_names)
        print(f"Dataset YAML: {yaml_path}")
    else:
        # User-provided dataset: expect dataset_dir/images and dataset_dir/labels or user creates them
        train_images = dataset_dir / "images" / "train"
        val_images = dataset_dir / "images" / "val"
        train_labels = dataset_dir / "labels" / "train"
        val_labels = dataset_dir / "labels" / "val"
        if not train_images.exists():
            train_images = dataset_dir / "images"
        if not val_images.exists():
            val_images = train_images
        train_labels.mkdir(parents=True, exist_ok=True)
        val_labels.mkdir(parents=True, exist_ok=True)
        yaml_path = create_dataset_yaml(output_dir, train_images, val_images, train_labels, val_labels, keypoint_names)
        print(f"No COCO annotation provided. Create images in {train_images} and labels in {train_labels}.")
        print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
