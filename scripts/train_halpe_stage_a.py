#!/usr/bin/env python3
"""
Train a HALPE-26 pose model (Stage A) with per-keypoint loss weighting.

Uses ``WeightedPoseTrainer`` to emphasise ankle/heel/toe keypoints.
Reads training hyperparameters from ``config/train_halpe_stage_a.yaml``
(or a custom YAML via ``--config``).

Example
-------
::

    # From project root, inside the poetry env:
    python scripts/train_halpe_stage_a.py
    python scripts/train_halpe_stage_a.py --config config/train_halpe_stage_a.yaml --device 0
    python scripts/train_halpe_stage_a.py --resume runs/pose/halpe26_stage_a/weights/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_project_root = Path(__file__).resolve().parent.parent
_src = _project_root / "src"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ultralytics import YOLO  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train HALPE-26 Stage A (weighted pose)")
    ap.add_argument(
        "--config", type=Path,
        default=_project_root / "config" / "train_halpe_stage_a.yaml",
        help="Training config YAML",
    )
    ap.add_argument("--device", type=str, default=None, help="Override device (0, cpu, …)")
    ap.add_argument("--resume", type=str, default=None, help="Resume from checkpoint .pt")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs")
    ap.add_argument("--batch", type=int, default=None, help="Override batch size")
    ap.add_argument("--imgsz", type=int, default=None, help="Override image size")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    model_path = cfg.pop("model", "yolo11s-pose.pt")
    if args.resume:
        model_path = args.resume
        cfg["resume"] = True

    kpt_weights = cfg.pop("kpt_weights", None)

    if args.device is not None:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.imgsz is not None:
        cfg["imgsz"] = args.imgsz

    cfg.pop("task", None)

    from gait_analysis.weighted_pose_trainer import WeightedPoseTrainer  # noqa: E402

    # Inject kpt_weights into overrides so the trainer can read them
    if kpt_weights is not None:
        WeightedPoseTrainer.custom_kpt_weights = kpt_weights

    model = YOLO(model_path)
    model.train(trainer=WeightedPoseTrainer, **cfg)

    print("\n✓ Stage A training complete.")
    print(f"  Best weights: {cfg.get('project', 'runs/pose')}/{cfg.get('name', 'halpe26_stage_a')}/weights/best.pt")


if __name__ == "__main__":
    main()
