#!/usr/bin/env python3
"""
Train a HALPE-26 foot-crop pose model (Stage B safeguard).

Uses ``WeightedPoseTrainer`` on the cropped dataset produced by
``scripts/build_foot_crops.py``.

Example
-------
::

    python scripts/train_halpe_stage_b.py
    python scripts/train_halpe_stage_b.py --config config/train_halpe_stage_b.yaml --device 0
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
    ap = argparse.ArgumentParser(description="Train HALPE-26 Stage B (foot crop)")
    ap.add_argument(
        "--config", type=Path,
        default=_project_root / "config" / "train_halpe_stage_b.yaml",
        help="Training config YAML",
    )
    ap.add_argument("--device", type=str, default=None, help="Override device")
    ap.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    model_path = cfg.pop("model", "yolo11n-pose.pt")
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

    cfg.pop("task", None)

    from gait_analysis.weighted_pose_trainer import WeightedPoseTrainer  # noqa: E402

    if kpt_weights is not None:
        cfg["kpt_weights"] = kpt_weights

    model = YOLO(model_path)
    model.train(trainer=WeightedPoseTrainer, **cfg)

    print("\n✓ Stage B training complete.")
    print(f"  Best weights: {cfg.get('project', 'runs/pose')}/{cfg.get('name', 'halpe26_stage_b_foot')}/weights/best.pt")


if __name__ == "__main__":
    main()
