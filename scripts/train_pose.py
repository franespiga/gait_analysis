#!/usr/bin/env python3
"""
Train YOLOv8 pose model with foot keypoints.

- Configurable hyperparameters via YAML and CLI.
- Supports resume and fine-tune.
- Optional evaluation and ONNX export.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import yaml
from ultralytics import YOLO


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLOv8 pose model (foot keypoints).")
    ap.add_argument("--config", type=Path, default=Path("config/training.yaml"), help="Training config YAML")
    ap.add_argument("--data", type=Path, default=None, help="Dataset YAML path (overrides config)")
    ap.add_argument("--model", type=str, default=None, help="Base model (e.g. yolov8m-pose.pt)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--project", type=Path, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    ap.add_argument("--export-onnx", action="store_true", help="Export to ONNX after training")
    ap.add_argument("--device", type=str, default="", help="Device: 0, cuda, or auto (default: auto = use GPU if available)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = args.data or Path(cfg.get("dataset_yaml") or cfg.get("data") or "data/foot_pose/data.yaml")
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}. Run scripts/prepare_dataset.py first.")
    data = str(data.resolve())
    model_name = args.model or cfg.get("model", "yolov8m-pose.pt")
    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 100)
    batch = args.batch if args.batch is not None else cfg.get("batch", 16)
    imgsz = args.imgsz if args.imgsz is not None else cfg.get("imgsz", 960)
    project = Path(args.project or cfg.get("project", "runs/pose"))
    name = str(args.name or cfg.get("name", "foot_pose"))
    resume = args.resume or cfg.get("resume", False)
    export_onnx = args.export_onnx or cfg.get("export_onnx", False)
    device = (args.device or cfg.get("device") or "auto").strip() or "auto"
    
    import torch 
    print(torch.cuda.is_available())
    print(torch.__version__)
    
    print(torch.version.cuda)
    if torch.cuda.device_count():
       print(torch.cuda.get_device_name(0))
    print("TRAINING WITH DEVICE: ", device)

    model = YOLO(model_name)
    train_args = dict(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(project),
        name=name,
        exist_ok=cfg.get("exist_ok", False),
        pretrained=cfg.get("pretrained", True),
        patience=cfg.get("patience", 20),
        optimizer=cfg.get("optimizer", "auto"),
        lr0=cfg.get("lr0", 0.01),
        lrf=cfg.get("lrf", 0.01),
        momentum=cfg.get("momentum", 0.937),
        weight_decay=cfg.get("weight_decay", 0.0005),
        warmup_epochs=cfg.get("warmup_epochs", 3.0),
        warmup_momentum=cfg.get("warmup_momentum", 0.8),
        warmup_bias_lr=cfg.get("warmup_bias_lr", 0.1),
        hsv_h=cfg.get("hsv_h", 0.015),
        hsv_s=cfg.get("hsv_s", 0.7),
        hsv_v=cfg.get("hsv_v", 0.4),
        degrees=cfg.get("degrees", 0.0),
        translate=cfg.get("translate", 0.1),
        scale=cfg.get("scale", 0.5),
        shear=cfg.get("shear", 0.0),
        perspective=cfg.get("perspective", 0.0001),
        fliplr=cfg.get("fliplr", 0.5),
        mosaic=cfg.get("mosaic", 1.0),
        mixup=cfg.get("mixup", 0.0),
        device=device,
    )
    if resume:
        resume_path = project / name / "weights" / "last.pt"
        if resume_path.exists():
            model = YOLO(str(resume_path))
            print(f"Resuming from {resume_path}")
    model.train(**train_args)
    if export_onnx:
        export_dir = Path(cfg.get("export_dir", "runs/pose/export"))
        export_dir.mkdir(parents=True, exist_ok=True)
        best = project / name / "weights" / "best.pt"
        if best.exists():
            m = YOLO(str(best))
            m.export(format="onnx", imgsz=imgsz, dir=export_dir)
            print(f"Exported ONNX to {export_dir}")


if __name__ == "__main__":
    main()
