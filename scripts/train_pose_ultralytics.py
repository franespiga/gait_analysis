#!/usr/bin/env python3
"""
Ultralytics-aligned training for YOLOv8 pose / keypoint fine-tuning.

Follows Ultralytics Train mode for device handling, resume, and batch:
- device: 0, [0,1], -1 (idle GPU), mps (Apple Silicon), cpu, or auto
- resume: loads last.pt and calls model.train(resume=True)
- batch: int, -1 (auto 60% GPU), or float (e.g. 0.7 for utilization fraction)

See: https://docs.ultralytics.com/modes/train/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Union

import yaml
from ultralytics import YOLO


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_device(
    value: Union[str, int, List[int], None],
) -> Union[int, List[int], str, None]:
    """
    Parse device for Ultralytics: int (0, -1), list ([0,1], [-1,-1]), or str (cpu, mps, cuda).
    Returns None for auto (let Ultralytics choose).
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if s in ("", "auto"):
        return None
    if s in ("cpu", "cuda", "mps"):
        return s
    if "," in s:
        parts = [x.strip() for x in s.split(",") if x.strip()]
        return [int(x) for x in parts]
    return int(s)


def parse_batch(value: Any) -> Union[int, float]:
    """Parse batch: int (16, -1) or float (0.7 for utilization)."""
    if value is None:
        return 16
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip()
    if "." in s:
        return float(s)
    return int(s)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train YOLOv8 pose (keypoint) with Ultralytics-aligned device/resume/batch."
    )
    ap.add_argument("--config", type=Path, default=Path("config/training.yaml"), help="Training config YAML")
    ap.add_argument("--data", type=Path, default=None, help="Dataset YAML path (overrides config)")
    ap.add_argument("--model", type=str, default=None, help="Base model (e.g. yolov8m-pose.pt)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=str, default=None, help="Batch size: 16, -1 (auto), or 0.7 (utilization)")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--project", type=Path, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint (resume=True)")
    ap.add_argument("--export-onnx", action="store_true", help="Export best.pt to ONNX after training")
    ap.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: 0, 0,1, -1 (idle GPU), mps (Apple Silicon), cpu, or auto",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = args.data or Path(cfg.get("dataset_yaml") or cfg.get("data") or "data/foot_pose/data.yaml")
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}. Run scripts/prepare_dataset.py first.")
    data = str(data.resolve())

    model_name = args.model or cfg.get("model", "yolov8n-pose.pt")
    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 100)
    batch_cfg = args.batch if args.batch is not None else cfg.get("batch", 16)
    batch = parse_batch(batch_cfg)
    imgsz = args.imgsz if args.imgsz is not None else cfg.get("imgsz", 320)
    project = Path(args.project or cfg.get("project", "runs/pose"))
    name = str(args.name or cfg.get("name", "foot_pose"))
    resume = args.resume or cfg.get("resume", False)
    export_onnx = args.export_onnx or cfg.get("export_onnx", False)

    device_raw = (args.device or cfg.get("device") or "auto").strip() or "auto"
    device = parse_device(device_raw)

    if resume:
        resume_path = project / name / "weights" / "last.pt"
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume requested but no checkpoint at {resume_path}. Run without --resume first."
            )
        model = YOLO(str(resume_path))
        train_kw: dict[str, Any] = {
            "resume": True,
            "data": data,
        }
        if device is not None:
            train_kw["device"] = device
        model.train(**train_kw)
    else:
        model = YOLO(model_name)
        train_kw = dict(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=str(project),
            name=name,
            cache='ram', 
            workers=0,
            amp=True,
            freeze=10, 
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
            mosaic=cfg.get("mosaic", 0.0),
            mixup=cfg.get("mixup", 0.0),
        )
        if device is not None:
            train_kw["device"] = device
        model.train(**train_kw)

    if export_onnx:
        export_dir = Path(cfg.get("export_dir", "runs/pose/export"))
        export_dir.mkdir(parents=True, exist_ok=True)
        best = project / name / "weights" / "best.pt"
        if best.exists():
            m = YOLO(str(best))
            m.export(format="onnx", imgsz=imgsz, dir=str(export_dir))


if __name__ == "__main__":
    main()
