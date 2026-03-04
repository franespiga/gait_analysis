#!/usr/bin/env python3
"""
Production inference: run person detection + pose on video, temporal smoothing,
gait event detection (IC/TO), contact classification, and output per-event CSV,
per-step CSV, and summary JSON. Optional annotated video.

Usage:
  python scripts/infer_video.py path/to/video.mp4
  python scripts/infer_video.py path/to/video.mp4 --config config/inference.yaml --output-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Project root and src on path
_project_root = Path(__file__).resolve().parent.parent
_src = _project_root / "src"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gait_analysis.inference_pipeline import (
    PipelineConfig,
    run_pipeline,
    raw_to_six_keypoints,
    events_to_dataframe,
    step_records_to_dataframe,
    YOLO_LOWER_TO_6,
)


def load_inference_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def build_pipeline_config(cfg: dict) -> PipelineConfig:
    ev = cfg.get("events", {})
    sm = cfg.get("smoothing", {})
    contact = cfg.get("contact", {})
    cal = cfg.get("calibration", {})
    scale = None
    if cal.get("scale_factor_m_per_px") is not None:
        scale = float(cal["scale_factor_m_per_px"])
    elif cal.get("reference_height_px") is not None and cal.get("reference_height_m"):
        ref_px = float(cal["reference_height_px"])
        ref_m = float(cal["reference_height_m"])
        if ref_px > 0:
            scale = ref_m / ref_px
    return PipelineConfig(
        velocity_threshold=float(ev.get("velocity_threshold", 2.0)),
        stability_frames=int(ev.get("stability_frames", 5)),
        min_step_duration_ms=float(ev.get("min_step_duration_ms", 200.0)),
        smoothing_window=int(sm.get("window_length", 11)),
        smoothing_polyorder=int(sm.get("polyorder", 3)),
        pitch_threshold_deg=float(contact.get("pitch_threshold_deg", 8.0)),
        stabilization_delta_ms=float(contact.get("stabilization_delta_ms", 50.0)),
        flat_window_ms=float(contact.get("flat_window_ms", 80.0)),
        conf_threshold=float(cfg.get("conf_threshold", 0.3)),
        scale_m_per_px=scale,
        fps_aware_smoothing=bool(ev.get("fps_aware", True)),
    )


def extract_keypoints_from_video(
    video_path: Path,
    model_path: str,
    conf_threshold: float = 0.5,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run YOLO pose on video; return (keypoints T,K,3), times_ms (T,), fps.
    Select main person per frame (highest keypoint confidence).
    """
    from ultralytics import YOLO
    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    keypoints_list = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False, conf=conf_threshold, device=device)
        kpts = None
        confs = None
        if results and len(results) > 0 and results[0].keypoints is not None:
            xy = results[0].keypoints.xy
            c = results[0].keypoints.conf
            if xy is not None and len(xy) > 0:
                xy = xy.cpu().numpy()
                c = c.cpu().numpy() if c is not None else np.ones((xy.shape[0], xy.shape[1]))
                # Main person: highest mean confidence
                mean_conf = np.mean(c, axis=1)
                idx = int(np.argmax(mean_conf))
                kpts = xy[idx]  # (K, 2)
                confs = c[idx]   # (K,)
        if kpts is not None:
            # (K, 3) with x, y, conf
            K = kpts.shape[0]
            out = np.zeros((K, 3), dtype=np.float64)
            out[:, :2] = kpts
            out[:, 2] = confs if confs is not None else 0.5
            keypoints_list.append(out)
        else:
            keypoints_list.append(np.zeros((6, 3), dtype=np.float64))
        frame_idx += 1
    cap.release()
    if not keypoints_list:
        return np.zeros((0, 6, 3)), np.zeros(0), fps
    stacked = np.array(keypoints_list)
    T, K, _ = stacked.shape
    # Map to 6 keypoints if 10 (yolo_lower)
    if K == 10:
        six_kpt = np.zeros((T, 6, 3))
        for t in range(T):
            for i, src in enumerate(YOLO_LOWER_TO_6):
                six_kpt[t, i, :] = stacked[t, src, :]
        stacked = six_kpt
    elif K != 6:
        stacked = stacked[:, :6, :].copy()
    times_ms = (np.arange(T, dtype=float) / fps) * 1000.0
    return stacked, times_ms, fps


def draw_annotated_frame(
    frame: np.ndarray,
    keypoints: np.ndarray,
    events_this_frame: list,
    step_records: list,
) -> np.ndarray:
    """Draw keypoints and optional event markers on frame. Simple overlay."""
    out = frame.copy()
    if keypoints.size == 0:
        return out
    K = keypoints.shape[0]
    for k in range(K):
        x, y = keypoints[k, 0], keypoints[k, 1]
        c = keypoints[k, 2] if keypoints.shape[-1] > 2 else 1.0
        if c < 0.2:
            continue
        ix, iy = int(x), int(y)
        cv2.circle(out, (ix, iy), 5, (0, 255, 0), -1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run gait analysis on video: pose → smooth → IC/TO → metrics → CSV + JSON."
    )
    ap.add_argument("video", type=Path, help="Input video (mp4, mov, etc.)")
    ap.add_argument("--config", type=Path, default=Path("config/inference.yaml"), help="Inference config YAML")
    ap.add_argument("--model", type=str, default=None, help="Override model path (pose .pt)")
    ap.add_argument("--output-dir", type=Path, default=None, help="Output directory (default from config)")
    ap.add_argument("--no-events-csv", action="store_true", help="Skip per-event CSV")
    ap.add_argument("--no-steps-csv", action="store_true", help="Skip per-step CSV")
    ap.add_argument("--no-summary-json", action="store_true", help="Skip summary JSON")
    ap.add_argument("--annotated-video", type=Path, default=None, help="Save annotated video to this path")
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    cfg = load_inference_config(args.config)
    model_path = args.model or cfg.get("model_path") or "yolov8m-pose.pt"
    # Prefer 6-keypoint foot model; fallback to yolo_lower if path exists
    foot_pose_path = Path("models/foot_pose/best.pt")
    yolo_lower_path = Path("models/yolo_lower/best.pt")
    if args.model is None:
        if foot_pose_path.exists():
            model_path = str(foot_pose_path)
        elif yolo_lower_path.exists():
            model_path = str(yolo_lower_path)

    output_dir = args.output_dir or Path(cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem

    pipeline_cfg = build_pipeline_config(cfg)
    conf_threshold = float(cfg.get("conf_threshold", 0.5))
    device = args.device or cfg.get("device", "auto")

    print(f"Loading model: {model_path}")
    keypoints_ts, times_ms, fps = extract_keypoints_from_video(
        args.video,
        model_path,
        conf_threshold=conf_threshold,
        device=device,
    )
    T = keypoints_ts.shape[0]
    print(f"Frames: {T}, FPS: {fps:.2f}")

    if T == 0:
        print("No frames processed. Exiting.")
        return

    events, step_records, summary = run_pipeline(
        keypoints_ts,
        times_ms,
        fps,
        config=pipeline_cfg,
    )

    # Outputs
    if not args.no_events_csv and cfg.get("save_per_event_csv", True):
        df_ev = events_to_dataframe(events)
        events_path = output_dir / f"{stem}_events.csv"
        df_ev.to_csv(events_path, index=False)
        print(f"Events CSV: {events_path}")

    if not args.no_steps_csv and cfg.get("save_per_step_csv", True):
        df_steps = step_records_to_dataframe(step_records)
        steps_path = output_dir / f"{stem}_steps.csv"
        df_steps.to_csv(steps_path, index=False)
        print(f"Steps CSV: {steps_path}")

    if not args.no_summary_json and cfg.get("save_summary_json", True):
        summary_path = output_dir / f"{stem}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"Summary JSON: {summary_path}")

    if args.annotated_video or cfg.get("save_annotated_video", False):
        out_video = args.annotated_video or (output_dir / f"{stem}_annotated.mp4")
        cap = cv2.VideoCapture(str(args.video))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
        events_by_frame = {}
        for e in events:
            events_by_frame.setdefault(e.frame, []).append(e)
        for t in range(T):
            ret, frame = cap.read()
            if not ret:
                break
            kpt = keypoints_ts[t]
            ev_list = events_by_frame.get(t, [])
            frame = draw_annotated_frame(frame, kpt, ev_list, step_records)
            writer.write(frame)
        cap.release()
        writer.release()
        print(f"Annotated video: {out_video}")

    print("Done.")
    print(f"  Total steps: {summary.total_steps}")
    print(f"  Cadence: {summary.cadence_steps_per_min:.1f} steps/min")
    print(f"  Step time (mean): {summary.step_time_mean_ms:.1f} ms")


if __name__ == "__main__":
    main()
