"""
Streamlit app for real-time and video gait analysis.

- Sidebar: choose Upload video or Webcam; select file or camera.
- Model: select pose model (YOLO COCO nano/small/medium or YOLO Lower Body if available).
- Start Gait Analysis: run pose estimation and show keypoints (skeleton + points) on the video.
- Legend (top-right): frames, steps, steps/min, average step length.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
# Ensure package is importable when run from app/ or project root
_project_root = Path(__file__).resolve().parent.parent
_src = _project_root / "src"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gait_analysis.base_detector import GaitKeypoints
from gait_analysis.advanced_visualizer import AdvancedGaitVisualizer
from gait_analysis.multi_backend_main import create_detector, create_analyzer


# Built-in model options: (label, backend, model_path, conf_thresh).
# Additional custom models are discovered dynamically from models/*/weights/best.pt.
MODEL_OPTIONS = [
    ("YOLO COCO (nano, fast)", "yolo_coco", "yolov8n-pose.pt", 0.5),
    ("YOLO COCO (small)", "yolo_coco", "yolov8s-pose.pt", 0.5),
    ("YOLO COCO (medium)", "yolo_coco", "yolov8m-pose.pt", 0.5),
    ("YOLO Lower Body (heel/toe)", "yolo_lower", str(_project_root / "models" / "yolo_lower" / "best.pt"), 0.25),
]
DEFAULT_MODEL_INDEX = 0


def _running_step_length_px(analyzer) -> float | None:
    """Compute average step length (pixels) from current steps. Returns None if < 2 steps."""
    step_lengths = []
    for tracker_name in ("left_tracker", "right_tracker"):
        tracker = getattr(analyzer, tracker_name, None)
        if tracker is None:
            continue
        steps = getattr(tracker, "steps", [])
        for i in range(1, len(steps)):
            a, b = steps[i - 1], steps[i]
            ax = getattr(a, "contact_ankle_x", None)
            ay = getattr(a, "contact_ankle_y", None)
            bx = getattr(b, "contact_ankle_x", None)
            by = getattr(b, "contact_ankle_y", None)
            if ax is not None and ay is not None and bx is not None and by is not None:
                step_lengths.append(math.hypot(bx - ax, by - ay))
    if not step_lengths:
        return None
    return sum(step_lengths) / len(step_lengths)


def _draw_metrics_legend(
    frame,
    frame_count: int,
    total_steps: int,
    elapsed_sec: float,
    avg_step_length_px: float | None,
):
    """Draw metrics box in top-right corner (BGR frame modified in place)."""
    h, w = frame.shape[:2]
    box_w, box_x = 260, w - 270
    box_y, box_h = 10, 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (200, 200, 200), 1)
    y_off = box_y + 24
    line_h = 22
    color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    cv2.putText(frame, f"Frames: {frame_count}", (box_x + 8, y_off), font, scale, color, 1)
    y_off += line_h
    cv2.putText(frame, f"Steps: {total_steps}", (box_x + 8, y_off), font, scale, color, 1)
    y_off += line_h
    steps_per_min = (total_steps / elapsed_sec * 60) if elapsed_sec > 0 else 0
    cv2.putText(frame, f"Steps/min: {steps_per_min:.1f}", (box_x + 8, y_off), font, scale, color, 1)
    y_off += line_h
    step_len_str = f"{avg_step_length_px:.1f} px" if avg_step_length_px is not None else "—"
    cv2.putText(frame, f"Avg step length: {step_len_str}", (box_x + 8, y_off), font, scale, color, 1)


def _gait_keypoints_to_frame_keypoints(keypoints: GaitKeypoints):
    """Build FrameKeypoints-like object for visualizer (raw_keypoints only)."""
    from gait_analysis.detector import FrameKeypoints
    raw = keypoints.raw_keypoints
    if raw is not None and hasattr(raw, "shape") and len(raw.shape) >= 2:
        raw = np.asarray(raw, dtype=np.float64)
    return FrameKeypoints(
        frame_number=keypoints.frame_number,
        timestamp_ms=keypoints.timestamp_ms,
        keypoints={},
        raw_keypoints=raw
    )


def _draw_keypoints_fallback(frame: np.ndarray, raw_keypoints: np.ndarray, conf_min: float = 0.2) -> None:
    """Ensure keypoints are visible: draw circles for each point (BGR frame in place)."""
    if raw_keypoints is None or raw_keypoints.size == 0:
        return
    kpts = np.asarray(raw_keypoints)
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        return
    for i in range(len(kpts)):
        x, y = float(kpts[i, 0]), float(kpts[i, 1])
        conf = float(kpts[i, 2]) if kpts.shape[1] > 2 else 1.0
        if conf < conf_min or x <= 0 or y <= 0:
            continue
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(frame, (ix, iy), 6, (0, 255, 0), 2)
        cv2.circle(frame, (ix, iy), 2, (0, 255, 255), -1)


def _discover_models_dir() -> list[tuple[str, str, str, float]]:
    """
    Discover custom models under models/ directory.

    Each top-level folder under models/ becomes an option if it contains
    weights/best.pt. The option label is "<folder> (models/<folder>)".
    """
    models_root = _project_root / "models"
    if not models_root.exists():
        return []
    out: list[tuple[str, str, str, float]] = []
    for sub in sorted(p for p in models_root.iterdir() if p.is_dir()):
        weights = sub / "weights" / "best.pt"
        if not weights.exists():
            continue
        name = sub.name
        label = f"{name} (models/{name})"
        backend = "yolo_lower" if "lower" in name.lower() else "yolo_coco"
        conf = 0.25 if backend == "yolo_lower" else 0.3
        out.append((label, backend, str(weights), conf))
    return out


def _get_available_models() -> list[tuple[str, str, str, float]]:
    """
    Return list of (label, backend, model_path, conf_thresh) for models that are available.

    Includes built-in presets and all models discovered under models/*/weights/best.pt.
    """
    out: list[tuple[str, str, str, float]] = []

    # Built-in options (filter out ones whose paths are missing, e.g. yolo_lower weights)
    for label, backend, model_path, conf in MODEL_OPTIONS:
        # For file-backed models, require file to exist
        if ("/" in str(model_path) or "\\" in str(model_path)) and Path(str(model_path)).suffix == ".pt":
            if not Path(str(model_path)).exists():
                continue
        out.append((label, backend, str(model_path), conf))

    # Custom models under models/*/weights/best.pt
    out.extend(_discover_models_dir())

    # Fallback to COCO presets if nothing else is available
    if not out:
        out = [
            (label, backend, str(model_path), conf)
            for label, backend, model_path, conf in MODEL_OPTIONS
            if backend == "yolo_coco"
        ]
    return out


@st.cache_resource
def _load_detector_and_analyzer(backend: str, model_path: str):
    """Load detector and analyzer once and reuse.

    Tries to load a YOLOv8 pose model first; if that fails for a built-in
    model name (e.g. 'yolov8n-pose.pt'), it retries with the corresponding
    YOLO11 pose name (e.g. 'yolo11n-pose.pt').
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        detector = create_detector(backend, model_path, device=device)
    except Exception as e:
        alt_model_path = None
        # Only attempt fallback for non-path model names (no slashes)
        if "yolov8" in model_path and ("/" not in model_path and "\\" not in model_path):
            alt_model_path = model_path.replace("yolov8", "yolo11")
        elif "yolo11" in model_path and ("/" not in model_path and "\\" not in model_path):
            alt_model_path = model_path.replace("yolo11", "yolov8")
        if alt_model_path is None:
            raise
        detector = create_detector(backend, alt_model_path, device=device)
    analyzer = create_analyzer(detector)
    visualizer = AdvancedGaitVisualizer(show_angles=True, show_skeleton=True, show_stats=True)
    return detector, analyzer, visualizer


def _run_video_analysis(video_path: str, detector, analyzer, visualizer, backend: str, conf_thresh: float):
    """Process video file and yield (annotated_frame_bgr, frame_count, analysis_state, elapsed_sec)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_number = 0
    analysis_state = None
    start_time = None
    import time
    start_time = time.perf_counter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_ms = (frame_number / fps) * 1000 if fps > 0 else 0
            keypoints = detector.detect(frame, frame_number, timestamp_ms, conf_threshold=conf_thresh)
            if keypoints is not None:
                analysis_state = analyzer.process_frame(keypoints)
            viz_results = detector.get_visualization_results(frame)
            frame_kpts = None
            if keypoints is not None and keypoints.raw_keypoints is not None:
                frame_kpts = _gait_keypoints_to_frame_keypoints(keypoints)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            annotated = visualizer.draw_frame(
                frame, frame_kpts, analysis_state, viz_results,
                frame_number, total_frames, backend=backend
            )
            if keypoints is not None and keypoints.raw_keypoints is not None:
                _draw_keypoints_fallback(annotated, keypoints.raw_keypoints, conf_min=0.2)
            elapsed = time.perf_counter() - start_time
            total_steps = 0
            if analysis_state:
                total_steps = analysis_state.get("left_steps", 0) + analysis_state.get("right_steps", 0)
            avg_step_len = _running_step_length_px(analyzer)
            _draw_metrics_legend(annotated, frame_number, total_steps, elapsed, avg_step_len)
            yield cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), frame_number, analysis_state, elapsed
            frame_number += 1
    finally:
        cap.release()


def _run_webcam_analysis(camera_index: int, detector, analyzer, visualizer, backend: str, conf_thresh: float, stop_flag: list):
    """Generate frames from webcam with gait overlay. stop_flag is a list; set stop_flag[0]=True to stop."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_number = 0
    analysis_state = None
    import time
    start_time = time.perf_counter()
    try:
        while not (stop_flag and stop_flag[0]):
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_ms = (frame_number / fps) * 1000 if fps > 0 else 0
            keypoints = detector.detect(frame, frame_number, timestamp_ms, conf_threshold=conf_thresh)
            if keypoints is not None:
                analysis_state = analyzer.process_frame(keypoints)
            viz_results = detector.get_visualization_results(frame)
            frame_kpts = None
            if keypoints is not None and keypoints.raw_keypoints is not None:
                frame_kpts = _gait_keypoints_to_frame_keypoints(keypoints)
            annotated = visualizer.draw_frame(
                frame, frame_kpts, analysis_state, viz_results,
                frame_number, 0, backend=backend
            )
            if keypoints is not None and keypoints.raw_keypoints is not None:
                _draw_keypoints_fallback(annotated, keypoints.raw_keypoints, conf_min=0.2)
            elapsed = time.perf_counter() - start_time
            total_steps = 0
            if analysis_state:
                total_steps = analysis_state.get("left_steps", 0) + analysis_state.get("right_steps", 0)
            avg_step_len = _running_step_length_px(analyzer)
            _draw_metrics_legend(annotated, frame_number, total_steps, elapsed, avg_step_len)
            yield cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), frame_number, analysis_state, elapsed
            frame_number += 1
    finally:
        cap.release()


def _list_webcam_indices(max_try: int = 5) -> list[int]:
    """Return list of camera indices that open successfully."""
    indices = []
    for i in range(max_try):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            indices.append(i)
            cap.release()
    return indices


def main():
    st.set_page_config(page_title="Gait Analysis", layout="wide")
    st.title("Gait Analysis")
    st.caption("Pose estimation from waist down with step count, steps/min, and step length.")

    sidebar = st.sidebar
    sidebar.header("Input")
    input_type = sidebar.radio("Source", ["Upload video", "Webcam"], index=0)

    video_path = None
    camera_index = 0
    cameras: list[int] = []
    if input_type == "Upload video":
        uploaded = sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as f:
                f.write(uploaded.read())
                video_path = f.name
    else:
        cameras = _list_webcam_indices()
        if not cameras:
            sidebar.warning("No webcams detected.")
        elif len(cameras) == 1:
            sidebar.info("Using camera 0")
            camera_index = 0
        else:
            camera_index = sidebar.selectbox("Webcam", cameras, index=0)

    sidebar.header("Model")
    available = _get_available_models()
    model_labels = [m[0] for m in available]
    default_idx = min(DEFAULT_MODEL_INDEX, len(available) - 1) if available else 0
    selected_label = sidebar.selectbox(
        "Pose model",
        model_labels,
        index=default_idx,
        help="Built-in YOLO COCO presets plus any models found under models/*/weights/best.pt.",
    )
    selected = next((m for m in available if m[0] == selected_label), available[0])
    _, backend, model_path, conf_thresh = selected

    # Initialize or retrieve model state in session
    if "gait_model_state" not in st.session_state:
        st.session_state["gait_model_state"] = {
            "bundle": None,
            "backend": None,
            "model_path": None,
            "conf_thresh": None,
            "label": None,
        }
    model_state = st.session_state["gait_model_state"]

    start_btn = sidebar.button("Start Gait Analysis")
    sidebar.markdown("---")
    load_btn = sidebar.button("Load model")

    if load_btn:
        try:
            detector, analyzer, visualizer = _load_detector_and_analyzer(backend, model_path)
        except Exception as e:
            model_state["bundle"] = None
            st.error(f"Failed to load model '{selected_label}': {e}")
        else:
            model_state["bundle"] = (detector, analyzer, visualizer)
            model_state["backend"] = backend
            model_state["model_path"] = model_path
            model_state["conf_thresh"] = conf_thresh
            model_state["label"] = selected_label
            sidebar.success(f"Model loaded: {selected_label}")

    if not start_btn:
        st.info("Select input, choose a model, click **Load model**, then click **Start Gait Analysis**.")
        return

    if input_type == "Upload video" and not video_path:
        st.error("Please upload a video first.")
        return
    if input_type == "Webcam" and not cameras:
        st.error("No webcam available.")
        return

    if not model_state.get("bundle"):
        st.error("Please choose a model and click **Load model** in the sidebar before starting analysis.")
        return

    detector, analyzer, visualizer = model_state["bundle"]
    backend = model_state.get("backend", backend)
    conf_thresh = model_state.get("conf_thresh", conf_thresh)

    place = st.empty()
    stop_placeholder = st.sidebar.empty()
    stop_flag = [False]  # mutable so generator can see when to stop

    if input_type == "Upload video":
        gen = _run_video_analysis(video_path, detector, analyzer, visualizer, backend, conf_thresh)
        try:
            stop_btn = stop_placeholder.button("Stop", key="stop_btn")
            for rgb_frame, frame_count, state, elapsed in gen:
                if stop_btn:
                    break
                place.image(rgb_frame, channels="RGB", use_container_width=True)
        finally:
            gen.close()
        if video_path and Path(video_path).exists():
            try:
                Path(video_path).unlink()
            except Exception:
                pass
    else:
        gen = _run_webcam_analysis(camera_index, detector, analyzer, visualizer, backend, conf_thresh, stop_flag)
        try:
            stop_btn = stop_placeholder.button("Stop", key="stop_btn")
            for rgb_frame, frame_count, state, elapsed in gen:
                if stop_flag[0]:
                    break
                if stop_btn:
                    stop_flag[0] = True
                    gen.close()
                    break
                place.image(rgb_frame, channels="RGB", use_container_width=True)
        finally:
            try:
                gen.close()
            except Exception:
                pass

    place.empty()
    st.success("Analysis stopped.")


if __name__ == "__main__":
    main()
