"""
Production inference pipeline: person detection + pose, temporal smoothing,
IC/TO detection, contact classification, and gait metrics.

Supports 6-keypoint (COCO-WholeBody foot) or 10-keypoint (yolo_lower) models.
Outputs: per-event CSV, per-step CSV, summary JSON, optional annotated video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

from .smoothing import smooth_keypoints, fps_aware_window_length
from .gait_events import (
    GaitEvent,
    detect_ic_to,
    events_to_contact_types,
    annotate_event_quality,
    ContactType,
)
from .gait_metrics import (
    build_step_records,
    compute_summary,
    GaitSummary,
    StepRecord,
)


# Map 10-keypoint yolo_lower to 6-keypoint: L_HEEL, L_BIG_TOE, L_SMALL_TOE, R_*
# yolo_lower: 0-1 hip, 2-3 knee, 4-5 ankle, 6-7 heel, 8-9 toe (foot)
YOLO_LOWER_TO_6 = [6, 8, 8, 7, 9, 9]  # L_HEEL=6, L_BIG_TOE=8, L_SMALL_TOE=8, R_* same


@dataclass
class PipelineConfig:
    velocity_threshold: float = 2.0
    stability_frames: int = 5
    min_step_duration_ms: float = 200.0
    smoothing_window: int = 11
    smoothing_polyorder: int = 3
    pitch_threshold_deg: float = 8.0
    stabilization_delta_ms: float = 50.0
    flat_window_ms: float = 80.0
    flat_stable_velocity: float = 1.5
    overlap_min_interfoot_px: float = 40.0
    conf_threshold: float = 0.3
    scale_m_per_px: Optional[float] = None
    fps_aware_smoothing: bool = True
    min_consecutive_bout: int = 2


def select_main_person(
    keypoints_list: list[np.ndarray],
    confidences_list: list[Optional[np.ndarray]],
    method: str = "highest_confidence",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select one person per frame; return (T, K, 3) and (T,) times.
    keypoints_list[i] shape (P, K, 3) or (K, 3), confidences_list[i] (P,) or (K,).
    """
    if not keypoints_list:
        return np.zeros((0, 6, 3)), np.zeros(0)
    out = []
    times = []
    for i, kpts in enumerate(keypoints_list):
        if kpts.ndim == 2:
            kpts = kpts[np.newaxis, ...]
        conf = confidences_list[i] if confidences_list and i < len(confidences_list) else None
        if conf is None and kpts.shape[-1] >= 3:
            conf = np.mean(kpts[..., 2], axis=1)
        if kpts.shape[0] == 0:
            continue
        if method == "highest_confidence" and conf is not None:
            idx = int(np.argmax(conf))
        else:
            idx = 0
        out.append(kpts[idx])
        times.append(i)
    if not out:
        return np.zeros((0, 6, 3)), np.zeros(0)
    return np.array(out), np.array(times, dtype=float)


def raw_to_six_keypoints(raw: np.ndarray, num_keypoints: int) -> np.ndarray:
    """Convert raw (T, K, 3) to (T, 6, 3) using mapping if K==10 (yolo_lower)."""
    T, K, _ = raw.shape
    if K >= 6:
        if K == 10:
            out = np.zeros((T, 6, 3))
            for t in range(T):
                for i, src in enumerate(YOLO_LOWER_TO_6):
                    out[t, i, :] = raw[t, src, :]
            return out
        return raw[:, :6, :].copy()
    out = np.zeros((T, 6, 3))
    out[:, :K, :] = raw
    return out


def run_pipeline(
    keypoints_ts: np.ndarray,
    times_ms: np.ndarray,
    fps: float,
    config: Optional[PipelineConfig] = None,
) -> tuple[list[GaitEvent], list[StepRecord], GaitSummary]:
    """
    Run event detection, classification, and metrics on (T, 6, 3) keypoints.
    """
    cfg = config or PipelineConfig()
    T = keypoints_ts.shape[0]
    if T == 0:
        return [], [], GaitSummary()

    window = cfg.smoothing_window
    if cfg.fps_aware_smoothing and fps > 0:
        window = fps_aware_window_length(fps, 0.15)
        if window % 2 == 0:
            window += 1
    smoothed = smooth_keypoints(
        keypoints_ts,
        method="savgol",
        window_length=window,
        polyorder=cfg.smoothing_polyorder,
        conf_threshold=cfg.conf_threshold,
    )
    events = detect_ic_to(
        smoothed,
        times_ms,
        velocity_threshold=cfg.velocity_threshold,
        stability_frames=cfg.stability_frames,
        min_step_duration_ms=cfg.min_step_duration_ms,
        conf_threshold=cfg.conf_threshold,
    )
    events = events_to_contact_types(
        smoothed,
        times_ms,
        events,
        pitch_threshold_deg=cfg.pitch_threshold_deg,
        stabilization_delta_ms=cfg.stabilization_delta_ms,
        flat_window_ms=cfg.flat_window_ms,
        flat_stable_velocity=cfg.flat_stable_velocity,
        conf_threshold=cfg.conf_threshold,
    )
    events = annotate_event_quality(
        smoothed,
        events,
        conf_threshold=cfg.conf_threshold,
        overlap_min_interfoot_px=cfg.overlap_min_interfoot_px,
    )
    step_records = build_step_records(events, scale_m_per_px=cfg.scale_m_per_px)
    duration_s = (times_ms[-1] - times_ms[0]) / 1000.0 if len(times_ms) > 1 else 0.0
    summary = compute_summary(step_records, duration_s, scale_m_per_px=cfg.scale_m_per_px)
    return events, step_records, summary


@dataclass
class FullPipelineResult:
    """Complete result from :func:`run_full_pipeline`."""
    events: list[GaitEvent] = field(default_factory=list)
    step_records: list[StepRecord] = field(default_factory=list)
    summary: Optional[GaitSummary] = None
    bout_summary: Optional[dict] = None
    severity: Optional[dict] = None
    extended_metrics: Optional[dict] = None


def run_full_pipeline(
    keypoints_ts: np.ndarray,
    times_ms: np.ndarray,
    fps: float,
    config: Optional[PipelineConfig] = None,
    full_keypoints: Optional[np.ndarray] = None,
) -> FullPipelineResult:
    """Run core pipeline + bout detection + severity + extended metrics.

    This is the single canonical entrypoint that should be used for
    paper-aligned analysis.  It calls :func:`run_pipeline` then
    layers on bout detection, severity index, and extended metrics.

    Parameters
    ----------
    keypoints_ts : ndarray (T, 6, 3)
        6-keypoint foot trajectories.
    times_ms : ndarray (T,)
    fps : float
    config : PipelineConfig, optional
    full_keypoints : ndarray (T, K, 3), optional
        Original keypoint array (10 or 26 keypoints) for COM proxy.
    """
    from .bout_detection import detect_toe_walking_bouts, compute_bout_summary
    from .severity import compute_severity_index
    from .extended_metrics import (
        compute_heel_rise_timing,
        compute_plantarflexion_proxy,
        compute_foot_strike_variability,
        compute_com_sway_proxy,
    )

    events, step_records, summary = run_pipeline(keypoints_ts, times_ms, fps, config)

    result = FullPipelineResult(
        events=events,
        step_records=step_records,
        summary=summary,
    )

    cfg = config or PipelineConfig()
    # Bouts
    bouts = detect_toe_walking_bouts(events, min_consecutive=cfg.min_consecutive_bout)
    ic_count = sum(1 for e in events if e.event_type == "IC")
    bout_summary_obj = compute_bout_summary(bouts, ic_count)
    result.bout_summary = bout_summary_obj.to_dict()

    # Severity
    sev = compute_severity_index(summary, bout_summary_obj)
    result.severity = sev.to_dict()

    # Extended metrics
    ext: dict = {}

    hr = compute_heel_rise_timing(
        keypoints_ts, times_ms, events, conf_threshold=cfg.conf_threshold
    )
    lats = [r.heel_rise_latency_ms for r in hr if r.heel_rise_latency_ms is not None]
    ext["heel_rise_latency_mean_ms"] = float(np.mean(lats)) if lats else None
    ext["heel_rise_latency_std_ms"] = float(np.std(lats, ddof=1)) if len(lats) > 1 else None

    pf = compute_plantarflexion_proxy(
        keypoints_ts, times_ms, events, conf_threshold=cfg.conf_threshold
    )
    pitches = [r.pitch_deg for r in pf if r.pitch_deg is not None]
    ext["plantarflexion_proxy_mean_deg"] = float(np.mean(pitches)) if pitches else None
    ext["plantarflexion_proxy_std_deg"] = float(np.std(pitches, ddof=1)) if len(pitches) > 1 else None

    fsv = compute_foot_strike_variability(events)
    ext["foot_strike_entropy"] = fsv.shannon_entropy
    ext["foot_strike_variability"] = fsv.to_dict()

    if full_keypoints is not None and full_keypoints.shape[1] >= 10:
        com = compute_com_sway_proxy(
            full_keypoints, hip_left_idx=0, hip_right_idx=1,
            conf_threshold=cfg.conf_threshold
        )
        ext["com_sway_proxy"] = com.to_dict()

    result.extended_metrics = ext
    return result


def events_to_dataframe(events: list[GaitEvent]) -> pd.DataFrame:
    rows = []
    for e in events:
        rows.append({
            "event_type": e.event_type,
            "foot": e.foot,
            "frame": e.frame,
            "time_ms": e.time_ms,
            "contact_type": e.contact_type.value if e.contact_type else None,
            "x": e.x,
            "y": e.y,
            "quality_score": e.quality_score,
            "overlap_flag": e.overlap_flag,
            "low_confidence_flag": e.low_confidence_flag,
            "excluded_from_primary_metrics": e.excluded_from_primary_metrics,
        })
    return pd.DataFrame(rows)


def step_records_to_dataframe(records: list[StepRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "foot": r.foot,
            "ic_time_ms": r.ic_time_ms,
            "to_time_ms": r.to_time_ms,
            "next_ic_time_ms": r.next_ic_time_ms,
            "contact_type": r.contact_type,
            "stride_time_ms": r.stride_time_ms,
            "stance_time_ms": r.stance_time_ms,
            "swing_time_ms": r.swing_time_ms,
            "step_length_px": r.step_length_px,
            "step_length_m": r.step_length_m,
            "stride_length_px": r.stride_length_px,
            "stride_length_m": r.stride_length_m,
            "quality_score": r.quality_score,
            "overlap_flag": r.overlap_flag,
            "low_confidence_flag": r.low_confidence_flag,
            "excluded_from_primary_metrics": r.excluded_from_primary_metrics,
        })
    return pd.DataFrame(rows)
