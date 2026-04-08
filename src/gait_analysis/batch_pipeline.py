"""
Batch (multi-video) gait analysis pipeline with metadata support.

Processes a manifest CSV of videos, runs the full analysis pipeline on
each, and aggregates results per trial and per participant.

Manifest CSV format
-------------------
::

    participant_id,trial_id,video_path,condition,group,notes
    P001,T01,videos/P001_trial1.mp4,barefoot,ASD,
    P001,T02,videos/P001_trial2.mp4,shoes,ASD,
    P002,T01,videos/P002_trial1.mp4,barefoot,control,

Only ``video_path`` is required.  All other columns are carried through
to the output and used for grouping / aggregation when present.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .inference_pipeline import (
    PipelineConfig,
    run_pipeline,
    events_to_dataframe,
    step_records_to_dataframe,
)
from .bout_detection import detect_toe_walking_bouts, compute_bout_summary
from .severity import compute_severity_index
from .extended_metrics import (
    compute_heel_rise_timing,
    compute_plantarflexion_proxy,
    compute_foot_strike_variability,
    compute_com_sway_proxy,
)
from .gait_events import GaitEvent
from .gait_metrics import GaitSummary, StepRecord

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Full analysis result for a single video / trial."""
    video_path: str
    participant_id: str = ""
    trial_id: str = ""
    condition: str = ""
    group: str = ""
    notes: str = ""

    events: list[GaitEvent] = field(default_factory=list)
    step_records: list[StepRecord] = field(default_factory=list)
    summary: Optional[GaitSummary] = None
    bout_summary: Optional[dict] = None
    severity: Optional[dict] = None
    extended: Optional[dict] = None
    error: Optional[str] = None

    def summary_row(self) -> dict:
        """Flat dict suitable for a summary CSV / DataFrame."""
        row: dict = {
            "video_path": self.video_path,
            "participant_id": self.participant_id,
            "trial_id": self.trial_id,
            "condition": self.condition,
            "group": self.group,
        }
        if self.summary:
            row.update(self.summary.to_dict())
        if self.bout_summary:
            for k, v in self.bout_summary.items():
                if k != "bouts":
                    row[f"bout_{k}"] = v
        if self.severity:
            row["severity_overall"] = self.severity.get("overall")
            row["severity_class"] = self.severity.get("classification")
        if self.error:
            row["error"] = self.error
        return row


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load a manifest CSV into a list of dicts."""
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _extract_keypoints_from_video(
    video_path: Path,
    model_path: str,
    conf_threshold: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Thin wrapper that imports and calls the extraction function."""
    import sys
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.infer_video import extract_keypoints_from_video
    return extract_keypoints_from_video(
        video_path, model_path, conf_threshold=conf_threshold, device=device
    )


def run_single_trial(
    video_path: Path,
    model_path: str,
    config: PipelineConfig,
    device: str = "auto",
    conf_threshold: float = 0.25,
    metadata: Optional[dict] = None,
    halpe26_to_6: Optional[list[int]] = None,
) -> TrialResult:
    """Run the full pipeline on one video and return structured results.

    Parameters
    ----------
    video_path : Path
        Input video file.
    model_path : str
        YOLO pose model weights.
    config : PipelineConfig
        Pipeline configuration (thresholds, smoothing, calibration).
    device : str
        Inference device.
    conf_threshold : float
        Keypoint confidence threshold for extraction.
    metadata : dict, optional
        Manifest row with participant_id, trial_id, condition, group.
    halpe26_to_6 : list[int], optional
        Mapping from HALPE-26 keypoints to 6-keypoint foot schema.
        Defaults to ``[24, 20, 22, 25, 21, 23]``.
    """
    meta = metadata or {}
    result = TrialResult(
        video_path=str(video_path),
        participant_id=meta.get("participant_id", ""),
        trial_id=meta.get("trial_id", ""),
        condition=meta.get("condition", ""),
        group=meta.get("group", ""),
        notes=meta.get("notes", ""),
    )

    try:
        keypoints_ts, times_ms, fps = _extract_keypoints_from_video(
            video_path, model_path, conf_threshold, device
        )
    except Exception as exc:
        result.error = f"Extraction failed: {exc}"
        logger.error("Video %s: %s", video_path, result.error)
        return result

    T = keypoints_ts.shape[0]
    if T == 0:
        result.error = "No frames extracted"
        return result

    # Map to 6-keypoint foot schema
    from .inference_pipeline import raw_to_six_keypoints, YOLO_LOWER_TO_6
    h26 = halpe26_to_6 or [24, 20, 22, 25, 21, 23]
    num_kpts = keypoints_ts.shape[1]
    if num_kpts == 26:
        kp6 = keypoints_ts[:, h26, :].copy()
    elif num_kpts == 10:
        kp6 = raw_to_six_keypoints(keypoints_ts, num_kpts)
    else:
        kp6 = keypoints_ts[:, :6, :].copy() if num_kpts >= 6 else keypoints_ts

    # Core pipeline
    events, step_records, summary = run_pipeline(kp6, times_ms, fps, config=config)
    result.events = events
    result.step_records = step_records
    result.summary = summary

    # Bouts
    bouts = detect_toe_walking_bouts(events, min_consecutive=config.min_consecutive_bout)
    ic_count = sum(1 for e in events if e.event_type == "IC")
    bout_summary = compute_bout_summary(bouts, ic_count)
    result.bout_summary = bout_summary.to_dict()

    # Severity
    sev = compute_severity_index(summary, bout_summary)
    result.severity = sev.to_dict()

    # Extended metrics
    ext: dict = {}
    hr = compute_heel_rise_timing(kp6, times_ms, events)
    latencies = [r.heel_rise_latency_ms for r in hr if r.heel_rise_latency_ms is not None]
    ext["heel_rise_latency_mean_ms"] = float(np.mean(latencies)) if latencies else None
    ext["heel_rise_latency_std_ms"] = float(np.std(latencies, ddof=1)) if len(latencies) > 1 else None

    pf = compute_plantarflexion_proxy(kp6, times_ms, events)
    pitches = [r.pitch_deg for r in pf if r.pitch_deg is not None]
    ext["plantarflexion_proxy_mean_deg"] = float(np.mean(pitches)) if pitches else None
    ext["plantarflexion_proxy_std_deg"] = float(np.std(pitches, ddof=1)) if len(pitches) > 1 else None

    fsv = compute_foot_strike_variability(events)
    ext["foot_strike_entropy"] = fsv.shannon_entropy

    # COM proxy only if full keypoints available
    if num_kpts >= 10:
        com = compute_com_sway_proxy(keypoints_ts, hip_left_idx=0, hip_right_idx=1)
        ext["com_sway_lateral_std_px"] = com.lateral_std_px
    result.extended = ext

    return result


def run_batch(
    manifest_path: Path,
    model_path: str,
    config: PipelineConfig,
    output_dir: Path,
    device: str = "auto",
    conf_threshold: float = 0.25,
) -> list[TrialResult]:
    """Process all videos in a manifest CSV.

    Parameters
    ----------
    manifest_path : Path
        CSV file with at least a ``video_path`` column.
    model_path : str
        YOLO model weights path.
    config : PipelineConfig
        Pipeline configuration.
    output_dir : Path
        Directory for output CSVs and JSONs.
    device : str
    conf_threshold : float

    Returns
    -------
    list[TrialResult]
    """
    manifest = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[TrialResult] = []

    for i, row in enumerate(manifest):
        vp = row.get("video_path", "")
        if not vp:
            logger.warning("Row %d: missing video_path, skipping", i)
            continue
        video_path = Path(vp)
        if not video_path.is_absolute():
            video_path = manifest_path.parent / video_path
        if not video_path.exists():
            logger.warning("Video not found: %s", video_path)
            results.append(TrialResult(video_path=str(video_path), error="File not found"))
            continue

        logger.info("Processing [%d/%d]: %s", i + 1, len(manifest), video_path.name)
        trial = run_single_trial(
            video_path, model_path, config, device, conf_threshold, metadata=row
        )
        results.append(trial)

        # Per-trial outputs
        trial_stem = video_path.stem
        trial_dir = output_dir / trial_stem
        trial_dir.mkdir(exist_ok=True)
        if trial.events:
            events_to_dataframe(trial.events).to_csv(
                trial_dir / "events.csv", index=False
            )
        if trial.step_records:
            step_records_to_dataframe(trial.step_records).to_csv(
                trial_dir / "steps.csv", index=False
            )
        trial_json = trial.summary_row()
        trial_json["bout_details"] = trial.bout_summary
        trial_json["severity_details"] = trial.severity
        trial_json["extended_metrics"] = trial.extended
        with open(trial_dir / "summary.json", "w") as f:
            json.dump(trial_json, f, indent=2, default=str)

    # Aggregate summary CSV
    summary_rows = [r.summary_row() for r in results]
    pd.DataFrame(summary_rows).to_csv(output_dir / "batch_summary.csv", index=False)

    # Per-participant aggregation
    if any(r.participant_id for r in results):
        from .stats_utils import aggregate_trials
        agg = aggregate_trials(summary_rows, participant_col="participant_id")
        pd.DataFrame(agg).to_csv(output_dir / "participant_summary.csv", index=False)

    logger.info("Batch complete: %d trials → %s", len(results), output_dir)
    return results
