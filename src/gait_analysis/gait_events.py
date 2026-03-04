"""
Gait event detection: Initial Contact (IC) and Toe-Off (TO) per foot.

Uses velocity and stability windows. Classifies each IC as heel-strike,
toe-walk, or flat using foot pitch and contact timing.
Keypoint schema: L_HEEL, L_BIG_TOE, L_SMALL_TOE, R_HEEL, R_BIG_TOE, R_SMALL_TOE (indices 0-5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ContactType(Enum):
    HEEL_STRIKE = "heel_strike"
    TOE_WALK = "toe_walk"
    FLAT = "flat"
    UNKNOWN = "unknown"


@dataclass
class GaitEvent:
    """Single gait event (IC or TO)."""
    event_type: str  # "IC" or "TO"
    foot: str       # "left" or "right"
    frame: int
    time_ms: float
    contact_type: Optional[ContactType] = None  # for IC only
    # Optional: position at event (e.g. heel or toe x,y for step length)
    x: Optional[float] = None
    y: Optional[float] = None


# Indices for 6-keypoint foot schema
L_HEEL, L_BIG_TOE, L_SMALL_TOE = 0, 1, 2
R_HEEL, R_BIG_TOE, R_SMALL_TOE = 3, 4, 5

FOOT_KEYPOINTS = {
    "left": {"heel": L_HEEL, "toe": L_BIG_TOE, "small_toe": L_SMALL_TOE},
    "right": {"heel": R_HEEL, "toe": R_BIG_TOE, "small_toe": R_SMALL_TOE},
}


def _velocity_y(series: np.ndarray) -> np.ndarray:
    """Vertical velocity (positive = downward in image)."""
    v = np.zeros_like(series, dtype=float)
    v[1:] = np.diff(series)
    return v


def _stable_mask(velocity: np.ndarray, threshold: float, min_frames: int) -> np.ndarray:
    """True where velocity is below threshold for at least min_frames in a row (expanded)."""
    below = np.abs(velocity) <= threshold
    T = len(below)
    out = np.zeros(T, dtype=bool)
    for i in range(T):
        end = min(i + min_frames, T)
        if np.all(below[i:end]):
            out[i] = True
    return out


def detect_ic_to(
    keypoints: np.ndarray,
    times_ms: np.ndarray,
    velocity_threshold: float = 2.0,
    stability_frames: int = 5,
    min_step_duration_ms: float = 200.0,
    conf_threshold: float = 0.3,
) -> list[GaitEvent]:
    """
    Detect Initial Contact (IC) and Toe-Off (TO) per foot from smoothed keypoints.

    keypoints: (T, 6, 3) with (x, y, conf) for L_HEEL, L_BIG_TOE, L_SMALL_TOE, R_HEEL, R_BIG_TOE, R_SMALL_TOE.
    times_ms: (T,) frame times in ms.

    Logic: Use heel Y (or toe Y) trajectory; IC when foot becomes stable (low velocity);
    TO when foot leaves stability (velocity increases). Filter by min_step_duration_ms.
    """
    events: list[GaitEvent] = []
    T = keypoints.shape[0]
    if T < stability_frames + 2:
        return events

    dt_ms = (times_ms[-1] - times_ms[0]) / (T - 1) if T > 1 else 0

    for foot in ("left", "right"):
        heel_idx = FOOT_KEYPOINTS[foot]["heel"]
        toe_idx = FOOT_KEYPOINTS[foot]["toe"]
        heel_y = keypoints[:, heel_idx, 1]
        toe_y = keypoints[:, toe_idx, 1]
        heel_conf = keypoints[:, heel_idx, 2]
        toe_conf = keypoints[:, toe_idx, 2]
        # Use heel for primary stability when confident; else toe
        use_y = np.where(heel_conf >= conf_threshold, heel_y, toe_y)
        valid = (heel_conf >= conf_threshold) | (toe_conf >= conf_threshold)
        if not np.any(valid):
            continue
        vy = _velocity_y(use_y)
        stable = _stable_mask(vy, velocity_threshold, stability_frames)
        # IC: transition from not stable to stable
        # TO: transition from stable to not stable
        for i in range(1, T):
            if stable[i] and not stable[i - 1]:
                events.append(GaitEvent(
                    event_type="IC",
                    foot=foot,
                    frame=i,
                    time_ms=float(times_ms[i]),
                    contact_type=None,
                    x=float(keypoints[i, heel_idx, 0]) if heel_conf[i] >= conf_threshold else None,
                    y=float(keypoints[i, heel_idx, 1]) if heel_conf[i] >= conf_threshold else None,
                ))
            if not stable[i] and stable[i - 1]:
                events.append(GaitEvent(
                    event_type="TO",
                    foot=foot,
                    frame=i,
                    time_ms=float(times_ms[i]),
                    contact_type=None,
                ))

    # Filter: enforce min_step_duration_ms between consecutive events on same foot
    events.sort(key=lambda e: (e.time_ms, e.event_type))
    filtered: list[GaitEvent] = []
    last_ic_time: dict[str, float] = {}
    for e in events:
        if e.event_type == "IC":
            t_prev = last_ic_time.get(e.foot, -1e9)
            if e.time_ms - t_prev >= min_step_duration_ms:
                filtered.append(e)
                last_ic_time[e.foot] = e.time_ms
        else:
            filtered.append(e)
    return filtered


def classify_contact(
    keypoints: np.ndarray,
    times_ms: np.ndarray,
    ic_event: GaitEvent,
    pitch_threshold_deg: float = 8.0,
    stabilization_delta_ms: float = 50.0,
    flat_window_ms: float = 80.0,
    conf_threshold: float = 0.3,
) -> ContactType:
    """
    Classify an IC event as heel-strike, toe-walk, or flat.

    - Heel-strike: heel stabilizes first, heel lower than toe (higher Y in image).
    - Toe-walk: toe stabilizes first, toe lower than heel.
    - Flat: both stabilize within flat_window_ms, small pitch angle.
    """
    t0 = ic_event.time_ms
    frame0 = ic_event.frame
    foot = ic_event.foot
    heel_idx = FOOT_KEYPOINTS[foot]["heel"]
    toe_idx = FOOT_KEYPOINTS[foot]["toe"]
    T = keypoints.shape[0]
    if frame0 >= T:
        return ContactType.UNKNOWN
    heel_y = keypoints[:, heel_idx, 1]
    toe_y = keypoints[:, toe_idx, 1]
    heel_conf = keypoints[:, heel_idx, 2]
    toe_conf = keypoints[:, toe_idx, 2]
    if heel_conf[frame0] < conf_threshold or toe_conf[frame0] < conf_threshold:
        return ContactType.UNKNOWN

    # Vertical position at IC (higher Y = lower in scene)
    heel_y0 = heel_y[frame0]
    toe_y0 = toe_y[frame0]
    diff_y = heel_y0 - toe_y0  # positive => heel lower (heel strike in image coords)

    # Look backward for "who stabilized first" in a short window before IC
    dt_ms = (times_ms[-1] - times_ms[0]) / max(1, T - 1)
    look_back_frames = max(2, int(stabilization_delta_ms / dt_ms)) if dt_ms > 0 else 5
    start = max(0, frame0 - look_back_frames)
    v_heel = np.abs(np.diff(heel_y[start : frame0 + 1]))
    v_toe = np.abs(np.diff(toe_y[start : frame0 + 1]))
    mean_v_heel = np.mean(v_heel) if len(v_heel) else 1.0
    mean_v_toe = np.mean(v_toe) if len(v_toe) else 1.0

    # Pitch: angle of foot (heel-toe line) from horizontal; small => flat
    dx = keypoints[frame0, toe_idx, 0] - keypoints[frame0, heel_idx, 0]
    dy = toe_y0 - heel_y0
    pitch_deg = np.degrees(np.abs(np.arctan2(dy, dx))) if (dx != 0 or dy != 0) else 0.0

    if pitch_deg < pitch_threshold_deg:
        return ContactType.FLAT
    if diff_y > 0 and mean_v_heel <= mean_v_toe:
        return ContactType.HEEL_STRIKE
    if diff_y < 0 and mean_v_toe <= mean_v_heel:
        return ContactType.TOE_WALK
    if diff_y > 0:
        return ContactType.HEEL_STRIKE
    return ContactType.TOE_WALK


def events_to_contact_types(
    keypoints: np.ndarray,
    times_ms: np.ndarray,
    events: list[GaitEvent],
    pitch_threshold_deg: float = 8.0,
    stabilization_delta_ms: float = 50.0,
    flat_window_ms: float = 80.0,
    conf_threshold: float = 0.3,
) -> list[GaitEvent]:
    """Assign contact_type to each IC event in place and return events."""
    for e in events:
        if e.event_type == "IC":
            e.contact_type = classify_contact(
                keypoints,
                times_ms,
                e,
                pitch_threshold_deg=pitch_threshold_deg,
                stabilization_delta_ms=stabilization_delta_ms,
                flat_window_ms=flat_window_ms,
                conf_threshold=conf_threshold,
            )
    return events
