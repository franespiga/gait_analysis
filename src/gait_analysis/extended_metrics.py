"""
Extended gait metrics beyond core spatiotemporal parameters.

All metrics that cannot be directly measured from 2-D video keypoints are
clearly documented as **proxies** with their assumptions.

Includes
--------
- heel-rise timing proxy
- plantarflexion / foot-pitch proxy at IC
- foot-strike type variability (Shannon entropy + transition matrix)
- COM sway proxy (hip-midpoint lateral deviation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .gait_events import GaitEvent, ContactType, FOOT_KEYPOINTS


# ---------------------------------------------------------------------------
# Heel-rise timing (proxy)
# ---------------------------------------------------------------------------

@dataclass
class HeelRiseRecord:
    """Timing from IC to first upward heel movement during stance."""
    foot: str
    ic_frame: int
    ic_time_ms: float
    heel_rise_frame: Optional[int] = None
    heel_rise_time_ms: Optional[float] = None
    heel_rise_latency_ms: Optional[float] = None  # None if not detected


def compute_heel_rise_timing(
    keypoints: np.ndarray,
    times_ms: np.ndarray,
    events: list[GaitEvent],
    rise_velocity_threshold: float = -1.0,
    max_search_ms: float = 500.0,
    conf_threshold: float = 0.3,
) -> list[HeelRiseRecord]:
    """Detect when the heel begins to rise after initial contact.

    **Proxy assumption:** In image coordinates (Y increases downward),
    heel-rise corresponds to the heel Y coordinate *decreasing*.  We
    detect the first frame after IC where the vertical velocity of the
    heel falls below ``rise_velocity_threshold`` (negative = upward)
    within ``max_search_ms``.

    Parameters
    ----------
    keypoints : ndarray (T, 6, 3)
        Smoothed 6-keypoint foot trajectories.
    times_ms : ndarray (T,)
    events : list[GaitEvent]
        Classified event list (only IC events are used).
    rise_velocity_threshold : float
        Vertical velocity (px/frame) that signals upward heel motion.
        Default -1.0 (heel moves ≥ 1 px upward per frame).
    max_search_ms : float
        Maximum time after IC to search for heel-rise.

    Returns
    -------
    list[HeelRiseRecord]
        One record per IC event.
    """
    T = keypoints.shape[0]
    dt_ms = (times_ms[-1] - times_ms[0]) / max(1, T - 1) if T > 1 else 0
    max_frames = int(max_search_ms / dt_ms) if dt_ms > 0 else 20

    records: list[HeelRiseRecord] = []
    for ev in events:
        if ev.event_type != "IC":
            continue
        heel_idx = FOOT_KEYPOINTS[ev.foot]["heel"]
        rec = HeelRiseRecord(foot=ev.foot, ic_frame=ev.frame, ic_time_ms=ev.time_ms)
        heel_conf = keypoints[:, heel_idx, 2]
        if ev.frame >= T or heel_conf[ev.frame] < conf_threshold:
            records.append(rec)
            continue
        heel_y = keypoints[:, heel_idx, 1]
        end = min(ev.frame + max_frames, T)
        for f in range(ev.frame + 1, end):
            if heel_conf[f] < conf_threshold:
                continue
            vy = heel_y[f] - heel_y[f - 1]
            if vy <= rise_velocity_threshold:
                rec.heel_rise_frame = f
                rec.heel_rise_time_ms = float(times_ms[f])
                rec.heel_rise_latency_ms = float(times_ms[f] - ev.time_ms)
                break
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Plantarflexion proxy (foot pitch at IC)
# ---------------------------------------------------------------------------

@dataclass
class PlantarflexionRecord:
    """Foot pitch angle at initial contact — proxy for plantarflexion."""
    foot: str
    ic_frame: int
    ic_time_ms: float
    pitch_deg: Optional[float] = None
    contact_type: Optional[str] = None


def compute_plantarflexion_proxy(
    keypoints: np.ndarray,
    times_ms: np.ndarray,
    events: list[GaitEvent],
    conf_threshold: float = 0.3,
) -> list[PlantarflexionRecord]:
    """Compute foot pitch angle at each IC as a plantarflexion proxy.

    **Proxy assumption:** The angle of the heel→toe vector from horizontal
    approximates the sagittal-plane foot angle.  Positive pitch = toe
    lower than heel (plantarflexed); negative = dorsiflexed.
    This is a *2-D projection* and does not account for out-of-plane
    rotation or true ankle joint kinematics.
    """
    T = keypoints.shape[0]
    records: list[PlantarflexionRecord] = []
    for ev in events:
        if ev.event_type != "IC":
            continue
        heel_idx = FOOT_KEYPOINTS[ev.foot]["heel"]
        toe_idx = FOOT_KEYPOINTS[ev.foot]["toe"]
        rec = PlantarflexionRecord(
            foot=ev.foot,
            ic_frame=ev.frame,
            ic_time_ms=ev.time_ms,
            contact_type=ev.contact_type.value if ev.contact_type else None,
        )
        if ev.frame >= T:
            records.append(rec)
            continue
        hc = keypoints[ev.frame, heel_idx, 2]
        tc = keypoints[ev.frame, toe_idx, 2]
        if hc < conf_threshold or tc < conf_threshold:
            records.append(rec)
            continue
        dx = keypoints[ev.frame, toe_idx, 0] - keypoints[ev.frame, heel_idx, 0]
        # In image coords Y-down → positive dy means toe is lower
        dy = keypoints[ev.frame, toe_idx, 1] - keypoints[ev.frame, heel_idx, 1]
        rec.pitch_deg = float(np.degrees(np.arctan2(dy, abs(dx)))) if (dx != 0 or dy != 0) else 0.0
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Foot-strike type variability
# ---------------------------------------------------------------------------

@dataclass
class FootStrikeVariability:
    """Variability statistics for the sequence of contact types."""
    total_classified: int = 0
    shannon_entropy: float = 0.0
    transition_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    type_proportions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_classified": self.total_classified,
            "shannon_entropy": round(self.shannon_entropy, 4),
            "transition_matrix": self.transition_matrix,
            "type_proportions": {k: round(v, 4) for k, v in self.type_proportions.items()},
        }


def compute_foot_strike_variability(
    events: list[GaitEvent],
) -> FootStrikeVariability:
    """Compute variability of foot-strike type over the trial.

    Returns Shannon entropy of the contact-type distribution and a
    first-order transition probability matrix.
    """
    ic_types = [
        ev.contact_type.value
        for ev in sorted(events, key=lambda e: e.time_ms)
        if ev.event_type == "IC" and ev.contact_type is not None
        and ev.contact_type != ContactType.UNKNOWN
    ]
    result = FootStrikeVariability(total_classified=len(ic_types))
    if not ic_types:
        return result

    labels = sorted(set(ic_types))
    counts = {lab: ic_types.count(lab) for lab in labels}
    total = len(ic_types)
    props = {lab: c / total for lab, c in counts.items()}
    result.type_proportions = props

    # Shannon entropy (higher = more variable)
    probs = np.array(list(props.values()))
    probs = probs[probs > 0]
    result.shannon_entropy = float(-np.sum(probs * np.log2(probs)))

    # First-order transition matrix
    trans: dict[str, dict[str, int]] = {l: {m: 0 for m in labels} for l in labels}
    for i in range(len(ic_types) - 1):
        trans[ic_types[i]][ic_types[i + 1]] += 1
    trans_prob: dict[str, dict[str, float]] = {}
    for src, dests in trans.items():
        row_total = sum(dests.values())
        trans_prob[src] = {
            dst: (cnt / row_total if row_total else 0.0)
            for dst, cnt in dests.items()
        }
    result.transition_matrix = trans_prob
    return result


# ---------------------------------------------------------------------------
# COM sway proxy (lateral hip midpoint deviation)
# ---------------------------------------------------------------------------

@dataclass
class COMSwayProxy:
    """Centre-of-mass sway proxy from hip midpoint lateral deviation.

    **Proxy assumption:** The midpoint of left and right hip keypoints
    approximates the lateral position of the body's centre of mass in
    the frontal plane.  Its standard deviation gives a stability proxy.
    This is a *2-D image-space* measure and does not account for depth
    or true 3-D COM.
    """
    lateral_std_px: float = 0.0
    lateral_range_px: float = 0.0
    vertical_std_px: float = 0.0
    n_valid_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "lateral_std_px": round(self.lateral_std_px, 2),
            "lateral_range_px": round(self.lateral_range_px, 2),
            "vertical_std_px": round(self.vertical_std_px, 2),
            "n_valid_frames": self.n_valid_frames,
            "_note": "Proxy: 2-D hip-midpoint, not true 3-D COM.",
        }


def compute_com_sway_proxy(
    keypoints: np.ndarray,
    hip_left_idx: int = 0,
    hip_right_idx: int = 3,
    conf_threshold: float = 0.3,
) -> COMSwayProxy:
    """Compute lateral and vertical standard deviation of the hip midpoint.

    Parameters
    ----------
    keypoints : ndarray (T, K, 3)
        Full keypoint array (not the 6-keypoint foot subset — pass the
        original 10-keypoint or 26-keypoint array if available).  If only
        the 6-keypoint foot array is available, this function returns an
        empty result (hips are not in the foot schema).
    hip_left_idx, hip_right_idx : int
        Column indices for left and right hip in the keypoint array.
    """
    result = COMSwayProxy()
    T, K, _ = keypoints.shape
    if K <= max(hip_left_idx, hip_right_idx):
        return result

    lh = keypoints[:, hip_left_idx, :]
    rh = keypoints[:, hip_right_idx, :]
    valid = (lh[:, 2] >= conf_threshold) & (rh[:, 2] >= conf_threshold)
    if not np.any(valid):
        return result

    mid_x = (lh[valid, 0] + rh[valid, 0]) / 2.0
    mid_y = (lh[valid, 1] + rh[valid, 1]) / 2.0
    result.lateral_std_px = float(np.std(mid_x, ddof=1)) if len(mid_x) > 1 else 0.0
    result.lateral_range_px = float(np.ptp(mid_x))
    result.vertical_std_px = float(np.std(mid_y, ddof=1)) if len(mid_y) > 1 else 0.0
    result.n_valid_frames = int(np.sum(valid))
    return result
