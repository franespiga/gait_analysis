"""
Gait metrics from IC/TO events: cadence, step/stride time, stance/swing,
step/stride length (with calibration), speed, symmetry, variability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .gait_events import GaitEvent, ContactType


@dataclass
class StepRecord:
    """One step with timing and optional lengths (for per-step CSV).

    ``stride_time_ms`` stores the same-foot IC-to-IC interval (one full
    gait cycle), **not** the cross-foot step time.
    """
    foot: str
    ic_time_ms: float
    to_time_ms: float
    next_ic_time_ms: Optional[float] = None
    contact_type: str = "unknown"
    stride_time_ms: Optional[float] = None
    stance_time_ms: Optional[float] = None
    swing_time_ms: Optional[float] = None
    step_length_px: Optional[float] = None
    step_length_m: Optional[float] = None
    stride_length_px: Optional[float] = None
    stride_length_m: Optional[float] = None
    quality_score: Optional[float] = None
    overlap_flag: bool = False
    low_confidence_flag: bool = False
    excluded_from_primary_metrics: bool = False


def _pair_ic_to(events: list[GaitEvent]) -> list[tuple[GaitEvent, Optional[GaitEvent]]]:
    """Pair each IC with its next TO on the same foot."""
    by_foot: dict[str, list[GaitEvent]] = {"left": [], "right": []}
    for e in sorted(events, key=lambda x: x.time_ms):
        by_foot[e.foot].append(e)
    pairs: list[tuple[GaitEvent, Optional[GaitEvent]]] = []
    for foot in ("left", "right"):
        ev = by_foot[foot]
        i = 0
        while i < len(ev):
            if ev[i].event_type != "IC":
                i += 1
                continue
            ic = ev[i]
            to = None
            for j in range(i + 1, len(ev)):
                if ev[j].event_type == "TO":
                    to = ev[j]
                    break
            pairs.append((ic, to))
            i += 1
    return sorted(pairs, key=lambda p: p[0].time_ms)


def _next_ic_same_foot(events: list[GaitEvent], ic: GaitEvent) -> Optional[GaitEvent]:
    """Next IC on same foot after this IC."""
    cand = [e for e in events if e.event_type == "IC" and e.foot == ic.foot and e.time_ms > ic.time_ms]
    return min(cand, key=lambda e: e.time_ms) if cand else None


def build_step_records(
    events: list[GaitEvent],
    scale_m_per_px: Optional[float] = None,
) -> list[StepRecord]:
    """Build step records from events. Optionally convert lengths to m using scale_m_per_px."""
    pairs = _pair_ic_to(events)
    records: list[StepRecord] = []
    for ic, to in pairs:
        stance_ms = (to.time_ms - ic.time_ms) if to else None
        next_ic = _next_ic_same_foot(events, ic)
        stride_time_ms = (next_ic.time_ms - ic.time_ms) if next_ic else None
        swing_ms = (next_ic.time_ms - to.time_ms) if (to and next_ic) else None

        # Same-foot IC to next IC = stride length; step length = half (one step of that foot)
        stride_len_px = None
        if ic.x is not None and next_ic and next_ic.x is not None:
            stride_len_px = np.hypot(next_ic.x - ic.x, (next_ic.y or 0) - (ic.y or 0))
        step_len_px = (stride_len_px / 2.0) if stride_len_px is not None else None

        step_len_m = step_len_px * scale_m_per_px if (step_len_px and scale_m_per_px) else None
        stride_len_m = stride_len_px * scale_m_per_px if (stride_len_px and scale_m_per_px) else None

        records.append(StepRecord(
            foot=ic.foot,
            ic_time_ms=ic.time_ms,
            to_time_ms=to.time_ms if to else 0,
            next_ic_time_ms=next_ic.time_ms if next_ic else None,
            contact_type=ic.contact_type.value if ic.contact_type else "unknown",
            stride_time_ms=stride_time_ms,
            stance_time_ms=stance_ms,
            swing_time_ms=swing_ms,
            step_length_px=step_len_px,
            step_length_m=step_len_m,
            stride_length_px=stride_len_px,
            stride_length_m=stride_len_m,
            quality_score=ic.quality_score,
            overlap_flag=ic.overlap_flag,
            low_confidence_flag=ic.low_confidence_flag,
            excluded_from_primary_metrics=ic.excluded_from_primary_metrics,
        ))
    return records


@dataclass
class GaitSummary:
    """Summary metrics for JSON output.

    Terminology (biomechanical standard):
    - **step time**: IC of one foot → IC of the *contralateral* foot.
    - **stride time**: IC of one foot → next IC of the *same* foot (= gait cycle).
    """
    cadence_steps_per_min: float = 0.0
    step_time_mean_ms: float = 0.0
    step_time_std_ms: float = 0.0
    stride_time_mean_ms: float = 0.0
    stride_time_std_ms: float = 0.0
    stride_time_cv_percent: float = 0.0
    stance_time_mean_ms: float = 0.0
    stance_time_std_ms: float = 0.0
    swing_time_mean_ms: float = 0.0
    swing_time_std_ms: float = 0.0
    duty_factor: float = 0.0
    step_length_mean_px: float = 0.0
    step_length_std_px: float = 0.0
    step_length_mean_m: Optional[float] = None
    step_length_std_m: Optional[float] = None
    stride_length_mean_px: float = 0.0
    stride_length_std_px: float = 0.0
    stride_length_mean_m: Optional[float] = None
    stride_length_std_m: Optional[float] = None
    speed_m_per_s: Optional[float] = None
    symmetry_index_step_time: float = 0.0  # 0 = symmetric
    symmetry_index_stance: float = 0.0
    step_time_cv_percent: float = 0.0
    stance_time_cv_percent: float = 0.0
    step_length_cv_percent: float = 0.0
    total_steps: int = 0
    duration_s: float = 0.0
    heel_strike_count: int = 0
    toe_walk_count: int = 0
    flat_count: int = 0
    heel_strike_rate_pct: float = 0.0
    toe_walk_rate_pct: float = 0.0
    flat_rate_pct: float = 0.0

    def to_dict(self) -> dict:
        d: dict = {
            "cadence_steps_per_min": round(self.cadence_steps_per_min, 2),
            "step_time_mean_ms": round(self.step_time_mean_ms, 2),
            "step_time_std_ms": round(self.step_time_std_ms, 2),
            "stride_time_mean_ms": round(self.stride_time_mean_ms, 2),
            "stride_time_std_ms": round(self.stride_time_std_ms, 2),
            "stride_time_cv_percent": round(self.stride_time_cv_percent, 2),
            "stance_time_mean_ms": round(self.stance_time_mean_ms, 2),
            "stance_time_std_ms": round(self.stance_time_std_ms, 2),
            "swing_time_mean_ms": round(self.swing_time_mean_ms, 2),
            "swing_time_std_ms": round(self.swing_time_std_ms, 2),
            "duty_factor": round(self.duty_factor, 4),
            "step_length_mean_px": round(self.step_length_mean_px, 2),
            "step_length_std_px": round(self.step_length_std_px, 2),
            "stride_length_mean_px": round(self.stride_length_mean_px, 2),
            "stride_length_std_px": round(self.stride_length_std_px, 2),
            "speed_m_per_s": round(self.speed_m_per_s, 4) if self.speed_m_per_s is not None else None,
            "symmetry_index_step_time": round(self.symmetry_index_step_time, 4),
            "symmetry_index_stance": round(self.symmetry_index_stance, 4),
            "step_time_cv_percent": round(self.step_time_cv_percent, 2),
            "stance_time_cv_percent": round(self.stance_time_cv_percent, 2),
            "step_length_cv_percent": round(self.step_length_cv_percent, 2),
            "total_steps": self.total_steps,
            "duration_s": round(self.duration_s, 2),
            "heel_strike_count": self.heel_strike_count,
            "toe_walk_count": self.toe_walk_count,
            "flat_count": self.flat_count,
            "heel_strike_rate_pct": round(self.heel_strike_rate_pct, 2),
            "toe_walk_rate_pct": round(self.toe_walk_rate_pct, 2),
            "flat_rate_pct": round(self.flat_rate_pct, 2),
        }
        if self.step_length_mean_m is not None:
            d["step_length_mean_m"] = round(self.step_length_mean_m, 4)
            d["step_length_std_m"] = round(self.step_length_std_m or 0, 4)
        if self.stride_length_mean_m is not None:
            d["stride_length_mean_m"] = round(self.stride_length_mean_m, 4)
            d["stride_length_std_m"] = round(self.stride_length_std_m or 0, 4)
        return d


def _compute_cross_foot_step_times(step_records: list[StepRecord]) -> list[float]:
    """True step time: IC of one foot → IC of the *contralateral* foot.

    This is the biomechanically standard definition of step time,
    derived by interleaving ICs from both feet sorted chronologically.
    """
    all_ics = sorted(
        [(r.ic_time_ms, r.foot) for r in step_records],
        key=lambda x: x[0],
    )
    step_times: list[float] = []
    for i in range(len(all_ics) - 1):
        if all_ics[i][1] != all_ics[i + 1][1]:
            step_times.append(all_ics[i + 1][0] - all_ics[i][0])
    return step_times


def compute_summary(
    step_records: list[StepRecord],
    duration_s: float,
    scale_m_per_px: Optional[float] = None,
) -> GaitSummary:
    """Compute all required gait metrics from step records.

    Stride time is computed from same-foot IC intervals (``StepRecord.stride_time_ms``).
    True step time is derived from consecutive opposite-foot IC intervals.
    """
    s = GaitSummary()
    if not step_records:
        s.duration_s = duration_s
        return s

    s.total_steps = len(step_records)
    s.duration_s = duration_s
    for r in step_records:
        if r.contact_type == ContactType.HEEL_STRIKE.value:
            s.heel_strike_count += 1
        elif r.contact_type == ContactType.TOE_WALK.value:
            s.toe_walk_count += 1
        elif r.contact_type == ContactType.FLAT.value:
            s.flat_count += 1

    if s.total_steps > 0:
        s.heel_strike_rate_pct = 100.0 * s.heel_strike_count / s.total_steps
        s.toe_walk_rate_pct = 100.0 * s.toe_walk_count / s.total_steps
        s.flat_rate_pct = 100.0 * s.flat_count / s.total_steps

    stride_times = [r.stride_time_ms for r in step_records if r.stride_time_ms is not None]
    if stride_times:
        s.stride_time_mean_ms = float(np.mean(stride_times))
        s.stride_time_std_ms = float(np.std(stride_times, ddof=1)) if len(stride_times) > 1 else 0.0
        s.stride_time_cv_percent = (
            (s.stride_time_std_ms / s.stride_time_mean_ms * 100)
            if s.stride_time_mean_ms else 0.0
        )

    # True step time: IC → next IC on opposite foot.
    cross_step_times = _compute_cross_foot_step_times(step_records)
    if cross_step_times:
        s.step_time_mean_ms = float(np.mean(cross_step_times))
        s.step_time_std_ms = float(np.std(cross_step_times, ddof=1)) if len(cross_step_times) > 1 else 0.0
        s.step_time_cv_percent = (
            (s.step_time_std_ms / s.step_time_mean_ms * 100)
            if s.step_time_mean_ms else 0.0
        )
    elif stride_times:
        s.step_time_mean_ms = float(np.mean(stride_times)) / 2.0
        s.step_time_std_ms = (float(np.std(stride_times, ddof=1)) / 2.0) if len(stride_times) > 1 else 0.0

    stance_times = [r.stance_time_ms for r in step_records if r.stance_time_ms is not None]
    swing_times = [r.swing_time_ms for r in step_records if r.swing_time_ms is not None]
    step_len_px = [r.step_length_px for r in step_records if r.step_length_px is not None]
    stride_len_px = [r.stride_length_px for r in step_records if r.stride_length_px is not None]

    if stance_times:
        s.stance_time_mean_ms = float(np.mean(stance_times))
        s.stance_time_std_ms = float(np.std(stance_times, ddof=1)) if len(stance_times) > 1 else 0.0
        s.stance_time_cv_percent = (s.stance_time_std_ms / s.stance_time_mean_ms * 100) if s.stance_time_mean_ms else 0
    if swing_times:
        s.swing_time_mean_ms = float(np.mean(swing_times))
        s.swing_time_std_ms = float(np.std(swing_times, ddof=1)) if len(swing_times) > 1 else 0.0
    if stance_times and stride_times and np.mean(stride_times) > 0:
        s.duty_factor = float(np.mean(stance_times)) / float(np.mean(stride_times))
    if step_len_px:
        s.step_length_mean_px = float(np.mean(step_len_px))
        s.step_length_std_px = float(np.std(step_len_px, ddof=1)) if len(step_len_px) > 1 else 0.0
        s.step_length_cv_percent = (s.step_length_std_px / s.step_length_mean_px * 100) if s.step_length_mean_px else 0
    if stride_len_px:
        s.stride_length_mean_px = float(np.mean(stride_len_px))
        s.stride_length_std_px = float(np.std(stride_len_px, ddof=1)) if len(stride_len_px) > 1 else 0.0
    if scale_m_per_px and step_len_px:
        s.step_length_mean_m = float(np.mean(step_len_px)) * scale_m_per_px
        s.step_length_std_m = (float(np.std(step_len_px, ddof=1)) * scale_m_per_px) if len(step_len_px) > 1 else 0.0
    if scale_m_per_px and stride_len_px:
        s.stride_length_mean_m = float(np.mean(stride_len_px)) * scale_m_per_px
        s.stride_length_std_m = (float(np.std(stride_len_px, ddof=1)) * scale_m_per_px) if len(stride_len_px) > 1 else 0.0
    if duration_s > 0 and s.total_steps > 0:
        s.cadence_steps_per_min = (s.total_steps / duration_s) * 60.0
    if s.step_time_mean_ms > 0 and s.step_length_mean_m is not None:
        s.speed_m_per_s = s.step_length_mean_m / (s.step_time_mean_ms / 1000.0)
    elif duration_s > 0 and s.stride_length_mean_m is not None and s.total_steps >= 2:
        total_dist = s.stride_length_mean_m * (s.total_steps / 2.0)
        s.speed_m_per_s = total_dist / duration_s

    left_steps = [r for r in step_records if r.foot == "left" and r.stride_time_ms is not None]
    right_steps = [r for r in step_records if r.foot == "right" and r.stride_time_ms is not None]
    if left_steps and right_steps:
        left_st = np.mean([r.stride_time_ms for r in left_steps])
        right_st = np.mean([r.stride_time_ms for r in right_steps])
        s.symmetry_index_step_time = 2.0 * abs(left_st - right_st) / (left_st + right_st) if (left_st + right_st) else 0
    left_stance = [r.stance_time_ms for r in step_records if r.foot == "left" and r.stance_time_ms is not None]
    right_stance = [r.stance_time_ms for r in step_records if r.foot == "right" and r.stance_time_ms is not None]
    if left_stance and right_stance:
        ls = np.mean(left_stance)
        rs = np.mean(right_stance)
        s.symmetry_index_stance = 2.0 * abs(ls - rs) / (ls + rs) if (ls + rs) else 0

    return s
