"""
Toe-walking bout detection from classified gait events.

A **toe-walking bout** is a run of >= ``min_consecutive`` consecutive
forefoot-first (``TOE_WALK``) initial contacts without any intervening
heel-strike or flat contact.  Bouts are detected by scanning the
time-sorted IC event sequence.

Clinical rationale
------------------
Isolated forefoot contacts may be incidental (e.g., turning), while
*sustained* runs indicate habitual toe-walking, which is more clinically
relevant in children with ASD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .gait_events import GaitEvent, ContactType


@dataclass
class ToeWalkingBout:
    """One contiguous run of forefoot-first contacts."""
    start_idx: int          # index into the IC-only event list
    end_idx: int            # inclusive
    start_time_ms: float
    end_time_ms: float
    num_steps: int
    duration_ms: float
    foot_sequence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start_time_ms": round(self.start_time_ms, 2),
            "end_time_ms": round(self.end_time_ms, 2),
            "num_steps": self.num_steps,
            "duration_ms": round(self.duration_ms, 2),
            "foot_sequence": self.foot_sequence,
        }


@dataclass
class BoutSummary:
    """Aggregate statistics over all detected bouts."""
    total_bouts: int = 0
    total_bout_steps: int = 0
    mean_bout_length_steps: float = 0.0
    max_bout_length_steps: int = 0
    mean_bout_duration_ms: float = 0.0
    max_bout_duration_ms: float = 0.0
    total_bout_duration_ms: float = 0.0
    bout_step_rate_pct: float = 0.0  # % of total ICs that belong to bouts
    bouts: list[ToeWalkingBout] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_bouts": self.total_bouts,
            "total_bout_steps": self.total_bout_steps,
            "mean_bout_length_steps": round(self.mean_bout_length_steps, 2),
            "max_bout_length_steps": self.max_bout_length_steps,
            "mean_bout_duration_ms": round(self.mean_bout_duration_ms, 2),
            "max_bout_duration_ms": round(self.max_bout_duration_ms, 2),
            "total_bout_duration_ms": round(self.total_bout_duration_ms, 2),
            "bout_step_rate_pct": round(self.bout_step_rate_pct, 2),
            "bouts": [b.to_dict() for b in self.bouts],
        }


def detect_toe_walking_bouts(
    events: list[GaitEvent],
    min_consecutive: int = 2,
) -> list[ToeWalkingBout]:
    """Segment runs of consecutive forefoot-first ICs into bouts.

    Parameters
    ----------
    events : list[GaitEvent]
        Full event list (IC + TO, both feet).  Only IC events with a
        non-None ``contact_type`` are considered.
    min_consecutive : int
        Minimum number of consecutive TOE_WALK ICs to form a bout.

    Returns
    -------
    list[ToeWalkingBout]
        Detected bouts sorted chronologically.
    """
    ic_events = sorted(
        [e for e in events if e.event_type == "IC" and e.contact_type is not None],
        key=lambda e: e.time_ms,
    )

    bouts: list[ToeWalkingBout] = []
    run: list[tuple[int, GaitEvent]] = []  # (index, event)

    def _flush_run() -> None:
        if len(run) >= min_consecutive:
            first_idx, first_ev = run[0]
            last_idx, last_ev = run[-1]
            bouts.append(ToeWalkingBout(
                start_idx=first_idx,
                end_idx=last_idx,
                start_time_ms=first_ev.time_ms,
                end_time_ms=last_ev.time_ms,
                num_steps=len(run),
                duration_ms=last_ev.time_ms - first_ev.time_ms,
                foot_sequence=[ev.foot for _, ev in run],
            ))

    for idx, ic in enumerate(ic_events):
        if ic.contact_type == ContactType.TOE_WALK:
            run.append((idx, ic))
        else:
            _flush_run()
            run = []
    _flush_run()

    return bouts


def compute_bout_summary(
    bouts: list[ToeWalkingBout],
    total_ic_count: int,
) -> BoutSummary:
    """Compute aggregate bout statistics.

    Parameters
    ----------
    bouts : list[ToeWalkingBout]
        Output from :func:`detect_toe_walking_bouts`.
    total_ic_count : int
        Total number of classified IC events (for rate calculation).
    """
    s = BoutSummary(bouts=bouts)
    if not bouts:
        return s

    lengths = [b.num_steps for b in bouts]
    durations = [b.duration_ms for b in bouts]

    s.total_bouts = len(bouts)
    s.total_bout_steps = sum(lengths)
    s.mean_bout_length_steps = float(sum(lengths) / len(lengths))
    s.max_bout_length_steps = max(lengths)
    s.mean_bout_duration_ms = float(sum(durations) / len(durations))
    s.max_bout_duration_ms = float(max(durations))
    s.total_bout_duration_ms = float(sum(durations))
    s.bout_step_rate_pct = (
        100.0 * s.total_bout_steps / total_ic_count
        if total_ic_count > 0
        else 0.0
    )
    return s
