"""
Toe-walking severity index — composite score for clinical classification.

The severity index combines:
- forefoot-first contact rate (weight: 0.40)
- bout step rate — proportion of steps inside bouts (weight: 0.30)
- normalised mean bout length (weight: 0.20)
- stride-time variability penalty (weight: 0.10)

Each component is rescaled to 0–100 then combined.  Cut-offs for
``classification`` (none / mild / moderate / severe) are configurable.

**Clinical note:** These thresholds are heuristic defaults and should be
validated against expert-rated severity in the study population before
use in clinical decision-making.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .gait_metrics import GaitSummary
from .bout_detection import BoutSummary


_DEFAULT_WEIGHTS = {
    "forefoot_rate": 0.40,
    "bout_step_rate": 0.30,
    "mean_bout_length": 0.20,
    "stride_cv_penalty": 0.10,
}

_DEFAULT_CUT_OFFS = {
    "none": 10.0,
    "mild": 35.0,
    "moderate": 65.0,
    # >= moderate threshold → severe
}


@dataclass
class SeverityScore:
    """Composite toe-walking severity result."""
    overall: float = 0.0        # 0–100
    classification: str = "none"
    component_forefoot_rate: float = 0.0
    component_bout_step_rate: float = 0.0
    component_mean_bout_length: float = 0.0
    component_stride_cv_penalty: float = 0.0

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 2),
            "classification": self.classification,
            "component_forefoot_rate": round(self.component_forefoot_rate, 2),
            "component_bout_step_rate": round(self.component_bout_step_rate, 2),
            "component_mean_bout_length": round(self.component_mean_bout_length, 2),
            "component_stride_cv_penalty": round(self.component_stride_cv_penalty, 2),
        }


def compute_severity_index(
    summary: GaitSummary,
    bout_summary: BoutSummary,
    weights: Optional[dict[str, float]] = None,
    cut_offs: Optional[dict[str, float]] = None,
    max_bout_length_for_scaling: int = 20,
    max_stride_cv_for_scaling: float = 30.0,
) -> SeverityScore:
    """Compute the composite toe-walking severity index.

    Parameters
    ----------
    summary : GaitSummary
        Core spatiotemporal gait summary.
    bout_summary : BoutSummary
        Bout-level statistics.
    weights : dict, optional
        Component weights (must sum to 1.0).
    cut_offs : dict, optional
        Thresholds for classification labels.
    max_bout_length_for_scaling : int
        Bout lengths are capped at this value before 0-100 scaling.
    max_stride_cv_for_scaling : float
        Stride-time CV% is capped at this value before 0-100 scaling.
    """
    w = weights or _DEFAULT_WEIGHTS
    co = cut_offs or _DEFAULT_CUT_OFFS

    # Component 1: forefoot rate (already 0-100)
    c_ff = min(100.0, summary.toe_walk_rate_pct)

    # Component 2: bout step rate (already 0-100)
    c_bout_rate = min(100.0, bout_summary.bout_step_rate_pct)

    # Component 3: mean bout length → 0-100
    c_bout_len = min(
        100.0,
        (bout_summary.mean_bout_length_steps / max_bout_length_for_scaling) * 100.0
    ) if bout_summary.total_bouts > 0 else 0.0

    # Component 4: stride-time variability penalty → 0-100
    c_cv = min(
        100.0,
        (summary.stride_time_cv_percent / max_stride_cv_for_scaling) * 100.0
    )

    score = SeverityScore(
        component_forefoot_rate=c_ff,
        component_bout_step_rate=c_bout_rate,
        component_mean_bout_length=c_bout_len,
        component_stride_cv_penalty=c_cv,
    )

    score.overall = (
        w.get("forefoot_rate", 0.4) * c_ff
        + w.get("bout_step_rate", 0.3) * c_bout_rate
        + w.get("mean_bout_length", 0.2) * c_bout_len
        + w.get("stride_cv_penalty", 0.1) * c_cv
    )

    if score.overall < co.get("none", 10.0):
        score.classification = "none"
    elif score.overall < co.get("mild", 35.0):
        score.classification = "mild"
    elif score.overall < co.get("moderate", 65.0):
        score.classification = "moderate"
    else:
        score.classification = "severe"

    return score
