"""
Statistics utilities for gait analysis reporting.

Provides:
- descriptive statistics tables
- group comparison (parametric + non-parametric)
- effect sizes (Cohen's d, rank-biserial)
- repeated-trial aggregation by participant / condition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

@dataclass
class DescriptiveStats:
    """Standard descriptive statistics for a numeric vector."""
    n: int = 0
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    q25: float = 0.0
    q75: float = 0.0
    min: float = 0.0
    max: float = 0.0
    cv_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "median": round(self.median, 4),
            "q25": round(self.q25, 4),
            "q75": round(self.q75, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "cv_pct": round(self.cv_pct, 2),
        }


def descriptive(values: Sequence[float]) -> DescriptiveStats:
    """Compute descriptive statistics for a numeric sequence."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return DescriptiveStats()
    return DescriptiveStats(
        n=len(arr),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        median=float(np.median(arr)),
        q25=float(np.percentile(arr, 25)),
        q75=float(np.percentile(arr, 75)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        cv_pct=float(np.std(arr, ddof=1) / np.mean(arr) * 100) if np.mean(arr) != 0 and len(arr) > 1 else 0.0,
    )


def descriptive_table(
    metric_dict: dict[str, Sequence[float]],
) -> list[dict]:
    """Compute descriptive statistics for multiple metrics.

    Parameters
    ----------
    metric_dict : dict
        ``{metric_name: values}``

    Returns
    -------
    list[dict]
        One dict per metric with ``metric`` key and all stats.
    """
    rows = []
    for name, vals in metric_dict.items():
        row = descriptive(vals).to_dict()
        row["metric"] = name
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Group comparison
# ---------------------------------------------------------------------------

@dataclass
class GroupComparisonResult:
    """Result of a two-group comparison."""
    test: str = ""            # "t_test" | "mann_whitney" | "N/A"
    statistic: float = 0.0
    p_value: float = 1.0
    group_a_n: int = 0
    group_b_n: int = 0
    group_a_mean: float = 0.0
    group_b_mean: float = 0.0
    effect_size: float = 0.0
    effect_size_name: str = ""

    def to_dict(self) -> dict:
        return {
            "test": self.test,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "group_a_n": self.group_a_n,
            "group_b_n": self.group_b_n,
            "group_a_mean": round(self.group_a_mean, 4),
            "group_b_mean": round(self.group_b_mean, 4),
            "effect_size": round(self.effect_size, 4),
            "effect_size_name": self.effect_size_name,
        }


def compare_groups(
    group_a: Sequence[float],
    group_b: Sequence[float],
    parametric: bool = True,
    equal_var: bool = False,
) -> GroupComparisonResult:
    """Compare two independent groups.

    Parameters
    ----------
    group_a, group_b : array-like
        Values for each group.
    parametric : bool
        If True, use Welch's t-test; else Mann-Whitney U.
    equal_var : bool
        Passed to ``scipy.stats.ttest_ind`` when parametric.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    result = GroupComparisonResult(
        group_a_n=len(a),
        group_b_n=len(b),
        group_a_mean=float(np.mean(a)) if len(a) else 0.0,
        group_b_mean=float(np.mean(b)) if len(b) else 0.0,
    )
    if len(a) < 2 or len(b) < 2:
        result.test = "N/A"
        return result

    if not _HAS_SCIPY:
        result.test = "N/A (scipy not installed)"
        return result

    if parametric:
        stat, p = sp_stats.ttest_ind(a, b, equal_var=equal_var)
        result.test = "welch_t" if not equal_var else "student_t"
        result.statistic = float(stat)
        result.p_value = float(p)
        result.effect_size = cohens_d(a, b)
        result.effect_size_name = "cohens_d"
    else:
        stat, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        result.test = "mann_whitney_u"
        result.statistic = float(stat)
        result.p_value = float(p)
        result.effect_size = rank_biserial(a, b)
        result.effect_size_name = "rank_biserial_r"

    return result


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d (pooled standard deviation)."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def rank_biserial(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Rank-biserial correlation from Mann-Whitney U."""
    if not _HAS_SCIPY:
        return 0.0
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) < 1 or len(b) < 1:
        return 0.0
    u, _ = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
    n = len(a) * len(b)
    return float(1 - 2 * u / n) if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Repeated-trial aggregation
# ---------------------------------------------------------------------------

def aggregate_trials(
    records: list[dict],
    participant_col: str = "participant_id",
    metric_cols: Optional[list[str]] = None,
) -> list[dict]:
    """Aggregate metric columns across repeated trials per participant.

    Parameters
    ----------
    records : list[dict]
        One dict per trial / video with at least ``participant_col``.
    participant_col : str
        Column name for participant grouping.
    metric_cols : list[str], optional
        Columns to aggregate (mean, std, n).  If None, all numeric
        keys except ``participant_col`` are aggregated.

    Returns
    -------
    list[dict]
        One dict per participant with mean / std / n for each metric.
    """
    if not records:
        return []

    by_participant: dict[str, list[dict]] = {}
    for r in records:
        pid = r.get(participant_col, "unknown")
        by_participant.setdefault(str(pid), []).append(r)

    if metric_cols is None:
        metric_cols = [
            k for k in records[0]
            if k != participant_col and isinstance(records[0][k], (int, float))
        ]

    out = []
    for pid, trials in by_participant.items():
        row: dict = {participant_col: pid, "n_trials": len(trials)}
        for mc in metric_cols:
            vals = [t[mc] for t in trials if mc in t and t[mc] is not None]
            nums = [v for v in vals if isinstance(v, (int, float))]
            if nums:
                row[f"{mc}_mean"] = float(np.mean(nums))
                row[f"{mc}_std"] = float(np.std(nums, ddof=1)) if len(nums) > 1 else 0.0
            else:
                row[f"{mc}_mean"] = None
                row[f"{mc}_std"] = None
        out.append(row)
    return out
