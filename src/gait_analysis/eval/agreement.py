"""
Continuous-metric agreement utilities: MAE, ICC, Bland-Altman.

All functions work with plain numpy arrays and do not require
heavy optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def compute_mae(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Mean Absolute Error between two matched vectors."""
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(predicted) == 0:
        return 0.0
    return float(np.mean(np.abs(predicted - reference)))


# ---------------------------------------------------------------------------
# ICC (two-way random, single measures, absolute agreement — ICC(2,1))
# ---------------------------------------------------------------------------

@dataclass
class ICCResult:
    """Intraclass correlation coefficient result."""
    icc: float = 0.0
    f_value: float = 0.0
    df_between: int = 0
    df_within: int = 0
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


def compute_icc(
    ratings: np.ndarray,
    icc_type: str = "ICC2,1",
    alpha: float = 0.05,
) -> ICCResult:
    """Compute ICC for a ratings matrix (subjects x raters).

    Parameters
    ----------
    ratings : ndarray (n_subjects, n_raters)
        Each row is one subject; each column is one rater / method.
    icc_type : str
        Only ``"ICC2,1"`` (two-way random, single measures, absolute
        agreement) is implemented.
    alpha : float
        Significance level for confidence intervals (default 0.05 = 95% CI).

    Returns
    -------
    ICCResult
        Contains ICC value, ANOVA statistics, p-value, and CI.
    """
    ratings = np.asarray(ratings, dtype=float)
    n, k = ratings.shape
    if n < 2 or k < 2:
        return ICCResult()

    grand_mean = np.mean(ratings)
    ss_total = np.sum((ratings - grand_mean) ** 2)

    row_means = np.mean(ratings, axis=1)
    col_means = np.mean(ratings, axis=0)

    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    ss_columns = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_between - ss_columns

    df_between = n - 1
    df_columns = k - 1
    df_error = df_between * df_columns

    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 1
    ms_columns = ss_columns / df_columns if df_columns > 0 else 0

    # ICC(2,1) — two-way random, single measures, absolute agreement
    denom = ms_between + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    icc = (ms_between - ms_error) / denom if denom > 0 else 0.0

    f_val = ms_between / ms_error if ms_error > 0 else 0.0

    result = ICCResult(
        icc=float(np.clip(icc, -1.0, 1.0)),
        f_value=float(f_val),
        df_between=int(df_between),
        df_within=int(df_error),
    )

    # p-value and CI via the F-distribution (scipy optional)
    try:
        from scipy.stats import f as f_dist

        if ms_error > 0 and df_between > 0 and df_error > 0:
            result.p_value = float(1.0 - f_dist.cdf(f_val, df_between, df_error))

            # McGraw & Wong (1996) CI for ICC(2,1)
            f_l = f_dist.ppf(1.0 - alpha / 2, df_between, df_error)
            f_u = f_dist.ppf(alpha / 2, df_between, df_error)

            icc_l = (f_val / f_l - 1) / (f_val / f_l + k - 1) if f_l > 0 else -1.0
            icc_u = (f_val / f_u - 1) / (f_val / f_u + k - 1) if f_u > 0 else 1.0

            result.ci_lower = float(np.clip(icc_l, -1.0, 1.0))
            result.ci_upper = float(np.clip(icc_u, -1.0, 1.0))
    except ImportError:
        pass

    return result


# ---------------------------------------------------------------------------
# Bland-Altman
# ---------------------------------------------------------------------------

@dataclass
class BlandAltmanResult:
    """Bland-Altman agreement analysis result."""
    mean_diff: float = 0.0       # bias
    std_diff: float = 0.0
    upper_loa: float = 0.0       # mean + 1.96 * std
    lower_loa: float = 0.0       # mean - 1.96 * std
    means: Optional[np.ndarray] = None   # (x+y)/2 per pair
    diffs: Optional[np.ndarray] = None   # x-y per pair
    n: int = 0

    def to_dict(self) -> dict:
        return {
            "mean_diff_bias": round(self.mean_diff, 4),
            "std_diff": round(self.std_diff, 4),
            "upper_loa": round(self.upper_loa, 4),
            "lower_loa": round(self.lower_loa, 4),
            "n": self.n,
        }


def bland_altman(
    method_a: np.ndarray,
    method_b: np.ndarray,
    confidence: float = 1.96,
) -> BlandAltmanResult:
    """Compute Bland-Altman difference statistics.

    Parameters
    ----------
    method_a, method_b : ndarray
        Matched measurement vectors from two methods / raters.
    confidence : float
        Z-multiplier for limits of agreement (default 1.96 for 95%).

    Returns
    -------
    BlandAltmanResult
    """
    a = np.asarray(method_a, dtype=float)
    b = np.asarray(method_b, dtype=float)
    if len(a) == 0:
        return BlandAltmanResult()
    diffs = a - b
    means = (a + b) / 2.0
    md = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    return BlandAltmanResult(
        mean_diff=md,
        std_diff=sd,
        upper_loa=md + confidence * sd,
        lower_loa=md - confidence * sd,
        means=means,
        diffs=diffs,
        n=len(a),
    )
