"""
ROC / AUC analysis for binary classification of gait metrics.

Use case: discriminating ASD-toe-walking vs. controls using a single
continuous gait metric as a classifier.

Requires ``scikit-learn`` for full ROC curve computation; a lightweight
AUC-only fallback is provided otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class ROCResult:
    """ROC curve and AUC for a single metric."""
    auc: float = 0.0
    fpr: Optional[np.ndarray] = None
    tpr: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None
    optimal_threshold: Optional[float] = None
    optimal_sensitivity: Optional[float] = None
    optimal_specificity: Optional[float] = None
    metric_name: str = ""

    def to_dict(self) -> dict:
        d: dict = {
            "metric_name": self.metric_name,
            "auc": round(self.auc, 4),
        }
        if self.optimal_threshold is not None:
            d["optimal_threshold"] = round(self.optimal_threshold, 4)
            d["optimal_sensitivity"] = round(self.optimal_sensitivity or 0, 4)
            d["optimal_specificity"] = round(self.optimal_specificity or 0, 4)
        return d


def _trapezoidal_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Simple trapezoidal AUC."""
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def compute_roc_auc(
    y_true: Sequence[int],
    scores: Sequence[float],
    metric_name: str = "",
) -> ROCResult:
    """Compute ROC curve and AUC for a binary outcome.

    Parameters
    ----------
    y_true : sequence of int
        Binary labels (0 = control, 1 = positive / toe-walker).
    scores : sequence of float
        Continuous metric values (higher assumed more positive unless
        scikit-learn handles it).
    metric_name : str
        Human-readable name for reporting.

    Returns
    -------
    ROCResult
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return ROCResult(metric_name=metric_name)

    try:
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        auc_val = float(roc_auc_score(y_true, scores))
        # Youden's J for optimal cut-off
        j = tpr - fpr
        best = int(np.argmax(j))
        return ROCResult(
            auc=auc_val,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            optimal_threshold=float(thresholds[best]),
            optimal_sensitivity=float(tpr[best]),
            optimal_specificity=float(1 - fpr[best]),
            metric_name=metric_name,
        )
    except ImportError:
        # Fallback: simple AUC via sorting
        order = np.argsort(scores)
        y_sorted = y_true[order]
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        if n_pos == 0 or n_neg == 0:
            return ROCResult(metric_name=metric_name)
        tp = 0
        fp = 0
        fpr_list = [0.0]
        tpr_list = [0.0]
        for yi in reversed(y_sorted):
            if yi == 1:
                tp += 1
            else:
                fp += 1
            fpr_list.append(fp / n_neg)
            tpr_list.append(tp / n_pos)
        fpr_arr = np.array(fpr_list)
        tpr_arr = np.array(tpr_list)
        auc_val = _trapezoidal_auc(fpr_arr, tpr_arr)
        return ROCResult(
            auc=auc_val,
            fpr=fpr_arr,
            tpr=tpr_arr,
            metric_name=metric_name,
        )
