"""
Evaluation utilities for gait analysis validation.

Provides classification metrics, continuous-metric agreement, and
ROC/AUC analysis for comparing predicted gait labels and metrics
against expert annotations.

All heavy optional dependencies (scikit-learn, etc.) are imported
lazily so the core package works without them.
"""

from .classification import (
    evaluate_step_labels,
    StepLabelReport,
    confusion_matrix_from_labels,
)
from .agreement import compute_mae, compute_icc, bland_altman
from .roc import compute_roc_auc

__all__ = [
    "evaluate_step_labels",
    "StepLabelReport",
    "confusion_matrix_from_labels",
    "compute_mae",
    "compute_icc",
    "bland_altman",
    "compute_roc_auc",
]
