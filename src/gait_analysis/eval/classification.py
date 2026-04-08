"""
Classification metrics for step-label evaluation.

Compares predicted contact-type labels (heel_strike / toe_walk / flat)
against expert (gold-standard) annotations at the step level.

Requires ``scikit-learn`` for full metrics; a lightweight fallback is
provided when scikit-learn is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class StepLabelReport:
    """Classification report for step-level contact type labels."""
    accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: Optional[list[list[int]]] = None
    labels: list[str] = field(default_factory=list)
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_class": {
                k: {m: round(v, 4) for m, v in cls.items()}
                for k, cls in self.per_class.items()
            },
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
            "n_samples": self.n_samples,
        }


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _fallback_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: list[str],
) -> StepLabelReport:
    """Compute metrics without scikit-learn."""
    n = len(y_true)
    report = StepLabelReport(n_samples=n, labels=labels)
    if n == 0:
        return report

    report.accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n

    cm = [[0] * len(labels) for _ in labels]
    label_idx = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        ti = label_idx.get(t)
        pi = label_idx.get(p)
        if ti is not None and pi is not None:
            cm[ti][pi] += 1
    report.confusion_matrix = cm

    precisions, recalls, f1s = [], [], []
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(len(labels))) - tp
        fn = sum(cm[i]) - tp
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f = _safe_div(2 * p * r, p + r)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        report.per_class[lab] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "support": sum(cm[i]),
        }

    report.macro_precision = float(np.mean(precisions)) if precisions else 0.0
    report.macro_recall = float(np.mean(recalls)) if recalls else 0.0
    report.macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return report


def evaluate_step_labels(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[list[str]] = None,
) -> StepLabelReport:
    """Evaluate predicted step labels against ground truth.

    Parameters
    ----------
    y_true, y_pred : sequence of str
        Parallel sequences of contact-type labels
        (e.g. ``"heel_strike"``, ``"toe_walk"``, ``"flat"``).
    labels : list[str], optional
        Label set.  Inferred from data if not provided.

    Returns
    -------
    StepLabelReport
    """
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))

    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix as sk_cm,
        )
        report = StepLabelReport(n_samples=len(y_true), labels=labels)
        report.accuracy = float(accuracy_score(y_true, y_pred))
        p, r, f, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0,
        )
        report.macro_precision = float(np.mean(p))
        report.macro_recall = float(np.mean(r))
        report.macro_f1 = float(np.mean(f))
        for i, lab in enumerate(labels):
            report.per_class[lab] = {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(sup[i]),
            }
        cm = sk_cm(y_true, y_pred, labels=labels)
        report.confusion_matrix = cm.tolist()
        return report
    except ImportError:
        return _fallback_metrics(y_true, y_pred, labels)


def confusion_matrix_from_labels(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[list[str]] = None,
) -> tuple[list[list[int]], list[str]]:
    """Return (matrix, labels) for a confusion matrix.

    Rows = true, columns = predicted.
    """
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        ti, pi = idx.get(t), idx.get(p)
        if ti is not None and pi is not None:
            cm[ti][pi] += 1
    return cm, labels
