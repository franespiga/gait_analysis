"""Unit tests for eval/ package — classification, agreement, ROC."""

import numpy as np
import pytest

from gait_analysis.eval.classification import (
    evaluate_step_labels,
    confusion_matrix_from_labels,
)
from gait_analysis.eval.agreement import (
    compute_mae,
    compute_icc,
    bland_altman,
)
from gait_analysis.eval.roc import compute_roc_auc


class TestClassification:
    def test_perfect(self):
        labels = ["heel_strike", "toe_walk", "flat"]
        y_true = labels * 5
        y_pred = labels * 5
        report = evaluate_step_labels(y_true, y_pred)
        assert report.accuracy == 1.0
        assert report.macro_f1 == 1.0

    def test_all_wrong(self):
        y_true = ["heel_strike"] * 5
        y_pred = ["toe_walk"] * 5
        report = evaluate_step_labels(y_true, y_pred)
        assert report.accuracy == 0.0

    def test_mixed(self):
        y_true = ["heel_strike", "toe_walk", "heel_strike", "flat"]
        y_pred = ["heel_strike", "heel_strike", "heel_strike", "flat"]
        report = evaluate_step_labels(y_true, y_pred)
        assert 0 < report.accuracy < 1.0
        assert report.n_samples == 4

    def test_to_dict(self):
        y_true = ["heel_strike", "toe_walk"]
        y_pred = ["heel_strike", "toe_walk"]
        d = evaluate_step_labels(y_true, y_pred).to_dict()
        assert "accuracy" in d
        assert "per_class" in d

    def test_confusion_matrix(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "a", "b"]
        cm, labels = confusion_matrix_from_labels(y_true, y_pred)
        assert len(cm) == 2
        assert cm[0][0] == 1  # a correct
        assert cm[1][1] == 1  # b correct


class TestAgreement:
    def test_mae_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        assert compute_mae(a, a) == 0.0

    def test_mae_positive(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        assert compute_mae(a, b) == 1.0

    def test_icc_perfect(self):
        ratings = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
        result = compute_icc(ratings)
        assert result.icc > 0.99

    def test_icc_zero_agreement(self):
        # One rater gives opposite pattern
        ratings = np.array([[1, 4], [2, 3], [3, 2], [4, 1]], dtype=float)
        result = compute_icc(ratings)
        assert result.icc < 0.5

    def test_bland_altman_no_bias(self):
        a = np.array([10, 20, 30, 40, 50], dtype=float)
        b = a.copy()
        ba = bland_altman(a, b)
        assert ba.mean_diff == 0.0
        assert ba.n == 5

    def test_bland_altman_with_bias(self):
        a = np.array([10, 20, 30], dtype=float)
        b = np.array([12, 22, 32], dtype=float)
        ba = bland_altman(a, b)
        assert abs(ba.mean_diff - (-2.0)) < 0.01


class TestROC:
    def test_perfect_separation(self):
        y_true = [0, 0, 0, 1, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        result = compute_roc_auc(y_true, scores, metric_name="test")
        assert result.auc > 0.99
        assert result.metric_name == "test"

    def test_random(self):
        np.random.seed(42)
        y_true = [0] * 50 + [1] * 50
        scores = np.random.rand(100).tolist()
        result = compute_roc_auc(y_true, scores)
        assert 0.2 < result.auc < 0.8  # roughly chance

    def test_single_class(self):
        result = compute_roc_auc([0, 0, 0], [0.1, 0.2, 0.3])
        assert result.auc == 0.0  # can't compute

    def test_to_dict(self):
        result = compute_roc_auc([0, 1], [0.1, 0.9], metric_name="x")
        d = result.to_dict()
        assert "auc" in d
