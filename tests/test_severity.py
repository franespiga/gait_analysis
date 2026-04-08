"""Unit tests for severity.py — toe-walking severity index."""

import pytest

from gait_analysis.gait_metrics import GaitSummary
from gait_analysis.bout_detection import BoutSummary, ToeWalkingBout
from gait_analysis.severity import compute_severity_index, SeverityScore


def _make_summary(toe_walk_rate=0.0, stride_cv=0.0):
    s = GaitSummary()
    s.toe_walk_rate_pct = toe_walk_rate
    s.stride_time_cv_percent = stride_cv
    s.total_steps = 20
    return s


def _make_bout_summary(bout_rate=0.0, mean_length=0.0, n_bouts=0):
    return BoutSummary(
        total_bouts=n_bouts,
        total_bout_steps=int(mean_length * n_bouts),
        mean_bout_length_steps=mean_length,
        max_bout_length_steps=int(mean_length),
        bout_step_rate_pct=bout_rate,
    )


class TestSeverityIndex:
    def test_no_toe_walking(self):
        score = compute_severity_index(
            _make_summary(toe_walk_rate=0),
            _make_bout_summary(),
        )
        assert score.overall < 10
        assert score.classification == "none"

    def test_severe(self):
        score = compute_severity_index(
            _make_summary(toe_walk_rate=90, stride_cv=25),
            _make_bout_summary(bout_rate=80, mean_length=15, n_bouts=5),
        )
        assert score.overall > 65
        assert score.classification == "severe"

    def test_mild(self):
        score = compute_severity_index(
            _make_summary(toe_walk_rate=25, stride_cv=5),
            _make_bout_summary(bout_rate=15, mean_length=3, n_bouts=2),
        )
        assert 10 <= score.overall < 35
        assert score.classification == "mild"

    def test_to_dict(self):
        score = compute_severity_index(
            _make_summary(toe_walk_rate=50),
            _make_bout_summary(bout_rate=40, mean_length=5, n_bouts=3),
        )
        d = score.to_dict()
        assert "overall" in d
        assert "classification" in d
        assert "component_forefoot_rate" in d

    def test_custom_weights(self):
        weights = {
            "forefoot_rate": 1.0,
            "bout_step_rate": 0.0,
            "mean_bout_length": 0.0,
            "stride_cv_penalty": 0.0,
        }
        score = compute_severity_index(
            _make_summary(toe_walk_rate=50),
            _make_bout_summary(),
            weights=weights,
        )
        assert abs(score.overall - 50.0) < 0.1
