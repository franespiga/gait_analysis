"""Unit tests for gait_metrics.py — step records and summary computation."""

import numpy as np
import pytest

from gait_analysis.gait_events import GaitEvent, ContactType
from gait_analysis.gait_metrics import (
    StepRecord,
    GaitSummary,
    build_step_records,
    compute_summary,
    _compute_cross_foot_step_times,
)


def _make_events() -> list[GaitEvent]:
    """Create a minimal set of IC/TO events for both feet.

    Timeline (ms):
        0    left  IC (heel_strike)
        200  left  TO
        300  right IC (toe_walk)
        500  right TO
        600  left  IC (heel_strike)
        800  left  TO
        900  right IC (flat)
        1100 right TO
    """
    return [
        GaitEvent("IC", "left",  0,    0.0, ContactType.HEEL_STRIKE, x=10, y=300),
        GaitEvent("TO", "left",  6,  200.0),
        GaitEvent("IC", "right", 9,  300.0, ContactType.TOE_WALK, x=50, y=300),
        GaitEvent("TO", "right", 15, 500.0),
        GaitEvent("IC", "left",  18, 600.0, ContactType.HEEL_STRIKE, x=90, y=300),
        GaitEvent("TO", "left",  24, 800.0),
        GaitEvent("IC", "right", 27, 900.0, ContactType.FLAT, x=130, y=300),
        GaitEvent("TO", "right", 33, 1100.0),
    ]


class TestBuildStepRecords:
    def test_basic(self):
        events = _make_events()
        records = build_step_records(events)
        assert len(records) == 4  # 4 IC events
        # Check contact types carried through
        types = [r.contact_type for r in records]
        assert "heel_strike" in types
        assert "toe_walk" in types

    def test_stance_time(self):
        events = _make_events()
        records = build_step_records(events)
        left_recs = [r for r in records if r.foot == "left"]
        assert left_recs[0].stance_time_ms == 200.0

    def test_stride_length(self):
        events = _make_events()
        records = build_step_records(events)
        left_recs = [r for r in records if r.foot == "left"]
        # First left IC at x=10, next left IC at x=90 → stride = 80
        assert left_recs[0].stride_length_px is not None
        assert abs(left_recs[0].stride_length_px - 80.0) < 1.0

    def test_scale_conversion(self):
        events = _make_events()
        records = build_step_records(events, scale_m_per_px=0.01)
        for r in records:
            if r.step_length_px is not None:
                assert r.step_length_m is not None
                assert abs(r.step_length_m - r.step_length_px * 0.01) < 1e-6


class TestComputeCrossFootStepTimes:
    def test_alternating_feet(self):
        records = [
            StepRecord("left",  0.0,   200.0),
            StepRecord("right", 300.0, 500.0),
            StepRecord("left",  600.0, 800.0),
            StepRecord("right", 900.0, 1100.0),
        ]
        cross = _compute_cross_foot_step_times(records)
        # left→right: 300, right→left: 300, left→right: 300
        assert len(cross) == 3
        assert all(abs(t - 300.0) < 1e-6 for t in cross)


class TestComputeSummary:
    def test_rates(self):
        events = _make_events()
        records = build_step_records(events)
        summary = compute_summary(records, duration_s=1.1)
        assert summary.heel_strike_count == 2
        assert summary.toe_walk_count == 1
        assert summary.flat_count == 1
        assert abs(summary.heel_strike_rate_pct - 50.0) < 0.1
        assert abs(summary.toe_walk_rate_pct - 25.0) < 0.1
        assert abs(summary.flat_rate_pct - 25.0) < 0.1

    def test_stride_time_not_doubled(self):
        """Stride time should come from same-foot IC intervals, not 2×step."""
        events = _make_events()
        records = build_step_records(events)
        summary = compute_summary(records, duration_s=1.1)
        # Left IC intervals: 600-0=600 ms. Right IC intervals: 900-300=600 ms.
        assert abs(summary.stride_time_mean_ms - 600.0) < 1.0

    def test_step_time_cross_foot(self):
        """Step time should be opposite-foot IC intervals."""
        events = _make_events()
        records = build_step_records(events)
        summary = compute_summary(records, duration_s=1.1)
        # Cross-foot: 300-0=300, 600-300=300, 900-600=300 → mean 300
        assert abs(summary.step_time_mean_ms - 300.0) < 1.0

    def test_cadence(self):
        events = _make_events()
        records = build_step_records(events)
        summary = compute_summary(records, duration_s=1.1)
        expected_cadence = (4 / 1.1) * 60
        assert abs(summary.cadence_steps_per_min - expected_cadence) < 0.5

    def test_symmetry_index(self):
        events = _make_events()
        records = build_step_records(events)
        summary = compute_summary(records, duration_s=1.1)
        # Both feet have same stride times → symmetry ≈ 0
        assert summary.symmetry_index_step_time < 0.1

    def test_empty_records(self):
        summary = compute_summary([], duration_s=5.0)
        assert summary.total_steps == 0
        assert summary.cadence_steps_per_min == 0.0

    def test_to_dict_has_rates(self):
        events = _make_events()
        records = build_step_records(events)
        summary = compute_summary(records, duration_s=1.1)
        d = summary.to_dict()
        assert "heel_strike_rate_pct" in d
        assert "toe_walk_rate_pct" in d
        assert "flat_rate_pct" in d
        assert "stride_time_cv_percent" in d
