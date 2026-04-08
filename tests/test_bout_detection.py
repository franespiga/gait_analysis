"""Unit tests for bout_detection.py — toe-walking bout segmentation."""

import pytest

from gait_analysis.gait_events import GaitEvent, ContactType
from gait_analysis.bout_detection import (
    ToeWalkingBout,
    BoutSummary,
    detect_toe_walking_bouts,
    compute_bout_summary,
)


def _ic(foot, time_ms, ct):
    return GaitEvent("IC", foot, int(time_ms / 33.3), time_ms, ct)


class TestDetectBouts:
    def test_no_toe_walks(self):
        events = [
            _ic("left", 0, ContactType.HEEL_STRIKE),
            _ic("right", 300, ContactType.HEEL_STRIKE),
            _ic("left", 600, ContactType.HEEL_STRIKE),
        ]
        bouts = detect_toe_walking_bouts(events)
        assert len(bouts) == 0

    def test_single_toe_walk_below_min(self):
        events = [
            _ic("left", 0, ContactType.HEEL_STRIKE),
            _ic("right", 300, ContactType.TOE_WALK),
            _ic("left", 600, ContactType.HEEL_STRIKE),
        ]
        bouts = detect_toe_walking_bouts(events, min_consecutive=2)
        assert len(bouts) == 0

    def test_two_consecutive_toe_walks(self):
        events = [
            _ic("left", 0, ContactType.HEEL_STRIKE),
            _ic("right", 300, ContactType.TOE_WALK),
            _ic("left", 600, ContactType.TOE_WALK),
            _ic("right", 900, ContactType.HEEL_STRIKE),
        ]
        bouts = detect_toe_walking_bouts(events, min_consecutive=2)
        assert len(bouts) == 1
        assert bouts[0].num_steps == 2
        assert bouts[0].start_time_ms == 300.0
        assert bouts[0].end_time_ms == 600.0

    def test_long_bout(self):
        events = [_ic("left", t * 300, ContactType.TOE_WALK) for t in range(10)]
        bouts = detect_toe_walking_bouts(events, min_consecutive=2)
        assert len(bouts) == 1
        assert bouts[0].num_steps == 10

    def test_multiple_bouts(self):
        events = [
            _ic("left", 0, ContactType.TOE_WALK),
            _ic("right", 300, ContactType.TOE_WALK),
            _ic("left", 600, ContactType.HEEL_STRIKE),
            _ic("right", 900, ContactType.TOE_WALK),
            _ic("left", 1200, ContactType.TOE_WALK),
            _ic("right", 1500, ContactType.TOE_WALK),
        ]
        bouts = detect_toe_walking_bouts(events, min_consecutive=2)
        assert len(bouts) == 2

    def test_foot_sequence_tracked(self):
        events = [
            _ic("left", 0, ContactType.TOE_WALK),
            _ic("right", 300, ContactType.TOE_WALK),
            _ic("left", 600, ContactType.TOE_WALK),
        ]
        bouts = detect_toe_walking_bouts(events, min_consecutive=2)
        assert bouts[0].foot_sequence == ["left", "right", "left"]

    def test_to_events_ignored(self):
        events = [
            _ic("left", 0, ContactType.TOE_WALK),
            GaitEvent("TO", "left", 3, 100.0),
            _ic("right", 300, ContactType.TOE_WALK),
        ]
        bouts = detect_toe_walking_bouts(events, min_consecutive=2)
        assert len(bouts) == 1

    def test_unknown_breaks_bout(self):
        events = [
            _ic("left", 0, ContactType.TOE_WALK),
            _ic("right", 300, ContactType.UNKNOWN),
            _ic("left", 600, ContactType.TOE_WALK),
        ]
        bouts = detect_toe_walking_bouts(events, min_consecutive=2)
        assert len(bouts) == 0


class TestBoutSummary:
    def test_basic(self):
        events = [
            _ic("left", 0, ContactType.TOE_WALK),
            _ic("right", 300, ContactType.TOE_WALK),
            _ic("left", 600, ContactType.TOE_WALK),
            _ic("right", 900, ContactType.HEEL_STRIKE),
        ]
        bouts = detect_toe_walking_bouts(events, min_consecutive=2)
        summary = compute_bout_summary(bouts, total_ic_count=4)
        assert summary.total_bouts == 1
        assert summary.total_bout_steps == 3
        assert summary.mean_bout_length_steps == 3.0
        assert abs(summary.bout_step_rate_pct - 75.0) < 0.1

    def test_empty(self):
        summary = compute_bout_summary([], total_ic_count=10)
        assert summary.total_bouts == 0
        assert summary.bout_step_rate_pct == 0.0

    def test_to_dict(self):
        bouts = detect_toe_walking_bouts(
            [_ic("l", 0, ContactType.TOE_WALK), _ic("r", 300, ContactType.TOE_WALK)],
            min_consecutive=2,
        )
        summary = compute_bout_summary(bouts, total_ic_count=2)
        d = summary.to_dict()
        assert "total_bouts" in d
        assert "bouts" in d
        assert isinstance(d["bouts"], list)
