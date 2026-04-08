"""Unit tests for extended_metrics.py."""

import numpy as np
import pytest

from gait_analysis.gait_events import GaitEvent, ContactType
from gait_analysis.extended_metrics import (
    compute_heel_rise_timing,
    compute_plantarflexion_proxy,
    compute_foot_strike_variability,
    compute_com_sway_proxy,
)


def _make_kp6(T, heel_y_fn, toe_y_fn, conf=0.9):
    """Build (T, 6, 3) from callables heel_y_fn(t), toe_y_fn(t)."""
    kp = np.zeros((T, 6, 3))
    for t in range(T):
        hy = heel_y_fn(t)
        ty = toe_y_fn(t)
        kp[t, 0] = [50, hy, conf]
        kp[t, 1] = [70, ty, conf]
        kp[t, 2] = [65, ty, conf]
        kp[t, 3] = [150, hy + 5, conf]
        kp[t, 4] = [170, ty + 5, conf]
        kp[t, 5] = [165, ty + 5, conf]
    return kp


class TestHeelRiseTiming:
    def test_detects_rise(self):
        T = 60
        # Heel on ground (Y=300) until frame 15, then rises (Y decreases)
        def hy(t):
            return 300 if t < 15 else max(200, 300 - (t - 15) * 5)
        def ty(t):
            return 295

        kp = _make_kp6(T, hy, ty)
        times = np.arange(T, dtype=float) * 33.3
        events = [GaitEvent("IC", "left", 5, 5 * 33.3, ContactType.HEEL_STRIKE)]

        records = compute_heel_rise_timing(
            kp, times, events,
            rise_velocity_threshold=-1.0,
            max_search_ms=2000.0,  # long enough for frame 15+ to be found
        )
        assert len(records) == 1
        assert records[0].heel_rise_latency_ms is not None
        assert records[0].heel_rise_latency_ms > 0

    def test_no_rise_detected(self):
        T = 30
        kp = _make_kp6(T, lambda t: 300, lambda t: 295)
        times = np.arange(T, dtype=float) * 33.3
        events = [GaitEvent("IC", "left", 5, 5 * 33.3, ContactType.HEEL_STRIKE)]

        records = compute_heel_rise_timing(kp, times, events)
        assert records[0].heel_rise_latency_ms is None


class TestPlantarflexionProxy:
    def test_positive_pitch(self):
        """Toe lower than heel → positive pitch (plantarflexed)."""
        T = 20
        kp = _make_kp6(T, lambda t: 290, lambda t: 310)
        times = np.arange(T, dtype=float) * 33.3
        events = [GaitEvent("IC", "left", 10, 10 * 33.3, ContactType.TOE_WALK)]

        records = compute_plantarflexion_proxy(kp, times, events)
        assert len(records) == 1
        assert records[0].pitch_deg is not None
        assert records[0].pitch_deg > 0

    def test_negative_pitch(self):
        """Heel lower than toe → dorsiflexed proxy."""
        T = 20
        kp = _make_kp6(T, lambda t: 310, lambda t: 290)
        times = np.arange(T, dtype=float) * 33.3
        events = [GaitEvent("IC", "left", 10, 10 * 33.3, ContactType.HEEL_STRIKE)]

        records = compute_plantarflexion_proxy(kp, times, events)
        assert records[0].pitch_deg is not None
        assert records[0].pitch_deg < 0

    def test_low_confidence_returns_none(self):
        T = 20
        kp = _make_kp6(T, lambda t: 300, lambda t: 300, conf=0.1)
        times = np.arange(T, dtype=float) * 33.3
        events = [GaitEvent("IC", "left", 10, 333.0, ContactType.FLAT)]

        records = compute_plantarflexion_proxy(kp, times, events, conf_threshold=0.3)
        assert records[0].pitch_deg is None


class TestFootStrikeVariability:
    def test_all_same_type(self):
        events = [
            GaitEvent("IC", "left", i * 10, i * 333.0, ContactType.HEEL_STRIKE)
            for i in range(10)
        ]
        result = compute_foot_strike_variability(events)
        assert result.total_classified == 10
        assert result.shannon_entropy == 0.0  # no variability

    def test_mixed_types(self):
        events = [
            GaitEvent("IC", "left", 0, 0.0, ContactType.HEEL_STRIKE),
            GaitEvent("IC", "right", 1, 300.0, ContactType.TOE_WALK),
            GaitEvent("IC", "left", 2, 600.0, ContactType.HEEL_STRIKE),
            GaitEvent("IC", "right", 3, 900.0, ContactType.TOE_WALK),
        ]
        result = compute_foot_strike_variability(events)
        assert result.total_classified == 4
        assert result.shannon_entropy > 0

    def test_transition_matrix(self):
        events = [
            GaitEvent("IC", "l", 0, 0.0, ContactType.HEEL_STRIKE),
            GaitEvent("IC", "r", 1, 300.0, ContactType.TOE_WALK),
            GaitEvent("IC", "l", 2, 600.0, ContactType.HEEL_STRIKE),
        ]
        result = compute_foot_strike_variability(events)
        assert "heel_strike" in result.transition_matrix
        # heel_strike → toe_walk should have probability 1.0
        assert result.transition_matrix["heel_strike"]["toe_walk"] == 1.0


class TestCOMSwayProxy:
    def test_with_hips(self):
        T = 50
        kp = np.zeros((T, 10, 3))
        kp[:, 0, 0] = 100 + np.random.randn(T) * 5  # left hip x
        kp[:, 0, 1] = 200
        kp[:, 0, 2] = 0.9
        kp[:, 1, 0] = 200 + np.random.randn(T) * 5  # right hip x
        kp[:, 1, 1] = 200
        kp[:, 1, 2] = 0.9

        result = compute_com_sway_proxy(kp, hip_left_idx=0, hip_right_idx=1)
        assert result.n_valid_frames == T
        assert result.lateral_std_px > 0

    def test_six_keypoint_returns_empty(self):
        kp = np.zeros((20, 6, 3))
        result = compute_com_sway_proxy(kp, hip_left_idx=0, hip_right_idx=3)
        assert result.n_valid_frames == 0
