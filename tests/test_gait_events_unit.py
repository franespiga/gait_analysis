"""Unit tests for gait_events.py — IC/TO detection and contact classification."""

import numpy as np
import pytest

from gait_analysis.gait_events import (
    GaitEvent,
    ContactType,
    detect_ic_to,
    classify_contact,
    events_to_contact_types,
    _velocity_y,
    _stable_mask,
)


def _make_keypoints(heel_y_left, toe_y_left, heel_y_right=None, toe_y_right=None, conf=0.9):
    """Build (T, 6, 3) keypoints from 1-D heel/toe Y trajectories.

    X values are constant (heel at x=50, toe at x=70 per foot).
    """
    T = len(heel_y_left)
    if heel_y_right is None:
        heel_y_right = np.array(heel_y_left) + 5
    if toe_y_right is None:
        toe_y_right = np.array(toe_y_left) + 5
    kp = np.zeros((T, 6, 3))
    # L_HEEL=0, L_BIG_TOE=1, L_SMALL_TOE=2, R_HEEL=3, R_BIG_TOE=4, R_SMALL_TOE=5
    for t in range(T):
        kp[t, 0] = [50, heel_y_left[t], conf]    # L_HEEL
        kp[t, 1] = [70, toe_y_left[t], conf]     # L_BIG_TOE
        kp[t, 2] = [65, toe_y_left[t], conf]     # L_SMALL_TOE
        kp[t, 3] = [150, heel_y_right[t], conf]  # R_HEEL
        kp[t, 4] = [170, toe_y_right[t], conf]   # R_BIG_TOE
        kp[t, 5] = [165, toe_y_right[t], conf]   # R_SMALL_TOE
    return kp


class TestVelocity:
    def test_basic(self):
        series = np.array([0.0, 1.0, 3.0, 6.0])
        v = _velocity_y(series)
        assert v[0] == 0.0
        np.testing.assert_allclose(v[1:], [1.0, 2.0, 3.0])

    def test_constant(self):
        series = np.ones(10) * 5
        v = _velocity_y(series)
        np.testing.assert_allclose(v, 0.0)


class TestStableMask:
    def test_all_stable(self):
        vel = np.zeros(10)
        mask = _stable_mask(vel, threshold=1.0, min_frames=3)
        assert np.all(mask)

    def test_none_stable(self):
        vel = np.ones(10) * 10
        mask = _stable_mask(vel, threshold=1.0, min_frames=3)
        assert not np.any(mask)


class TestDetectICTO:
    def test_simple_gait_cycle(self):
        """One left foot swing-stance-swing cycle should produce IC and TO."""
        T = 60
        fps = 30.0
        heel_y = np.zeros(T)
        toe_y = np.zeros(T)
        # Simulate: frames 0-19 stance (stable), 20-39 swing (fast), 40-59 stance
        heel_y[:20] = 300
        heel_y[20:40] = np.linspace(300, 200, 20)
        heel_y[40:] = 300
        toe_y = heel_y - 5  # toe slightly higher

        kp = _make_keypoints(heel_y, toe_y)
        times_ms = np.arange(T) / fps * 1000

        events = detect_ic_to(
            kp, times_ms,
            velocity_threshold=2.0,
            stability_frames=3,
            min_step_duration_ms=100.0,
        )
        ic_events = [e for e in events if e.event_type == "IC"]
        to_events = [e for e in events if e.event_type == "TO"]
        assert len(ic_events) > 0, "Should detect at least one IC"
        assert len(to_events) > 0, "Should detect at least one TO"

    def test_empty_keypoints(self):
        kp = np.zeros((3, 6, 3))
        times = np.array([0, 33, 66], dtype=float)
        events = detect_ic_to(kp, times, stability_frames=5)
        assert events == []

    def test_min_step_duration_filter(self):
        """ICs closer than min_step_duration_ms should be filtered."""
        T = 100
        heel_y = np.ones(T) * 300
        # Two brief stance periods 50ms apart — should be filtered
        heel_y[10:15] = 300
        heel_y[15:20] = 200
        heel_y[20:25] = 300
        toe_y = heel_y - 5
        kp = _make_keypoints(heel_y, toe_y)
        times_ms = np.arange(T, dtype=float) * 10  # 100 fps → 10 ms/frame

        events = detect_ic_to(kp, times_ms, min_step_duration_ms=200.0)
        left_ics = [e for e in events if e.event_type == "IC" and e.foot == "left"]
        for i in range(1, len(left_ics)):
            assert left_ics[i].time_ms - left_ics[i - 1].time_ms >= 200.0


class TestClassifyContact:
    def _make_ic_event(self, frame, time_ms, foot="left"):
        return GaitEvent("IC", foot, frame, time_ms)

    def test_heel_strike(self):
        """Heel lower than toe → heel strike."""
        T = 30
        kp = np.zeros((T, 6, 3))
        for t in range(T):
            kp[t, 0] = [50, 310, 0.9]   # L_HEEL (lower = higher Y)
            kp[t, 1] = [70, 290, 0.9]   # L_BIG_TOE (higher = lower Y)
            kp[t, 2] = [65, 290, 0.9]
            kp[t, 3:] = kp[t, :3] + [100, 0, 0]
        times = np.arange(T, dtype=float) * 33.3
        ic = self._make_ic_event(15, 15 * 33.3)
        ct = classify_contact(kp, times, ic)
        assert ct == ContactType.HEEL_STRIKE

    def test_toe_walk(self):
        """Toe lower than heel → toe walk."""
        T = 30
        kp = np.zeros((T, 6, 3))
        for t in range(T):
            kp[t, 0] = [50, 290, 0.9]   # L_HEEL (higher)
            kp[t, 1] = [70, 310, 0.9]   # L_BIG_TOE (lower)
            kp[t, 2] = [65, 310, 0.9]
            kp[t, 3:] = kp[t, :3] + [100, 0, 0]
        times = np.arange(T, dtype=float) * 33.3
        ic = self._make_ic_event(15, 15 * 33.3)
        ct = classify_contact(kp, times, ic)
        assert ct == ContactType.TOE_WALK

    def test_flat_with_window(self):
        """Small pitch + simultaneous stabilisation → flat."""
        T = 30
        kp = np.zeros((T, 6, 3))
        for t in range(T):
            kp[t, 0] = [50, 300, 0.9]
            kp[t, 1] = [70, 301, 0.9]  # almost level
            kp[t, 2] = [65, 301, 0.9]
            kp[t, 3:] = kp[t, :3] + [100, 0, 0]
        times = np.arange(T, dtype=float) * 33.3
        ic = self._make_ic_event(15, 15 * 33.3)
        ct = classify_contact(kp, times, ic, pitch_threshold_deg=10.0, flat_window_ms=200.0)
        assert ct == ContactType.FLAT

    def test_low_confidence_returns_unknown(self):
        T = 20
        kp = np.zeros((T, 6, 3))
        kp[:, :, 2] = 0.1  # low conf
        times = np.arange(T, dtype=float) * 33.3
        ic = self._make_ic_event(10, 10 * 33.3)
        ct = classify_contact(kp, times, ic, conf_threshold=0.3)
        assert ct == ContactType.UNKNOWN


class TestEventsToContactTypes:
    def test_assigns_types(self):
        T = 40
        kp = np.zeros((T, 6, 3))
        for t in range(T):
            kp[t, 0] = [50, 310, 0.9]
            kp[t, 1] = [70, 290, 0.9]
            kp[t, 2] = [65, 290, 0.9]
            kp[t, 3:] = kp[t, :3] + [100, 0, 0]
        times = np.arange(T, dtype=float) * 33.3
        events = [
            GaitEvent("IC", "left", 10, 333.0),
            GaitEvent("TO", "left", 20, 666.0),
        ]
        result = events_to_contact_types(kp, times, events)
        ic = [e for e in result if e.event_type == "IC"][0]
        assert ic.contact_type is not None
        assert ic.contact_type != ContactType.UNKNOWN
