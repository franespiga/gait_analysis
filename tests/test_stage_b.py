"""Unit tests for Stage B foot refinement module."""

import numpy as np
import pytest

from gait_analysis.stage_b_refinement import (
    StageBConfig,
    needs_stage_b,
    compute_foot_crop,
    FOOT_KPT_INDICES,
)


class TestNeedsStageB:
    def test_disabled(self):
        cfg = StageBConfig(enabled=False)
        kpts = np.ones((26, 3))
        assert needs_stage_b(kpts, None, cfg) is False

    def test_low_foot_conf_triggers(self):
        cfg = StageBConfig(enabled=True, trigger_min_foot_conf=0.5)
        kpts = np.ones((26, 3))
        for idx in FOOT_KPT_INDICES:
            kpts[idx, 2] = 0.1
        assert needs_stage_b(kpts, None, cfg) is True

    def test_high_foot_conf_no_trigger(self):
        cfg = StageBConfig(enabled=True, trigger_min_foot_conf=0.3)
        kpts = np.ones((26, 3)) * 0.9
        kpts[:, :2] = 100
        # Make left/right feet far apart so overlap doesn't fire
        for idx in [15, 20, 22, 24]:
            kpts[idx, 0] = 50
        for idx in [16, 21, 23, 25]:
            kpts[idx, 0] = 300
        assert needs_stage_b(kpts, None, cfg) is False

    def test_overlap_triggers(self):
        cfg = StageBConfig(enabled=True, trigger_overlap_flag=True,
                           trigger_min_foot_conf=0.01)
        kpts = np.ones((26, 3)) * 0.9
        kpts[:, :2] = 200
        # All foot keypoints at same position → overlap
        assert needs_stage_b(kpts, None, cfg) is True

    def test_jitter_triggers(self):
        cfg = StageBConfig(enabled=True, trigger_max_jitter_px=5.0,
                           trigger_min_foot_conf=0.01,
                           trigger_overlap_flag=False)
        kpts = np.ones((26, 3)) * 0.9
        kpts[:, :2] = 200
        # Make left/right feet far apart so overlap doesn't fire
        for idx in [15, 20, 22, 24]:
            kpts[idx, 0] = 50
        for idx in [16, 21, 23, 25]:
            kpts[idx, 0] = 300

        prev = kpts.copy()
        for idx in FOOT_KPT_INDICES:
            kpts[idx, 0] += 50  # big shift
        assert needs_stage_b(kpts, prev, cfg) is True


class TestComputeFootCrop:
    def test_returns_box(self):
        kpts = np.zeros((26, 3))
        kpts[15] = [200, 400, 1.0]
        kpts[16] = [250, 410, 1.0]
        kpts[24] = [190, 430, 1.0]
        kpts[25] = [260, 435, 1.0]
        box = compute_foot_crop(kpts, 640, 480)
        assert box is not None
        x1, y1, x2, y2 = box
        assert x1 >= 0 and y1 >= 0
        assert x2 <= 640 and y2 <= 480
        assert x2 > x1 and y2 > y1

    def test_no_ankle_returns_none(self):
        kpts = np.zeros((26, 3))
        box = compute_foot_crop(kpts, 640, 480)
        assert box is None
