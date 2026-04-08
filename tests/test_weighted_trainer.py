"""Smoke test for weighted pose trainer imports and defaults."""

import pytest


class TestWeightedTrainerImport:
    def test_import(self):
        from gait_analysis.weighted_pose_trainer import (
            WeightedPoseTrainer,
            WeightedPoseModel,
            WeightedKeypointPoseLoss,
            DEFAULT_KPT_WEIGHTS_26,
        )
        assert len(DEFAULT_KPT_WEIGHTS_26) == 26

    def test_default_weights_structure(self):
        from gait_analysis.weighted_pose_trainer import DEFAULT_KPT_WEIGHTS_26
        # Ankles at indices 15, 16 should be 2.0
        assert DEFAULT_KPT_WEIGHTS_26[15] == 2.0
        assert DEFAULT_KPT_WEIGHTS_26[16] == 2.0
        # Foot points at 20-25 should be 3.0
        for i in range(20, 26):
            assert DEFAULT_KPT_WEIGHTS_26[i] == 3.0
        # Upper body should be 1.0
        for i in range(15):
            assert DEFAULT_KPT_WEIGHTS_26[i] == 1.0
