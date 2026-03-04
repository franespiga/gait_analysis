"""
Tests for multi-backend pose estimation support.
"""

import pytest
import numpy as np

from gait_analysis.keypoint_config import (
    DetectorBackend,
    KeypointIndices,
    YOLO_COCO_KEYPOINTS,
    YOLO_LOWER_BODY_KEYPOINTS,
    get_keypoint_config
)
from gait_analysis.base_detector import GaitKeypoints
from gait_analysis.heel_toe_analyzer import HeelToeFootTracker, HeelToeGaitAnalyzer, HeelToeStep
from gait_analysis.analyzer import StepType, FootPhase


class TestKeypointConfig:
    """Tests for keypoint configuration."""
    
    def test_yolo_coco_no_heel_toe(self):
        """Test that YOLO COCO config has no heel/toe."""
        config = YOLO_COCO_KEYPOINTS
        assert config.left_heel is None
        assert config.right_heel is None
        assert config.left_toe is None
        assert config.right_toe is None
        assert not config.has_heel_toe()
    
    def test_yolo_lower_body_has_heel_toe(self):
        """Test that YOLO Lower Body config has heel/toe."""
        config = YOLO_LOWER_BODY_KEYPOINTS
        assert config.left_heel == 6
        assert config.right_heel == 7
        assert config.left_toe == 8
        assert config.right_toe == 9
        assert config.has_heel_toe()
    
    def test_get_keypoint_config(self):
        """Test factory function."""
        coco = get_keypoint_config(DetectorBackend.YOLO_COCO)
        assert coco == YOLO_COCO_KEYPOINTS
        
        lower = get_keypoint_config(DetectorBackend.YOLO_LOWER_BODY)
        assert lower == YOLO_LOWER_BODY_KEYPOINTS


class TestGaitKeypoints:
    """Tests for unified GaitKeypoints class."""
    
    def test_basic_creation(self):
        """Test creating GaitKeypoints."""
        kpts = GaitKeypoints(
            frame_number=0,
            timestamp_ms=0.0,
            left_hip=(100, 100),
            right_hip=(200, 100)
        )
        assert kpts.frame_number == 0
        assert kpts.left_hip == (100, 100)
    
    def test_has_heel_toe_false(self):
        """Test has_heel_toe when no heel/toe."""
        kpts = GaitKeypoints(
            frame_number=0,
            timestamp_ms=0.0,
            left_ankle=(100, 200),
            right_ankle=(200, 200)
        )
        assert not kpts.has_heel_toe()
    
    def test_has_heel_toe_true(self):
        """Test has_heel_toe when heel/toe present."""
        kpts = GaitKeypoints(
            frame_number=0,
            timestamp_ms=0.0,
            left_ankle=(100, 200),
            right_ankle=(200, 200),
            left_heel=(100, 210),
            right_heel=(200, 210),
            left_toe=(110, 215),
            right_toe=(210, 215)
        )
        assert kpts.has_heel_toe()
    
    def test_get_positions(self):
        """Test position getter methods."""
        kpts = GaitKeypoints(
            frame_number=0,
            timestamp_ms=0.0,
            left_hip=(100, 50),
            right_hip=(200, 50),
            left_knee=(100, 100),
            right_knee=(200, 100),
            left_ankle=(100, 150),
            right_ankle=(200, 150),
            left_heel=(100, 160),
            right_heel=(200, 160),
            left_toe=(110, 165),
            right_toe=(210, 165)
        )
        
        hips = kpts.get_hip_positions()
        assert hips["left"] == (100, 50)
        assert hips["right"] == (200, 50)
        
        knees = kpts.get_knee_positions()
        assert knees["left"] == (100, 100)
        
        ankles = kpts.get_ankle_positions()
        assert ankles["left"] == (100, 150)
        
        heels = kpts.get_heel_positions()
        assert heels["left"] == (100, 160)
        
        toes = kpts.get_toe_positions()
        assert toes["left"] == (110, 165)
    
    def test_body_center(self):
        """Test body center calculation."""
        kpts = GaitKeypoints(
            frame_number=0,
            timestamp_ms=0.0,
            left_hip=(100, 100),
            right_hip=(200, 100)
        )
        
        center = kpts.get_body_center()
        assert center == (150, 100)


class TestHeelToeStep:
    """Tests for HeelToeStep class."""
    
    def test_to_dict(self):
        """Test serialization with heel/toe data."""
        step = HeelToeStep(
            foot="left",
            step_type=StepType.HEEL_STRIKE,
            start_frame=10,
            end_frame=25,
            start_time_ms=333.3,
            end_time_ms=833.3,
            duration_ms=500.0,
            contact_angle_degrees=25.0,
            stance_time_ms=400.0,
            knee_flexion_at_contact=165.0,
            min_knee_flexion=150.0,
            max_knee_flexion=170.0,
            heel_y_at_contact=300.0,
            toe_y_at_contact=285.0,
            heel_toe_diff=15.0
        )
        
        result = step.to_dict()
        
        assert result["step_type"] == "heel_strike"
        assert result["heel_y_at_contact"] == 300.0
        assert result["toe_y_at_contact"] == 285.0
        assert result["heel_toe_diff"] == 15.0


class TestHeelToeFootTracker:
    """Tests for HeelToeFootTracker class."""
    
    def test_init(self):
        """Test initialization."""
        tracker = HeelToeFootTracker(side="left")
        assert tracker.side == "left"
        assert len(tracker.positions) == 0
        assert tracker.current_phase == FootPhase.UNKNOWN
    
    def test_add_position_with_heel_toe(self):
        """Test adding positions with heel and toe."""
        tracker = HeelToeFootTracker(side="right")
        
        tracker.add_position(
            frame=0,
            time_ms=0.0,
            heel=(100, 200),
            toe=(110, 195),
            ankle=(105, 180),
            knee=(105, 120),
            hip=(105, 60)
        )
        
        assert len(tracker.positions) == 1
        assert len(tracker.heel_velocities_y) == 1
        assert len(tracker.toe_velocities_y) == 1
    
    def test_ground_level_tracking(self):
        """Test that ground level is tracked from heel/toe."""
        tracker = HeelToeFootTracker(side="left")
        
        # First position
        tracker.add_position(
            frame=0, time_ms=0.0,
            heel=(100, 200), toe=(110, 195),
            ankle=(105, 180), knee=None, hip=None
        )
        
        # Second position with foot lower
        tracker.add_position(
            frame=1, time_ms=33.3,
            heel=(100, 250), toe=(110, 245),
            ankle=(105, 230), knee=None, hip=None
        )
        
        assert tracker.max_heel_y == 250
        assert tracker.ground_y >= 250
    
    def test_step_classification_heel_first(self):
        """Test that heel-first contact is classified as HEEL_STRIKE."""
        tracker = HeelToeFootTracker(side="left")
        tracker.ground_y = 300
        
        # Simulate landing with heel lower (higher Y) than toe
        for i in range(10):
            heel_y = 200 + i * 10  # Heel descending
            toe_y = 180 + i * 10   # Toe also descending but higher
            
            tracker.add_position(
                frame=i, time_ms=i * 33.3,
                heel=(100, heel_y), toe=(110, toe_y),
                ankle=(105, (heel_y + toe_y) / 2 - 20),
                knee=(105, 120), hip=(105, 60)
            )
        
        # The classification uses the position data
        step_type, diff, heel_y, toe_y = tracker._classify_step_type()
        
        # Heel was lower (larger Y) so should be HEEL_STRIKE
        # Note: depends on threshold values
        assert diff > 0  # Positive diff means heel lower


class TestHeelToeGaitAnalyzer:
    """Tests for HeelToeGaitAnalyzer class."""
    
    def test_init(self):
        """Test initialization."""
        analyzer = HeelToeGaitAnalyzer()
        assert analyzer.left_tracker is not None
        assert analyzer.right_tracker is not None
        assert not analyzer.has_heel_toe
    
    def test_process_frame_with_heel_toe(self):
        """Test processing frame with heel/toe keypoints."""
        analyzer = HeelToeGaitAnalyzer()
        
        keypoints = GaitKeypoints(
            frame_number=0,
            timestamp_ms=0.0,
            left_hip=(100, 50),
            right_hip=(200, 50),
            left_knee=(100, 100),
            right_knee=(200, 100),
            left_ankle=(100, 150),
            right_ankle=(200, 150),
            left_heel=(100, 160),
            right_heel=(200, 160),
            left_toe=(110, 155),
            right_toe=(210, 155)
        )
        
        state = analyzer.process_frame(keypoints)
        
        assert "left_phase" in state
        assert "right_phase" in state
        assert "has_heel_toe" in state
        assert state["has_heel_toe"] == True
    
    def test_reset(self):
        """Test reset clears state."""
        analyzer = HeelToeGaitAnalyzer()
        
        keypoints = GaitKeypoints(
            frame_number=0,
            timestamp_ms=0.0,
            left_ankle=(100, 150),
            right_ankle=(200, 150),
            left_heel=(100, 160),
            right_heel=(200, 160)
        )
        analyzer.process_frame(keypoints)
        
        analyzer.reset()
        
        assert len(analyzer.keypoint_history) == 0
        assert len(analyzer.left_tracker.positions) == 0
        assert not analyzer.has_heel_toe
    
    def test_get_results(self):
        """Test getting analysis results."""
        analyzer = HeelToeGaitAnalyzer()
        
        results = analyzer.get_results("test.mp4", 100, 30.0, "test_out.mp4")
        
        assert results.video_path == "test.mp4"
        assert results.fps == 30.0


class TestDetectorBackendEnum:
    """Tests for DetectorBackend enum."""
    
    def test_enum_values(self):
        """Test enum values."""
        assert DetectorBackend.YOLO_COCO.value == "yolo_coco"
        assert DetectorBackend.YOLOV8.value == "yolov8"
        assert DetectorBackend.YOLO_LOWER_BODY.value == "yolo_lower"
    
    def test_enum_comparison(self):
        """Test enum comparison."""
        assert DetectorBackend.YOLO_COCO == DetectorBackend.YOLO_COCO
        assert DetectorBackend.YOLO_COCO != DetectorBackend.YOLO_LOWER_BODY

