"""
Tests for the advanced angle analyzer module.
"""

import pytest
import numpy as np
from gait_analysis.angle_analyzer import (
    EnhancedGaitAnalyzer,
    EnhancedFootTracker,
    EnhancedStep,
    EnhancedGaitAnalysisResult,
    FootIdentifier,
    calculate_statistics,
    calculate_knee_angle
)
from gait_analysis.analyzer import StepType, FootPhase
from gait_analysis.detector import FrameKeypoints


class TestCalculateStatistics:
    """Tests for calculate_statistics function."""
    
    def test_empty_list(self):
        """Test with empty list."""
        result = calculate_statistics([])
        assert result["count"] == 0
        assert result["mean"] == 0.0
        assert result["min"] == 0.0
        assert result["max"] == 0.0
        assert result["median"] == 0.0
        assert result["std"] == 0.0
    
    def test_single_value(self):
        """Test with single value."""
        result = calculate_statistics([5.0])
        assert result["count"] == 1
        assert result["mean"] == 5.0
        assert result["min"] == 5.0
        assert result["max"] == 5.0
        assert result["median"] == 5.0
        assert result["std"] == 0.0
    
    def test_multiple_values(self):
        """Test with multiple values."""
        result = calculate_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["count"] == 5
        assert result["mean"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["median"] == 3.0
        assert result["std"] == pytest.approx(1.41, rel=0.1)
    
    def test_std_calculation(self):
        """Test standard deviation calculation."""
        # Values with known std
        values = [10.0, 10.0, 10.0, 10.0]  # All same = std 0
        result = calculate_statistics(values)
        assert result["std"] == 0.0
        
        # Different values
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = calculate_statistics(values)
        assert result["std"] > 0


class TestCalculateKneeAngle:
    """Tests for knee angle calculation."""
    
    def test_straight_leg(self):
        """Test with a straight leg (180 degrees)."""
        hip = (100, 0)
        knee = (100, 100)
        ankle = (100, 200)
        
        angle = calculate_knee_angle(hip, knee, ankle)
        assert angle == pytest.approx(180.0, rel=0.01)
    
    def test_bent_knee(self):
        """Test with a bent knee."""
        hip = (100, 0)
        knee = (100, 100)
        ankle = (150, 200)  # Ankle shifted right
        
        angle = calculate_knee_angle(hip, knee, ankle)
        assert angle < 180.0
        assert angle > 90.0
    
    def test_right_angle(self):
        """Test with 90 degree bend."""
        hip = (100, 0)
        knee = (100, 100)
        ankle = (200, 100)  # Ankle at same height as knee, to the right
        
        angle = calculate_knee_angle(hip, knee, ankle)
        assert angle == pytest.approx(90.0, rel=0.01)
    
    def test_missing_position(self):
        """Test with missing position returns None."""
        assert calculate_knee_angle(None, (100, 100), (100, 200)) is None
        assert calculate_knee_angle((100, 0), None, (100, 200)) is None
        assert calculate_knee_angle((100, 0), (100, 100), None) is None


class TestFootIdentifier:
    """Tests for FootIdentifier class."""
    
    def test_init(self):
        """Test initialization."""
        identifier = FootIdentifier()
        assert identifier.left_avg_x is None
        assert identifier.right_avg_x is None
        assert identifier.hip_mismatch_count == 0
    
    def test_reset(self):
        """Test reset clears state."""
        identifier = FootIdentifier()
        identifier.left_x_history = [100, 110, 120]
        identifier.hip_mismatch_count = 5
        
        identifier.reset()
        
        assert len(identifier.left_x_history) == 0
        assert identifier.hip_mismatch_count == 0
    
    def test_validate_correct_positions(self):
        """Test validation with correct positions."""
        identifier = FootIdentifier()
        
        keypoints = FrameKeypoints(
            frame_number=0,
            timestamp_ms=0.0,
            keypoints={
                "left_ankle": (80.0, 200.0, 0.9),   # Left of center
                "right_ankle": (120.0, 200.0, 0.9), # Right of center
                "left_knee": (80.0, 150.0, 0.9),
                "right_knee": (120.0, 150.0, 0.9),
                "left_hip": (90.0, 100.0, 0.9),
                "right_hip": (110.0, 100.0, 0.9),   # Center at x=100
            }
        )
        
        ankles, knees, hips, confidence = identifier.validate_and_correct(keypoints)
        
        # Should not swap - positions are correct
        assert ankles["left"][0] == pytest.approx(80.0)
        assert ankles["right"][0] == pytest.approx(120.0)
        assert confidence >= 0.7
    
    def test_history_tracking(self):
        """Test that position history is tracked."""
        identifier = FootIdentifier()
        
        for i in range(5):
            keypoints = FrameKeypoints(
                frame_number=i,
                timestamp_ms=i * 33.3,
                keypoints={
                    "left_ankle": (80.0 + i, 200.0, 0.9),
                    "right_ankle": (120.0 + i, 200.0, 0.9),
                    "left_knee": (80.0, 150.0, 0.9),
                    "right_knee": (120.0, 150.0, 0.9),
                    "left_hip": (90.0, 100.0, 0.9),
                    "right_hip": (110.0, 100.0, 0.9),
                }
            )
            identifier.validate_and_correct(keypoints)
        
        assert len(identifier.left_x_history) == 5
        assert len(identifier.right_x_history) == 5
        assert identifier.left_avg_x is not None
        assert identifier.right_avg_x is not None


class TestEnhancedStep:
    """Tests for EnhancedStep class."""
    
    def test_to_dict(self):
        """Test EnhancedStep serialization."""
        step = EnhancedStep(
            foot="left",
            step_type=StepType.HEEL_STRIKE,
            start_frame=10,
            end_frame=25,
            start_time_ms=333.3,
            end_time_ms=833.3,
            duration_ms=500.0,
            contact_angle_degrees=25.5,
            stance_time_ms=400.0,
            knee_flexion_at_contact=165.0,
            min_knee_flexion=145.0,
            max_knee_flexion=175.0
        )
        
        result = step.to_dict()
        
        assert result["foot"] == "left"
        assert result["step_type"] == "heel_strike"
        assert result["duration_ms"] == 500.0
        assert result["contact_angle_degrees"] == 25.5
        assert result["stance_time_ms"] == 400.0
        assert result["knee_flexion_at_contact"] == 165.0
        assert result["min_knee_flexion"] == 145.0
        assert result["max_knee_flexion"] == 175.0
    
    def test_default_knee_values(self):
        """Test default knee flexion values."""
        step = EnhancedStep(
            foot="right",
            step_type=StepType.TOE_STRIKE,
            start_frame=10,
            end_frame=25,
            start_time_ms=333.3,
            end_time_ms=833.3,
            duration_ms=500.0,
            contact_angle_degrees=60.0,
            stance_time_ms=350.0
        )
        
        assert step.knee_flexion_at_contact == 180.0
        assert step.min_knee_flexion == 180.0
        assert step.max_knee_flexion == 180.0


class TestEnhancedFootTracker:
    """Tests for EnhancedFootTracker class."""
    
    def test_init(self):
        """Test initialization."""
        tracker = EnhancedFootTracker(side="left")
        assert tracker.side == "left"
        assert len(tracker.positions) == 0
        assert len(tracker.knee_angles) == 0
        assert tracker.current_phase == FootPhase.UNKNOWN
    
    def test_add_position_with_hip(self):
        """Test adding positions with hip coordinates."""
        tracker = EnhancedFootTracker(side="right")
        tracker.add_position(0, 0.0, 100.0, 200.0, 100.0, 150.0, 100.0, 100.0)
        tracker.add_position(1, 33.3, 105.0, 205.0, 105.0, 155.0, 105.0, 105.0)
        
        assert len(tracker.positions) == 2
        assert len(tracker.velocities_x) == 2
        assert len(tracker.velocities_y) == 2
        assert len(tracker.knee_angles) == 2
    
    def test_knee_angle_tracking(self):
        """Test that knee angles are tracked."""
        tracker = EnhancedFootTracker(side="left")
        
        # Add position with hip, knee, ankle for angle calculation
        # Hip at (100, 0), knee at (100, 100), ankle at (100, 200) = straight leg
        tracker.add_position(0, 0.0, 100.0, 200.0, 100.0, 100.0, 100.0, 0.0)
        
        assert len(tracker.knee_angles) == 1
        assert tracker.knee_angles[0] == pytest.approx(180.0, rel=0.01)
    
    def test_ground_y_tracking(self):
        """Test that ground level is tracked."""
        tracker = EnhancedFootTracker(side="left")
        tracker.add_position(0, 0.0, 100.0, 200.0)
        tracker.add_position(1, 33.3, 100.0, 250.0)  # Higher Y = lower in image
        
        assert tracker.max_ankle_y == 250.0
        assert tracker.ground_y == 250.0


class TestEnhancedGaitAnalysisResult:
    """Tests for EnhancedGaitAnalysisResult class."""
    
    def test_compute_metrics_empty(self):
        """Test metrics with no steps."""
        result = EnhancedGaitAnalysisResult(
            video_path="test.mp4",
            total_frames=100,
            fps=30.0,
            duration_seconds=3.33
        )
        
        metrics = result.compute_metrics()
        
        assert result.total_steps == 0
        assert result.flat_foot_steps == 0
        assert metrics["overall"]["all_steps"]["angles"]["count"] == 0
    
    def test_compute_metrics_with_steps(self):
        """Test metrics computation with steps including flat_foot."""
        result = EnhancedGaitAnalysisResult(
            video_path="test.mp4",
            total_frames=300,
            fps=30.0,
            duration_seconds=10.0,
            left_steps=[
                EnhancedStep("left", StepType.HEEL_STRIKE, 10, 25, 333, 833, 500, 25.0, 400, 165, 150, 170),
                EnhancedStep("left", StepType.TOE_STRIKE, 50, 65, 1666, 2166, 500, 60.0, 350, 170, 160, 175),
                EnhancedStep("left", StepType.FLAT_FOOT, 80, 95, 2666, 3166, 500, 45.0, 380, 168, 155, 172),
            ],
            right_steps=[
                EnhancedStep("right", StepType.HEEL_STRIKE, 30, 45, 1000, 1500, 500, 28.0, 420, 162, 148, 168),
            ]
        )
        
        metrics = result.compute_metrics()
        
        assert result.total_steps == 4
        assert result.correct_steps == 2
        assert result.incorrect_steps == 1
        assert result.flat_foot_steps == 1
        
        # Check percentages
        assert result.correct_percentage == 50.0
        assert result.incorrect_percentage == 25.0
        assert result.flat_foot_percentage == 25.0
        
        # Check angle statistics
        assert metrics["overall"]["correct_steps"]["angles"]["count"] == 2
        assert metrics["overall"]["incorrect_steps"]["angles"]["count"] == 1
        assert metrics["overall"]["flat_foot_steps"]["angles"]["count"] == 1
        
        # Check knee flexion is included
        assert "knee_flexion" in metrics["overall"]["correct_steps"]
        assert "at_contact" in metrics["overall"]["correct_steps"]["knee_flexion"]
        assert "min_during_stance" in metrics["overall"]["correct_steps"]["knee_flexion"]
    
    def test_to_dict_includes_flat_foot(self):
        """Test that to_dict includes flat_foot in summary."""
        result = EnhancedGaitAnalysisResult(
            video_path="test.mp4",
            total_frames=100,
            fps=30.0,
            duration_seconds=3.33,
            output_video_path="test_analyzed.mp4"
        )
        
        data = result.to_dict()
        
        assert "video_info" in data
        assert "summary" in data
        assert "metrics" in data
        assert "detailed_steps" in data
        
        # Check flat_foot in summary
        assert "flat_foot_steps" in data["summary"]
        assert "flat_foot_percentage" in data["summary"]
        
        # Check flat_foot in metrics
        assert "flat_foot_steps" in data["metrics"]["overall"]


class TestEnhancedGaitAnalyzer:
    """Tests for EnhancedGaitAnalyzer class."""
    
    def test_init(self):
        """Test initialization."""
        analyzer = EnhancedGaitAnalyzer()
        assert analyzer.left_tracker is not None
        assert analyzer.right_tracker is not None
        assert analyzer.foot_identifier is not None
    
    def test_reset(self):
        """Test analyzer reset."""
        analyzer = EnhancedGaitAnalyzer()
        
        # Add some data
        keypoints = FrameKeypoints(
            frame_number=0,
            timestamp_ms=0.0,
            keypoints={
                "left_ankle": (100.0, 200.0, 0.9),
                "right_ankle": (150.0, 200.0, 0.9),
                "left_knee": (100.0, 150.0, 0.9),
                "right_knee": (150.0, 150.0, 0.9),
                "left_hip": (100.0, 100.0, 0.9),
                "right_hip": (150.0, 100.0, 0.9),
            }
        )
        analyzer.process_frame(keypoints)
        
        # Reset
        analyzer.reset()
        
        assert len(analyzer.keypoint_history) == 0
        assert len(analyzer.left_tracker.positions) == 0
        assert analyzer.foot_identifier.left_avg_x is None
    
    def test_process_frame(self):
        """Test frame processing."""
        analyzer = EnhancedGaitAnalyzer()
        
        keypoints = FrameKeypoints(
            frame_number=0,
            timestamp_ms=0.0,
            keypoints={
                "left_ankle": (100.0, 200.0, 0.9),
                "right_ankle": (150.0, 200.0, 0.9),
                "left_knee": (100.0, 150.0, 0.9),
                "right_knee": (150.0, 150.0, 0.9),
                "left_hip": (100.0, 100.0, 0.9),
                "right_hip": (150.0, 100.0, 0.9),
            }
        )
        
        state = analyzer.process_frame(keypoints)
        
        assert "left_phase" in state
        assert "right_phase" in state
        assert "left_steps" in state
        assert "right_steps" in state
        assert "last_left_angle" in state
        assert "last_right_angle" in state
        assert "left_knee_angle" in state
        assert "right_knee_angle" in state
        assert "foot_id_confidence" in state
    
    def test_knee_angles_in_state(self):
        """Test that knee angles are included in state."""
        analyzer = EnhancedGaitAnalyzer()
        
        # Create keypoints that form straight legs
        keypoints = FrameKeypoints(
            frame_number=0,
            timestamp_ms=0.0,
            keypoints={
                "left_ankle": (100.0, 300.0, 0.9),
                "right_ankle": (200.0, 300.0, 0.9),
                "left_knee": (100.0, 200.0, 0.9),
                "right_knee": (200.0, 200.0, 0.9),
                "left_hip": (100.0, 100.0, 0.9),
                "right_hip": (200.0, 100.0, 0.9),
            }
        )
        
        state = analyzer.process_frame(keypoints)
        
        # Should have knee angles close to 180 (straight leg)
        assert state["left_knee_angle"] == pytest.approx(180.0, rel=0.01)
        assert state["right_knee_angle"] == pytest.approx(180.0, rel=0.01)
    
    def test_get_results(self):
        """Test getting analysis results."""
        analyzer = EnhancedGaitAnalyzer()
        
        results = analyzer.get_results("test.mp4", 100, 30.0, "test_analyzed.mp4")
        
        assert isinstance(results, EnhancedGaitAnalysisResult)
        assert results.video_path == "test.mp4"
        assert results.fps == 30.0
        assert results.output_video_path == "test_analyzed.mp4"
