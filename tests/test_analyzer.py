"""
Tests for the gait analyzer module.
"""

import pytest
from gait_analysis.analyzer import (
    GaitAnalyzer,
    FootTracker,
    StepType,
    FootPhase,
    Step,
    GaitAnalysisResult
)
from gait_analysis.detector import FrameKeypoints


class TestFootTracker:
    """Tests for FootTracker class."""
    
    def test_init(self):
        """Test FootTracker initialization."""
        tracker = FootTracker(side="left")
        assert tracker.side == "left"
        assert len(tracker.positions) == 0
        assert len(tracker.steps) == 0
        assert tracker.current_phase == FootPhase.UNKNOWN
    
    def test_add_position(self):
        """Test adding positions to tracker."""
        tracker = FootTracker(side="right")
        tracker.add_position(0, 0.0, 100.0, 200.0)
        tracker.add_position(1, 33.3, 105.0, 210.0)
        
        assert len(tracker.positions) == 2
        assert len(tracker.velocities) == 2


class TestStep:
    """Tests for Step class."""
    
    def test_to_dict(self):
        """Test Step serialization."""
        step = Step(
            foot="left",
            step_type=StepType.HEEL_STRIKE,
            start_frame=10,
            end_frame=25,
            start_time_ms=333.3,
            end_time_ms=833.3,
            duration_ms=500.0
        )
        
        result = step.to_dict()
        
        assert result["foot"] == "left"
        assert result["step_type"] == "heel_strike"
        assert result["duration_ms"] == 500.0


class TestGaitAnalysisResult:
    """Tests for GaitAnalysisResult class."""
    
    def test_compute_metrics_empty(self):
        """Test metrics computation with no steps."""
        result = GaitAnalysisResult(
            video_path="test.mp4",
            total_frames=100,
            fps=30.0,
            duration_seconds=3.33
        )
        
        result.compute_metrics()
        
        assert result.total_steps == 0
        assert result.correct_percentage == 0
    
    def test_compute_metrics_with_steps(self):
        """Test metrics computation with steps."""
        result = GaitAnalysisResult(
            video_path="test.mp4",
            total_frames=300,
            fps=30.0,
            duration_seconds=10.0,
            left_steps=[
                Step("left", StepType.HEEL_STRIKE, 10, 25, 333, 833, 500),
                Step("left", StepType.TOE_STRIKE, 50, 65, 1666, 2166, 500),
            ],
            right_steps=[
                Step("right", StepType.HEEL_STRIKE, 30, 45, 1000, 1500, 500),
            ]
        )
        
        result.compute_metrics()
        
        assert result.total_steps == 3
        assert result.correct_steps == 2
        assert result.incorrect_steps == 1
        assert result.correct_percentage == pytest.approx(66.7, rel=0.1)
    
    def test_to_dict(self):
        """Test result serialization."""
        result = GaitAnalysisResult(
            video_path="test.mp4",
            total_frames=100,
            fps=30.0,
            duration_seconds=3.33
        )
        result.compute_metrics()
        
        data = result.to_dict()
        
        assert "video_info" in data
        assert "summary" in data
        assert "timing" in data
        assert "per_foot" in data
        assert "detailed_steps" in data


class TestGaitAnalyzer:
    """Tests for GaitAnalyzer class."""
    
    def test_init(self):
        """Test GaitAnalyzer initialization."""
        analyzer = GaitAnalyzer()
        assert analyzer.left_tracker is not None
        assert analyzer.right_tracker is not None
    
    def test_reset(self):
        """Test analyzer reset."""
        analyzer = GaitAnalyzer()
        
        # Add some data
        keypoints = FrameKeypoints(
            frame_number=0,
            timestamp_ms=0.0,
            keypoints={
                "left_ankle": (100.0, 200.0, 0.9),
                "right_ankle": (150.0, 200.0, 0.9),
                "left_knee": (100.0, 150.0, 0.9),
                "right_knee": (150.0, 150.0, 0.9),
            }
        )
        analyzer.process_frame(keypoints)
        
        # Reset
        analyzer.reset()
        
        assert len(analyzer.keypoint_history) == 0
        assert len(analyzer.left_tracker.positions) == 0
    
    def test_process_frame(self):
        """Test frame processing."""
        analyzer = GaitAnalyzer()
        
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
    
    def test_get_results(self):
        """Test getting analysis results."""
        analyzer = GaitAnalyzer()
        
        results = analyzer.get_results("test.mp4", 100, 30.0)
        
        assert isinstance(results, GaitAnalysisResult)
        assert results.video_path == "test.mp4"
        assert results.fps == 30.0

