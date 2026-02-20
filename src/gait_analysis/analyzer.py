"""
Gait analysis module for detecting and classifying walking patterns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

from .detector import FrameKeypoints


class StepType(Enum):
    """
    Classification of step type based on foot strike pattern.
    
    Categories:
    - HEEL_STRIKE: Correct gait - heel touches ground first, then toe
    - TOE_STRIKE: Incorrect gait (toe walking) - toe touches first
    - FLAT_FOOT: Suboptimal gait - heel and toe touch simultaneously
                 (not as problematic as toe walking, but not ideal)
    - UNKNOWN: Could not be classified
    """
    HEEL_STRIKE = "heel_strike"  # Correct gait
    TOE_STRIKE = "toe_strike"    # Incorrect gait (toe walking)
    FLAT_FOOT = "flat_foot"      # Suboptimal gait (simultaneous contact)
    UNKNOWN = "unknown"


class FootPhase(Enum):
    """Phase of the foot during gait cycle."""
    SWING = "swing"       # Foot in the air
    STANCE = "stance"     # Foot on the ground
    UNKNOWN = "unknown"


@dataclass
class Step:
    """Represents a single step."""
    foot: str  # "left" or "right"
    step_type: StepType
    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    
    def to_dict(self) -> dict:
        return {
            "foot": self.foot,
            "step_type": self.step_type.value,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "duration_ms": self.duration_ms
        }


@dataclass 
class FootTracker:
    """Tracks the state of a single foot over time."""
    side: str  # "left" or "right"
    positions: list[tuple[int, float, float, float]] = field(default_factory=list)  # (frame, time_ms, x, y)
    velocities: list[float] = field(default_factory=list)  # vertical velocities
    current_phase: FootPhase = FootPhase.UNKNOWN
    phase_start_frame: int = 0
    phase_start_time: float = 0.0
    steps: list[Step] = field(default_factory=list)
    
    # Parameters for phase detection
    stance_velocity_threshold: float = 2.0  # pixels/frame - low velocity = stance
    swing_velocity_threshold: float = 5.0   # pixels/frame - high velocity = swing
    min_stance_frames: int = 3              # Minimum frames for valid stance
    
    def add_position(self, frame: int, time_ms: float, x: float, y: float, knee_y: Optional[float] = None):
        """Add a new position observation."""
        self.positions.append((frame, time_ms, x, y))
        
        # Calculate vertical velocity if we have enough positions
        if len(self.positions) >= 2:
            prev_frame, prev_time, prev_x, prev_y = self.positions[-2]
            dt = max(1, frame - prev_frame)
            vy = (y - prev_y) / dt  # positive = moving down
            self.velocities.append(vy)
        else:
            self.velocities.append(0.0)
        
        # Detect phase transitions
        self._update_phase(frame, time_ms, y, knee_y)
    
    def _update_phase(self, frame: int, time_ms: float, ankle_y: float, knee_y: Optional[float]):
        """Update the current phase based on foot position and velocity."""
        if len(self.velocities) < 3:
            return
        
        # Use smoothed velocity for stability
        recent_velocities = self.velocities[-3:]
        avg_velocity = np.mean(recent_velocities)
        
        # Detect stance phase (foot stationary, likely on ground)
        # In image coordinates, higher Y = lower in image = closer to ground
        if self.current_phase != FootPhase.STANCE:
            # Transitioning to stance: velocity low and foot relatively low
            if abs(avg_velocity) < self.stance_velocity_threshold:
                # Check if this is a valid transition
                if self.current_phase == FootPhase.SWING:
                    self._complete_step(frame, time_ms)
                
                self.current_phase = FootPhase.STANCE
                self.phase_start_frame = frame
                self.phase_start_time = time_ms
        
        elif self.current_phase == FootPhase.STANCE:
            # Transitioning to swing: velocity increases significantly
            if abs(avg_velocity) > self.swing_velocity_threshold:
                self.current_phase = FootPhase.SWING
                self.phase_start_frame = frame
                self.phase_start_time = time_ms
    
    def _complete_step(self, frame: int, time_ms: float):
        """Complete a step when transitioning from swing to stance."""
        if self.phase_start_frame == 0:
            return
            
        # Determine step type based on ankle trajectory during landing
        step_type = self._classify_step_type()
        
        duration = time_ms - self.phase_start_time
        
        if duration > 100:  # Minimum step duration 100ms to filter noise
            step = Step(
                foot=self.side,
                step_type=step_type,
                start_frame=self.phase_start_frame,
                end_frame=frame,
                start_time_ms=self.phase_start_time,
                end_time_ms=time_ms,
                duration_ms=duration
            )
            self.steps.append(step)
    
    def _classify_step_type(self) -> StepType:
        """
        Classify the step type based on ankle movement pattern.
        
        Heel strike: ankle moves forward then down (heel first)
        Toe strike: ankle moves down quickly (toe first)
        """
        if len(self.positions) < 5:
            return StepType.UNKNOWN
        
        # Look at the last few positions during landing
        recent = self.positions[-5:]
        
        # Calculate horizontal and vertical movement
        x_positions = [p[2] for p in recent]
        y_positions = [p[3] for p in recent]
        
        # Vertical velocity pattern
        y_velocities = np.diff(y_positions)
        
        # In heel strike, the ankle typically:
        # 1. Has a more gradual descent
        # 2. May have slight backward motion initially
        
        # In toe strike, the ankle typically:
        # 1. Descends more rapidly
        # 2. Less horizontal variation
        
        avg_descent_rate = np.mean(y_velocities)
        descent_variation = np.std(y_velocities)
        
        # Calculate horizontal movement
        x_range = max(x_positions) - min(x_positions)
        
        # Heuristic classification
        # Toe walking tends to have faster, more uniform descent
        if avg_descent_rate > 3 and descent_variation < 2:
            return StepType.TOE_STRIKE
        elif x_range > 10 and avg_descent_rate < 5:
            return StepType.HEEL_STRIKE
        else:
            return StepType.FLAT_FOOT


@dataclass
class GaitAnalysisResult:
    """Complete results of gait analysis."""
    video_path: str
    total_frames: int
    fps: float
    duration_seconds: float
    
    # Step counts
    left_steps: list[Step] = field(default_factory=list)
    right_steps: list[Step] = field(default_factory=list)
    
    # Computed metrics (filled in after analysis)
    total_steps: int = 0
    correct_steps: int = 0      # heel_strike
    incorrect_steps: int = 0    # toe_strike
    flat_foot_steps: int = 0    # flat_foot
    correct_percentage: float = 0.0
    incorrect_percentage: float = 0.0
    flat_foot_percentage: float = 0.0
    
    avg_step_time_ms: float = 0.0
    avg_correct_step_time_ms: float = 0.0
    avg_incorrect_step_time_ms: float = 0.0
    avg_flat_foot_step_time_ms: float = 0.0
    
    left_correct_steps: int = 0
    left_incorrect_steps: int = 0
    left_flat_foot_steps: int = 0
    right_correct_steps: int = 0
    right_incorrect_steps: int = 0
    right_flat_foot_steps: int = 0
    
    def compute_metrics(self):
        """Compute aggregate metrics from step data."""
        all_steps = self.left_steps + self.right_steps
        self.total_steps = len(all_steps)
        
        if self.total_steps == 0:
            return
        
        # Count by step type
        correct = [s for s in all_steps if s.step_type == StepType.HEEL_STRIKE]
        incorrect = [s for s in all_steps if s.step_type == StepType.TOE_STRIKE]
        flat_foot = [s for s in all_steps if s.step_type == StepType.FLAT_FOOT]
        
        self.correct_steps = len(correct)
        self.incorrect_steps = len(incorrect)
        self.flat_foot_steps = len(flat_foot)
        
        self.correct_percentage = (self.correct_steps / self.total_steps) * 100 if self.total_steps > 0 else 0
        self.incorrect_percentage = (self.incorrect_steps / self.total_steps) * 100 if self.total_steps > 0 else 0
        self.flat_foot_percentage = (self.flat_foot_steps / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        # Average times by step type
        all_durations = [s.duration_ms for s in all_steps]
        self.avg_step_time_ms = np.mean(all_durations) if all_durations else 0
        
        correct_durations = [s.duration_ms for s in correct]
        self.avg_correct_step_time_ms = np.mean(correct_durations) if correct_durations else 0
        
        incorrect_durations = [s.duration_ms for s in incorrect]
        self.avg_incorrect_step_time_ms = np.mean(incorrect_durations) if incorrect_durations else 0
        
        flat_foot_durations = [s.duration_ms for s in flat_foot]
        self.avg_flat_foot_step_time_ms = np.mean(flat_foot_durations) if flat_foot_durations else 0
        
        # Per-foot counts
        self.left_correct_steps = len([s for s in self.left_steps if s.step_type == StepType.HEEL_STRIKE])
        self.left_incorrect_steps = len([s for s in self.left_steps if s.step_type == StepType.TOE_STRIKE])
        self.left_flat_foot_steps = len([s for s in self.left_steps if s.step_type == StepType.FLAT_FOOT])
        self.right_correct_steps = len([s for s in self.right_steps if s.step_type == StepType.HEEL_STRIKE])
        self.right_incorrect_steps = len([s for s in self.right_steps if s.step_type == StepType.TOE_STRIKE])
        self.right_flat_foot_steps = len([s for s in self.right_steps if s.step_type == StepType.FLAT_FOOT])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_info": {
                "path": self.video_path,
                "total_frames": self.total_frames,
                "fps": self.fps,
                "duration_seconds": round(self.duration_seconds, 2)
            },
            "summary": {
                "total_steps": self.total_steps,
                "correct_steps": self.correct_steps,
                "incorrect_steps": self.incorrect_steps,
                "flat_foot_steps": self.flat_foot_steps,
                "correct_percentage": round(self.correct_percentage, 1),
                "incorrect_percentage": round(self.incorrect_percentage, 1),
                "flat_foot_percentage": round(self.flat_foot_percentage, 1)
            },
            "timing": {
                "avg_step_time_ms": round(self.avg_step_time_ms, 1),
                "avg_correct_step_time_ms": round(self.avg_correct_step_time_ms, 1),
                "avg_incorrect_step_time_ms": round(self.avg_incorrect_step_time_ms, 1),
                "avg_flat_foot_step_time_ms": round(self.avg_flat_foot_step_time_ms, 1)
            },
            "per_foot": {
                "left": {
                    "total_steps": len(self.left_steps),
                    "correct_steps": self.left_correct_steps,
                    "incorrect_steps": self.left_incorrect_steps,
                    "flat_foot_steps": self.left_flat_foot_steps
                },
                "right": {
                    "total_steps": len(self.right_steps),
                    "correct_steps": self.right_correct_steps,
                    "incorrect_steps": self.right_incorrect_steps,
                    "flat_foot_steps": self.right_flat_foot_steps
                }
            },
            "detailed_steps": {
                "left": [s.to_dict() for s in self.left_steps],
                "right": [s.to_dict() for s in self.right_steps]
            }
        }


class GaitAnalyzer:
    """
    Analyzes gait patterns from keypoint sequences.
    """
    
    def __init__(self):
        """Initialize the gait analyzer."""
        self.left_tracker = FootTracker(side="left")
        self.right_tracker = FootTracker(side="right")
        self.keypoint_history: list[FrameKeypoints] = []
        
    def reset(self):
        """Reset the analyzer for a new video."""
        self.left_tracker = FootTracker(side="left")
        self.right_tracker = FootTracker(side="right")
        self.keypoint_history = []
    
    def process_frame(self, keypoints: FrameKeypoints) -> dict:
        """
        Process a single frame's keypoints.
        
        Returns current state for visualization.
        """
        self.keypoint_history.append(keypoints)
        
        # Get positions
        ankles = keypoints.get_ankle_positions()
        knees = keypoints.get_knee_positions()
        
        # Update trackers
        if ankles["left"] is not None:
            x, y = ankles["left"]
            knee_y = knees["left"][1] if knees["left"] else None
            self.left_tracker.add_position(
                keypoints.frame_number, 
                keypoints.timestamp_ms, 
                x, y, knee_y
            )
        
        if ankles["right"] is not None:
            x, y = ankles["right"]
            knee_y = knees["right"][1] if knees["right"] else None
            self.right_tracker.add_position(
                keypoints.frame_number,
                keypoints.timestamp_ms,
                x, y, knee_y
            )
        
        # Return current state for visualization
        return {
            "left_phase": self.left_tracker.current_phase.value,
            "right_phase": self.right_tracker.current_phase.value,
            "left_steps": len(self.left_tracker.steps),
            "right_steps": len(self.right_tracker.steps),
            "last_left_step": self.left_tracker.steps[-1].step_type.value if self.left_tracker.steps else None,
            "last_right_step": self.right_tracker.steps[-1].step_type.value if self.right_tracker.steps else None,
        }
    
    def get_results(self, video_path: str, total_frames: int, fps: float) -> GaitAnalysisResult:
        """
        Get the complete analysis results.
        """
        duration = total_frames / fps if fps > 0 else 0
        
        result = GaitAnalysisResult(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration,
            left_steps=self.left_tracker.steps.copy(),
            right_steps=self.right_tracker.steps.copy()
        )
        
        result.compute_metrics()
        return result

