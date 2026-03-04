"""
Enhanced gait analyzer that uses actual heel and toe keypoints.

When heel and toe positions are available (from YOLO Lower Body or OpenPose),
this analyzer can precisely determine:
- Heel Strike: Heel touches ground first (correct gait)
- Toe Strike: Toe touches ground first (toe walking - incorrect)
- Flat Foot: Heel and toe touch simultaneously (suboptimal)

This provides much more accurate gait classification than estimating from ankle position alone.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from statistics import median

from .base_detector import GaitKeypoints
from .analyzer import StepType, FootPhase
from .angle_analyzer import (
    EnhancedStep,
    EnhancedFootTracker,
    EnhancedGaitAnalysisResult,
    FootIdentifier,
    calculate_knee_angle,
    calculate_statistics
)


@dataclass
class HeelToeStep(EnhancedStep):
    """
    Step with detailed heel/toe contact information.
    
    Extends EnhancedStep with actual heel/toe positions at contact.
    """
    # Heel/toe Y positions at contact (lower Y = higher in image = off ground)
    heel_y_at_contact: Optional[float] = None
    toe_y_at_contact: Optional[float] = None
    
    # Heel-toe differential at contact (positive = heel lower = heel first)
    heel_toe_diff: float = 0.0
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "heel_y_at_contact": round(self.heel_y_at_contact, 1) if self.heel_y_at_contact else None,
            "toe_y_at_contact": round(self.toe_y_at_contact, 1) if self.toe_y_at_contact else None,
            "heel_toe_diff": round(self.heel_toe_diff, 2)
        })
        return base


@dataclass
class HeelToeFootTracker:
    """
    Enhanced foot tracker using actual heel and toe positions.
    
    When heel/toe keypoints are available, step classification is based on
    actual contact positions rather than ankle trajectory estimation.
    """
    side: str  # "left" or "right"
    
    # Position history: (frame, time_ms, heel_x, heel_y, toe_x, toe_y, ankle_x, ankle_y, knee_x, knee_y, hip_x, hip_y)
    positions: list[tuple] = field(default_factory=list)
    
    # Velocities
    heel_velocities_y: list[float] = field(default_factory=list)
    toe_velocities_y: list[float] = field(default_factory=list)
    ankle_velocities_y: list[float] = field(default_factory=list)
    
    # Knee angles
    knee_angles: list[Optional[float]] = field(default_factory=list)
    
    # Phase tracking
    current_phase: FootPhase = FootPhase.UNKNOWN
    phase_start_frame: int = 0
    phase_start_time: float = 0.0
    
    # Steps
    steps: list[HeelToeStep] = field(default_factory=list)
    
    # Stance tracking
    stance_knee_angles: list[float] = field(default_factory=list)
    
    # Ground level tracking
    ground_y: float = 0.0
    max_heel_y: float = 0.0
    max_toe_y: float = 0.0
    
    # Parameters
    ground_contact_threshold: float = 15.0  # pixels from ground level
    velocity_threshold: float = 3.0         # pixels/frame for stance detection
    heel_toe_diff_threshold: float = 8.0    # pixels difference to determine heel/toe first
    
    def add_position(
        self,
        frame: int,
        time_ms: float,
        heel: Optional[tuple[float, float]],
        toe: Optional[tuple[float, float]],
        ankle: Optional[tuple[float, float]],
        knee: Optional[tuple[float, float]],
        hip: Optional[tuple[float, float]]
    ):
        """Add a new position observation with heel and toe."""
        heel_x = heel[0] if heel else None
        heel_y = heel[1] if heel else None
        toe_x = toe[0] if toe else None
        toe_y = toe[1] if toe else None
        ankle_x = ankle[0] if ankle else None
        ankle_y = ankle[1] if ankle else None
        knee_x = knee[0] if knee else None
        knee_y = knee[1] if knee else None
        hip_x = hip[0] if hip else None
        hip_y = hip[1] if hip else None
        
        self.positions.append((
            frame, time_ms,
            heel_x, heel_y, toe_x, toe_y,
            ankle_x, ankle_y, knee_x, knee_y, hip_x, hip_y
        ))
        
        # Update ground level estimates (higher Y = lower in image)
        if heel_y is not None and heel_y > self.max_heel_y:
            self.max_heel_y = heel_y
            self.ground_y = max(self.ground_y, heel_y)
        if toe_y is not None and toe_y > self.max_toe_y:
            self.max_toe_y = toe_y
            self.ground_y = max(self.ground_y, toe_y)
        
        # Calculate velocities
        self._update_velocities(frame, heel_y, toe_y, ankle_y)
        
        # Calculate knee angle
        knee_angle = calculate_knee_angle(hip, knee, ankle)
        self.knee_angles.append(knee_angle)
        
        if self.current_phase == FootPhase.STANCE and knee_angle is not None:
            self.stance_knee_angles.append(knee_angle)
        
        # Update phase
        self._update_phase(frame, time_ms, heel_y, toe_y)
    
    def _update_velocities(
        self,
        frame: int,
        heel_y: Optional[float],
        toe_y: Optional[float],
        ankle_y: Optional[float]
    ):
        """Update velocity calculations."""
        if len(self.positions) < 2:
            self.heel_velocities_y.append(0.0)
            self.toe_velocities_y.append(0.0)
            self.ankle_velocities_y.append(0.0)
            return
        
        prev = self.positions[-2]
        dt = max(1, frame - prev[0])
        
        # Heel velocity
        prev_heel_y = prev[3]
        if heel_y is not None and prev_heel_y is not None:
            self.heel_velocities_y.append((heel_y - prev_heel_y) / dt)
        else:
            self.heel_velocities_y.append(0.0)
        
        # Toe velocity
        prev_toe_y = prev[5]
        if toe_y is not None and prev_toe_y is not None:
            self.toe_velocities_y.append((toe_y - prev_toe_y) / dt)
        else:
            self.toe_velocities_y.append(0.0)
        
        # Ankle velocity
        prev_ankle_y = prev[7]
        if ankle_y is not None and prev_ankle_y is not None:
            self.ankle_velocities_y.append((ankle_y - prev_ankle_y) / dt)
        else:
            self.ankle_velocities_y.append(0.0)
    
    def _update_phase(
        self,
        frame: int,
        time_ms: float,
        heel_y: Optional[float],
        toe_y: Optional[float]
    ):
        """Update gait phase based on heel/toe positions."""
        if len(self.heel_velocities_y) < 3:
            return
        
        # Check if foot is on ground (close to max Y values)
        heel_on_ground = heel_y is not None and abs(heel_y - self.ground_y) < self.ground_contact_threshold
        toe_on_ground = toe_y is not None and abs(toe_y - self.ground_y) < self.ground_contact_threshold
        
        # Use velocity for phase detection
        recent_heel_vy = np.mean(self.heel_velocities_y[-3:]) if self.heel_velocities_y else 0
        recent_toe_vy = np.mean(self.toe_velocities_y[-3:]) if self.toe_velocities_y else 0
        
        foot_stationary = abs(recent_heel_vy) < self.velocity_threshold and abs(recent_toe_vy) < self.velocity_threshold
        
        if self.current_phase != FootPhase.STANCE:
            if (heel_on_ground or toe_on_ground) and foot_stationary:
                # Transitioning to stance - complete the step
                if self.current_phase == FootPhase.SWING:
                    self._complete_step(frame, time_ms)
                
                self.current_phase = FootPhase.STANCE
                self.phase_start_frame = frame
                self.phase_start_time = time_ms
                self.stance_knee_angles = []
                if self.knee_angles and self.knee_angles[-1] is not None:
                    self.stance_knee_angles.append(self.knee_angles[-1])
        
        elif self.current_phase == FootPhase.STANCE:
            # Check for swing phase transition
            if not foot_stationary and not heel_on_ground and not toe_on_ground:
                self.current_phase = FootPhase.SWING
                self.phase_start_frame = frame
                self.phase_start_time = time_ms
    
    def _classify_step_type(self) -> tuple[StepType, float, Optional[float], Optional[float]]:
        """
        Classify step type based on heel/toe positions at contact.
        
        Returns: (step_type, heel_toe_diff, heel_y, toe_y)
        """
        if len(self.positions) < 5:
            return StepType.UNKNOWN, 0.0, None, None
        
        # Get positions around contact (last few frames of swing phase)
        landing_positions = self.positions[-10:] if len(self.positions) >= 10 else self.positions[-5:]
        
        # Find the frame where contact likely occurred
        # (where heel or toe first gets close to ground level)
        contact_heel_y = None
        contact_toe_y = None
        
        for pos in landing_positions:
            heel_y = pos[3]
            toe_y = pos[5]
            
            if heel_y is not None and abs(heel_y - self.ground_y) < self.ground_contact_threshold * 2:
                contact_heel_y = heel_y
            if toe_y is not None and abs(toe_y - self.ground_y) < self.ground_contact_threshold * 2:
                contact_toe_y = toe_y
            
            # If we have both, we can compare
            if contact_heel_y is not None and contact_toe_y is not None:
                break
        
        # If we only have heel or toe, use the last available
        if contact_heel_y is None:
            for pos in reversed(landing_positions):
                if pos[3] is not None:
                    contact_heel_y = pos[3]
                    break
        
        if contact_toe_y is None:
            for pos in reversed(landing_positions):
                if pos[5] is not None:
                    contact_toe_y = pos[5]
                    break
        
        # Calculate heel-toe difference
        # Positive = heel is lower (larger Y) = heel contacts first
        # Negative = toe is lower = toe contacts first
        if contact_heel_y is not None and contact_toe_y is not None:
            heel_toe_diff = contact_heel_y - contact_toe_y
            
            if heel_toe_diff > self.heel_toe_diff_threshold:
                return StepType.HEEL_STRIKE, heel_toe_diff, contact_heel_y, contact_toe_y
            elif heel_toe_diff < -self.heel_toe_diff_threshold:
                return StepType.TOE_STRIKE, heel_toe_diff, contact_heel_y, contact_toe_y
            else:
                return StepType.FLAT_FOOT, heel_toe_diff, contact_heel_y, contact_toe_y
        
        # Fall back to velocity-based classification if heel/toe not available
        return self._classify_by_velocity()
    
    def _classify_by_velocity(self) -> tuple[StepType, float, Optional[float], Optional[float]]:
        """Fallback classification using velocity patterns."""
        if len(self.heel_velocities_y) < 5 or len(self.toe_velocities_y) < 5:
            return StepType.UNKNOWN, 0.0, None, None
        
        # Compare descent rates
        heel_descent = np.mean(self.heel_velocities_y[-5:])
        toe_descent = np.mean(self.toe_velocities_y[-5:])
        
        diff = heel_descent - toe_descent
        
        if diff > 2:  # Heel descending faster
            return StepType.HEEL_STRIKE, diff, None, None
        elif diff < -2:  # Toe descending faster
            return StepType.TOE_STRIKE, diff, None, None
        else:
            return StepType.FLAT_FOOT, diff, None, None
    
    def _calculate_contact_angle(self) -> float:
        """Calculate approximate contact angle from foot trajectory."""
        if len(self.positions) < 5:
            return 0.0
        
        recent = self.positions[-5:]
        
        # Use ankle trajectory for angle
        ankle_positions = [(p[6], p[7]) for p in recent if p[6] is not None and p[7] is not None]
        
        if len(ankle_positions) < 2:
            return 0.0
        
        # Calculate trajectory angle
        x_positions = [p[0] for p in ankle_positions]
        y_positions = [p[1] for p in ankle_positions]
        
        dx = x_positions[-1] - x_positions[0]
        dy = y_positions[-1] - y_positions[0]
        
        if abs(dx) < 0.01:
            return 90.0  # Vertical descent
        
        angle = np.degrees(np.arctan2(dy, abs(dx)))
        return abs(angle)
    
    def _complete_step(self, frame: int, time_ms: float):
        """Complete a step with heel/toe classification."""
        if self.phase_start_frame == 0:
            return
        
        # Classify step type using heel/toe positions
        step_type, heel_toe_diff, heel_y, toe_y = self._classify_step_type()
        
        # Calculate contact angle
        contact_angle = self._calculate_contact_angle()
        
        duration = time_ms - self.phase_start_time
        
        # Get knee flexion metrics
        knee_at_contact = 180.0
        min_knee = 180.0
        max_knee = 180.0
        
        if self.stance_knee_angles:
            knee_at_contact = self.stance_knee_angles[0]
            min_knee = min(self.stance_knee_angles)
            max_knee = max(self.stance_knee_angles)
        elif self.knee_angles:
            recent = [a for a in self.knee_angles[-10:] if a is not None]
            if recent:
                knee_at_contact = recent[0]
                min_knee = min(recent)
                max_knee = max(recent)
        
        # Contact position at step start (ankle) for step length
        contact_x, contact_y = None, None
        for p in self.positions:
            if p[0] >= self.phase_start_frame:
                contact_x, contact_y = p[6], p[7]  # ankle_x, ankle_y
                break
        if contact_x is None and self.positions:
            contact_x, contact_y = self.positions[-1][6], self.positions[-1][7]
        
        if duration > 100:  # Minimum step duration
            step = HeelToeStep(
                foot=self.side,
                step_type=step_type,
                start_frame=self.phase_start_frame,
                end_frame=frame,
                start_time_ms=self.phase_start_time,
                end_time_ms=time_ms,
                duration_ms=duration,
                contact_angle_degrees=contact_angle,
                stance_time_ms=duration,
                knee_flexion_at_contact=knee_at_contact,
                min_knee_flexion=min_knee,
                max_knee_flexion=max_knee,
                heel_y_at_contact=heel_y,
                toe_y_at_contact=toe_y,
                heel_toe_diff=heel_toe_diff,
                contact_ankle_x=contact_x,
                contact_ankle_y=contact_y
            )
            self.steps.append(step)
    
    def finalize_stance_times(self):
        """Update stance times based on actual phase durations."""
        pass  # Stance times are calculated during step completion


class HeelToeGaitAnalyzer:
    """
    Gait analyzer optimized for models with heel/toe keypoints.
    
    Provides much more accurate step classification by using actual
    heel and toe positions rather than estimating from ankle trajectory.
    """
    
    def __init__(self):
        self.left_tracker = HeelToeFootTracker(side="left")
        self.right_tracker = HeelToeFootTracker(side="right")
        self.foot_identifier = FootIdentifier()
        self.keypoint_history: list[GaitKeypoints] = []
        self.has_heel_toe: bool = False
    
    def reset(self):
        """Reset for a new video."""
        self.left_tracker = HeelToeFootTracker(side="left")
        self.right_tracker = HeelToeFootTracker(side="right")
        self.foot_identifier.reset()
        self.keypoint_history = []
        self.has_heel_toe = False
    
    def process_frame(self, keypoints: GaitKeypoints) -> dict:
        """Process a frame using heel/toe keypoints when available."""
        self.keypoint_history.append(keypoints)
        
        # Check if we have heel/toe data
        if keypoints.has_heel_toe():
            self.has_heel_toe = True
        
        # Get positions (foot identifier works on ankles/hips)
        heels = keypoints.get_heel_positions()
        toes = keypoints.get_toe_positions()
        ankles = keypoints.get_ankle_positions()
        knees = keypoints.get_knee_positions()
        hips = keypoints.get_hip_positions()
        
        # Update left tracker
        self.left_tracker.add_position(
            keypoints.frame_number,
            keypoints.timestamp_ms,
            heels["left"],
            toes["left"],
            ankles["left"],
            knees["left"],
            hips["left"]
        )
        
        # Update right tracker
        self.right_tracker.add_position(
            keypoints.frame_number,
            keypoints.timestamp_ms,
            heels["right"],
            toes["right"],
            ankles["right"],
            knees["right"],
            hips["right"]
        )
        
        # Get current knee angles
        left_knee_angle = self.left_tracker.knee_angles[-1] if self.left_tracker.knee_angles else None
        right_knee_angle = self.right_tracker.knee_angles[-1] if self.right_tracker.knee_angles else None
        
        return {
            "left_phase": self.left_tracker.current_phase.value,
            "right_phase": self.right_tracker.current_phase.value,
            "left_steps": len(self.left_tracker.steps),
            "right_steps": len(self.right_tracker.steps),
            "last_left_step": self.left_tracker.steps[-1].step_type.value if self.left_tracker.steps else None,
            "last_right_step": self.right_tracker.steps[-1].step_type.value if self.right_tracker.steps else None,
            "last_left_angle": self.left_tracker.steps[-1].contact_angle_degrees if self.left_tracker.steps else None,
            "last_right_angle": self.right_tracker.steps[-1].contact_angle_degrees if self.right_tracker.steps else None,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "has_heel_toe": self.has_heel_toe,
        }
    
    def get_results(
        self,
        video_path: str,
        total_frames: int,
        fps: float,
        output_video: str = None
    ) -> EnhancedGaitAnalysisResult:
        """Get analysis results."""
        self.left_tracker.finalize_stance_times()
        self.right_tracker.finalize_stance_times()
        
        duration = total_frames / fps if fps > 0 else 0
        
        result = EnhancedGaitAnalysisResult(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration,
            output_video_path=output_video,
            left_steps=self.left_tracker.steps.copy(),
            right_steps=self.right_tracker.steps.copy()
        )
        
        return result

