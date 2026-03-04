"""
Enhanced gait analysis with angle of contact calculations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
from statistics import median

from .detector import FrameKeypoints
from .analyzer import StepType, FootPhase


class FootIdentifier:
    """
    Robust foot identification using hip-based validation and spatial tracking.
    
    Combines two methods:
    1. Hip-based: Uses body center to validate left/right assignments
    2. Spatial tracking: Tracks foot positions over time to detect swaps
    """
    
    def __init__(self, history_size: int = 30, swap_threshold: float = 0.7):
        """
        Initialize the foot identifier.
        
        Args:
            history_size: Number of frames to keep in position history
            swap_threshold: Confidence threshold to trigger a swap correction
        """
        self.history_size = history_size
        self.swap_threshold = swap_threshold
        
        # Position history for spatial tracking
        self.left_x_history: list[float] = []
        self.right_x_history: list[float] = []
        
        # Track consecutive mismatches
        self.hip_mismatch_count = 0
        self.spatial_mismatch_count = 0
        
        # Running average positions
        self.left_avg_x: Optional[float] = None
        self.right_avg_x: Optional[float] = None
        
    def validate_and_correct(
        self,
        keypoints: FrameKeypoints
    ) -> tuple[dict[str, Optional[tuple[float, float]]], 
               dict[str, Optional[tuple[float, float]]], 
               dict[str, Optional[tuple[float, float]]],
               float]:
        """
        Validate foot assignments and correct if needed.
        
        Returns:
            Tuple of (corrected_ankles, corrected_knees, corrected_hips, confidence)
        """
        ankles = keypoints.get_ankle_positions()
        knees = keypoints.get_knee_positions()
        hips = keypoints.get_hip_positions()
        body_center = keypoints.get_body_center()
        
        # If we don't have both ankles, return as-is
        if ankles["left"] is None or ankles["right"] is None:
            return ankles, knees, hips, 0.5
        
        # Check 1: Hip-based validation
        hip_suggests_swap = self._check_hip_based(ankles, body_center)
        
        # Check 2: Spatial tracking validation
        spatial_suggests_swap = self._check_spatial_tracking(ankles)
        
        # Update histories
        self._update_history(ankles)
        
        # Calculate confidence and decide on correction
        confidence = 1.0
        should_swap = False
        
        if hip_suggests_swap and spatial_suggests_swap:
            # Both methods agree there's a mismatch
            self.hip_mismatch_count += 1
            self.spatial_mismatch_count += 1
            if self.hip_mismatch_count >= 3 and self.spatial_mismatch_count >= 3:
                should_swap = True
                confidence = 0.9
        elif hip_suggests_swap or spatial_suggests_swap:
            # Only one method suggests swap - be cautious
            if hip_suggests_swap:
                self.hip_mismatch_count += 1
            else:
                self.hip_mismatch_count = max(0, self.hip_mismatch_count - 1)
            
            if spatial_suggests_swap:
                self.spatial_mismatch_count += 1
            else:
                self.spatial_mismatch_count = max(0, self.spatial_mismatch_count - 1)
            
            confidence = 0.7
        else:
            # Both methods agree assignment is correct
            self.hip_mismatch_count = max(0, self.hip_mismatch_count - 1)
            self.spatial_mismatch_count = max(0, self.spatial_mismatch_count - 1)
            confidence = 1.0
        
        if should_swap:
            # Swap the assignments
            ankles = {"left": ankles["right"], "right": ankles["left"]}
            knees = {"left": knees["right"], "right": knees["left"]}
            hips = {"left": hips["right"], "right": hips["left"]}
            # Reset mismatch counts after swap
            self.hip_mismatch_count = 0
            self.spatial_mismatch_count = 0
            # Also swap the running averages
            self.left_avg_x, self.right_avg_x = self.right_avg_x, self.left_avg_x
        
        return ankles, knees, hips, confidence
    
    def _check_hip_based(
        self, 
        ankles: dict[str, Optional[tuple[float, float]]],
        body_center: Optional[tuple[float, float]]
    ) -> bool:
        """
        Check if ankle positions are consistent with body center.
        
        In a normal standing/walking pose (facing camera):
        - Left ankle should be to the left of body center (smaller X)
        - Right ankle should be to the right of body center (larger X)
        
        Returns True if swap is suggested.
        """
        if body_center is None:
            return False
        
        if ankles["left"] is None or ankles["right"] is None:
            return False
        
        center_x = body_center[0]
        left_x = ankles["left"][0]
        right_x = ankles["right"][0]
        
        # Check if positions are inverted
        # Allow some tolerance for crossed legs
        tolerance = 20  # pixels
        
        left_wrong = left_x > center_x + tolerance
        right_wrong = right_x < center_x - tolerance
        
        return left_wrong and right_wrong
    
    def _check_spatial_tracking(
        self,
        ankles: dict[str, Optional[tuple[float, float]]]
    ) -> bool:
        """
        Check if current positions are consistent with historical patterns.
        
        If the "left" foot has suddenly moved to where the "right" foot
        usually is (and vice versa), suggest a swap.
        
        Returns True if swap is suggested.
        """
        if self.left_avg_x is None or self.right_avg_x is None:
            return False
        
        if ankles["left"] is None or ankles["right"] is None:
            return False
        
        left_x = ankles["left"][0]
        right_x = ankles["right"][0]
        
        # Calculate how far current positions are from historical averages
        left_dist_to_left_avg = abs(left_x - self.left_avg_x)
        left_dist_to_right_avg = abs(left_x - self.right_avg_x)
        right_dist_to_left_avg = abs(right_x - self.left_avg_x)
        right_dist_to_right_avg = abs(right_x - self.right_avg_x)
        
        # Check if positions would make more sense swapped
        current_error = left_dist_to_left_avg + right_dist_to_right_avg
        swapped_error = left_dist_to_right_avg + right_dist_to_left_avg
        
        # Suggest swap if swapped positions would be closer to historical pattern
        # Use a margin to avoid flip-flopping
        margin = 30  # pixels
        return swapped_error < current_error - margin
    
    def _update_history(self, ankles: dict[str, Optional[tuple[float, float]]]):
        """Update position history with current frame."""
        if ankles["left"] is not None:
            self.left_x_history.append(ankles["left"][0])
            if len(self.left_x_history) > self.history_size:
                self.left_x_history.pop(0)
            self.left_avg_x = np.mean(self.left_x_history)
        
        if ankles["right"] is not None:
            self.right_x_history.append(ankles["right"][0])
            if len(self.right_x_history) > self.history_size:
                self.right_x_history.pop(0)
            self.right_avg_x = np.mean(self.right_x_history)
    
    def reset(self):
        """Reset the identifier state."""
        self.left_x_history = []
        self.right_x_history = []
        self.hip_mismatch_count = 0
        self.spatial_mismatch_count = 0
        self.left_avg_x = None
        self.right_avg_x = None


def calculate_knee_angle(
    hip: Optional[tuple[float, float]],
    knee: Optional[tuple[float, float]],
    ankle: Optional[tuple[float, float]]
) -> Optional[float]:
    """
    Calculate the knee flexion angle (hip-knee-ankle angle).
    
    Args:
        hip: (x, y) position of hip
        knee: (x, y) position of knee
        ankle: (x, y) position of ankle
        
    Returns:
        Angle in degrees. 180° = fully extended, lower values = more flexion.
        Returns None if any position is missing.
    """
    if hip is None or knee is None or ankle is None:
        return None
    
    # Vector from knee to hip
    v1 = (hip[0] - knee[0], hip[1] - knee[1])
    # Vector from knee to ankle
    v2 = (ankle[0] - knee[0], ankle[1] - knee[1])
    
    # Calculate angle using dot product
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return None
    
    cos_angle = dot / (mag1 * mag2)
    # Clamp to avoid numerical errors
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


@dataclass
class EnhancedStep:
    """Represents a single step with angle and knee flexion measurements."""
    foot: str  # "left" or "right"
    step_type: StepType
    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    
    # Angle of contact in degrees
    # For heel strike: angle between foot approach and horizontal (positive = heel down)
    # For toe strike: angle at toe vertex
    contact_angle_degrees: float
    
    # Stance time - time the foot is in contact with ground
    stance_time_ms: float
    
    # Knee flexion metrics (degrees, 180° = fully extended)
    knee_flexion_at_contact: float = 180.0  # Knee angle at initial contact
    min_knee_flexion: float = 180.0         # Minimum angle during stance (max flexion)
    max_knee_flexion: float = 180.0         # Maximum angle during stance (max extension)
    
    # Contact position (ankle at initial contact) for step length estimation (pixels)
    contact_ankle_x: Optional[float] = None
    contact_ankle_y: Optional[float] = None
    
    def to_dict(self) -> dict:
        d = {
            "foot": self.foot,
            "step_type": self.step_type.value,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_ms": round(self.start_time_ms, 1),
            "end_time_ms": round(self.end_time_ms, 1),
            "duration_ms": round(self.duration_ms, 1),
            "contact_angle_degrees": round(self.contact_angle_degrees, 2),
            "stance_time_ms": round(self.stance_time_ms, 1),
            "knee_flexion_at_contact": round(self.knee_flexion_at_contact, 2),
            "min_knee_flexion": round(self.min_knee_flexion, 2),
            "max_knee_flexion": round(self.max_knee_flexion, 2)
        }
        if self.contact_ankle_x is not None and self.contact_ankle_y is not None:
            d["contact_ankle_x"] = round(self.contact_ankle_x, 2)
            d["contact_ankle_y"] = round(self.contact_ankle_y, 2)
        return d


@dataclass
class EnhancedFootTracker:
    """Tracks foot with angle and knee flexion calculations."""
    side: str  # "left" or "right"
    
    # Position history: (frame, time_ms, ankle_x, ankle_y, knee_x, knee_y, hip_x, hip_y)
    positions: list[tuple[int, float, float, float, Optional[float], Optional[float], Optional[float], Optional[float]]] = field(default_factory=list)
    velocities_x: list[float] = field(default_factory=list)
    velocities_y: list[float] = field(default_factory=list)
    
    # Knee angle history for each frame
    knee_angles: list[Optional[float]] = field(default_factory=list)
    
    current_phase: FootPhase = FootPhase.UNKNOWN
    phase_start_frame: int = 0
    phase_start_time: float = 0.0
    
    steps: list[EnhancedStep] = field(default_factory=list)
    
    # Stance tracking
    stance_start_frame: int = 0
    stance_start_time: float = 0.0
    
    # Knee angles during current stance phase
    stance_knee_angles: list[float] = field(default_factory=list)
    
    # Parameters
    stance_velocity_threshold: float = 2.0
    swing_velocity_threshold: float = 5.0
    min_stance_frames: int = 3
    
    # Ground level estimation (updated dynamically)
    ground_y: float = 0.0
    max_ankle_y: float = 0.0  # Track lowest point (highest Y in image coords)
    
    def add_position(
        self, 
        frame: int, 
        time_ms: float, 
        ankle_x: float, 
        ankle_y: float,
        knee_x: Optional[float] = None,
        knee_y: Optional[float] = None,
        hip_x: Optional[float] = None,
        hip_y: Optional[float] = None
    ):
        """Add a new position observation."""
        self.positions.append((frame, time_ms, ankle_x, ankle_y, knee_x, knee_y, hip_x, hip_y))
        
        # Calculate knee angle if we have all positions
        hip = (hip_x, hip_y) if hip_x is not None and hip_y is not None else None
        knee = (knee_x, knee_y) if knee_x is not None and knee_y is not None else None
        ankle = (ankle_x, ankle_y)
        
        knee_angle = calculate_knee_angle(hip, knee, ankle)
        self.knee_angles.append(knee_angle)
        
        # Track knee angles during stance
        if self.current_phase == FootPhase.STANCE and knee_angle is not None:
            self.stance_knee_angles.append(knee_angle)
        
        # Update ground estimate (highest Y value = lowest in image)
        if ankle_y > self.max_ankle_y:
            self.max_ankle_y = ankle_y
            self.ground_y = ankle_y
        
        # Calculate velocities
        if len(self.positions) >= 2:
            prev = self.positions[-2]
            dt = max(1, frame - prev[0])
            vx = (ankle_x - prev[2]) / dt
            vy = (ankle_y - prev[3]) / dt  # positive = moving down
            self.velocities_x.append(vx)
            self.velocities_y.append(vy)
        else:
            self.velocities_x.append(0.0)
            self.velocities_y.append(0.0)
        
        self._update_phase(frame, time_ms, ankle_y)
    
    def _update_phase(self, frame: int, time_ms: float, ankle_y: float):
        """Update phase and detect step completions."""
        if len(self.velocities_y) < 3:
            return
        
        recent_vy = self.velocities_y[-3:]
        avg_vy = np.mean(recent_vy)
        
        # Detect transitions
        if self.current_phase != FootPhase.STANCE:
            if abs(avg_vy) < self.stance_velocity_threshold:
                # Transitioning to stance
                if self.current_phase == FootPhase.SWING:
                    self._complete_step(frame, time_ms)
                
                self.current_phase = FootPhase.STANCE
                self.phase_start_frame = frame
                self.phase_start_time = time_ms
                self.stance_start_frame = frame
                self.stance_start_time = time_ms
                # Reset stance knee angles for new stance phase
                self.stance_knee_angles = []
                # Add current knee angle if available
                if self.knee_angles and self.knee_angles[-1] is not None:
                    self.stance_knee_angles.append(self.knee_angles[-1])
        
        elif self.current_phase == FootPhase.STANCE:
            if abs(avg_vy) > self.swing_velocity_threshold:
                # Transitioning to swing - record stance end
                self.current_phase = FootPhase.SWING
                self.phase_start_frame = frame
                self.phase_start_time = time_ms
    
    def _calculate_contact_angle(self) -> tuple[float, StepType]:
        """
        Calculate the angle of contact based on ankle trajectory.
        
        Returns: (angle_degrees, step_type)
        
        For heel strike: angle of descent (positive = proper heel-first contact)
        For toe strike: angle at toe vertex (smaller angle = more toe walking)
        """
        if len(self.positions) < 5:
            return 0.0, StepType.UNKNOWN
        
        # Get positions during landing phase (last 5-10 frames before stance)
        landing_positions = self.positions[-10:] if len(self.positions) >= 10 else self.positions[-5:]
        
        # Calculate average velocity vector during landing
        vx_landing = np.mean(self.velocities_x[-5:]) if len(self.velocities_x) >= 5 else 0
        vy_landing = np.mean(self.velocities_y[-5:]) if len(self.velocities_y) >= 5 else 0
        
        # Calculate angle of approach
        # atan2(vy, vx) gives angle from horizontal
        # Positive vy = moving down, positive vx = moving right
        if abs(vx_landing) < 0.01 and abs(vy_landing) < 0.01:
            return 0.0, StepType.UNKNOWN
        
        angle_rad = np.arctan2(vy_landing, abs(vx_landing))
        angle_deg = np.degrees(angle_rad)
        
        # Analyze vertical velocity pattern for step type classification
        vy_variance = np.var(self.velocities_y[-5:]) if len(self.velocities_y) >= 5 else 0
        
        # Use knee-ankle relationship if available for better angle estimation
        last_pos = self.positions[-1]
        if last_pos[4] is not None and last_pos[5] is not None:
            # We have knee position - calculate shin angle
            ankle_x, ankle_y = last_pos[2], last_pos[3]
            knee_x, knee_y = last_pos[4], last_pos[5]
            
            # Shin vector (knee to ankle)
            shin_dx = ankle_x - knee_x
            shin_dy = ankle_y - knee_y  # positive = ankle below knee
            
            # Angle of shin from vertical
            shin_angle = np.degrees(np.arctan2(shin_dx, shin_dy))
            
            # Combine with velocity angle for better estimate
            angle_deg = (angle_deg + abs(shin_angle)) / 2
        
        # Classify step type based on descent pattern
        # Heel strike: more gradual descent, larger horizontal movement
        # Toe strike: steeper descent, less horizontal movement
        
        horizontal_movement = abs(np.mean([p[2] for p in landing_positions[-3:]]) - 
                                   np.mean([p[2] for p in landing_positions[:3]]))
        
        if vy_landing > 3 and vy_variance < 2 and horizontal_movement < 5:
            # Rapid, uniform descent with little horizontal = toe strike
            step_type = StepType.TOE_STRIKE
            # For toe strike, report the acute angle at toe
            angle_deg = 90 - abs(angle_deg) if abs(angle_deg) < 90 else abs(angle_deg) - 90
        elif horizontal_movement > 5 or angle_deg < 45:
            step_type = StepType.HEEL_STRIKE
        else:
            step_type = StepType.FLAT_FOOT
        
        return abs(angle_deg), step_type
    
    def _complete_step(self, frame: int, time_ms: float):
        """Complete a step with angle and knee flexion calculations."""
        if self.phase_start_frame == 0:
            return
        
        # Calculate contact angle and step type
        contact_angle, step_type = self._calculate_contact_angle()
        
        duration = time_ms - self.phase_start_time
        
        # Calculate stance time (time until next swing phase starts)
        # This will be updated when we transition to swing
        stance_time = duration  # Initial estimate
        
        # Get knee flexion metrics from stance phase
        knee_at_contact = 180.0
        min_knee = 180.0
        max_knee = 180.0
        
        if self.stance_knee_angles:
            knee_at_contact = self.stance_knee_angles[0]
            min_knee = min(self.stance_knee_angles)
            max_knee = max(self.stance_knee_angles)
        elif self.knee_angles:
            # Use recent knee angle if stance angles not available
            recent_angles = [a for a in self.knee_angles[-10:] if a is not None]
            if recent_angles:
                knee_at_contact = recent_angles[0]
                min_knee = min(recent_angles)
                max_knee = max(recent_angles)
        
        # Contact position at step start (for step length)
        contact_x, contact_y = None, None
        for p in self.positions:
            if p[0] >= self.phase_start_frame:
                contact_x, contact_y = p[2], p[3]
                break
        if contact_x is None and self.positions:
            contact_x, contact_y = self.positions[-1][2], self.positions[-1][3]
        
        if duration > 100:  # Minimum step duration
            step = EnhancedStep(
                foot=self.side,
                step_type=step_type,
                start_frame=self.phase_start_frame,
                end_frame=frame,
                start_time_ms=self.phase_start_time,
                end_time_ms=time_ms,
                duration_ms=duration,
                contact_angle_degrees=contact_angle,
                stance_time_ms=stance_time,
                knee_flexion_at_contact=knee_at_contact,
                min_knee_flexion=min_knee,
                max_knee_flexion=max_knee,
                contact_ankle_x=contact_x,
                contact_ankle_y=contact_y
            )
            self.steps.append(step)
    
    def finalize_stance_times(self):
        """Update stance times based on when swing phases started."""
        # Go through steps and update stance times based on actual phase durations
        for i, step in enumerate(self.steps):
            # Find the frame range where this step's stance phase occurred
            # and calculate actual time foot was on ground
            step_frames = [p for p in self.positions 
                         if step.start_frame <= p[0] <= step.end_frame + 30]  # Include some buffer
            
            if len(step_frames) >= 2:
                # Estimate stance time from low-velocity period
                stance_start = None
                stance_end = None
                
                for j, pos in enumerate(step_frames):
                    frame_idx = self.positions.index(pos)
                    if frame_idx < len(self.velocities_y):
                        vy = abs(self.velocities_y[frame_idx])
                        if vy < self.stance_velocity_threshold:
                            if stance_start is None:
                                stance_start = pos[1]  # time_ms
                            stance_end = pos[1]
                
                if stance_start is not None and stance_end is not None:
                    step.stance_time_ms = stance_end - stance_start


def calculate_statistics(values: list[float]) -> dict:
    """Calculate mean, min, max, median, and standard deviation for a list of values."""
    if not values:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "std": 0.0,
            "count": 0
        }
    
    return {
        "mean": round(float(np.mean(values)), 2),
        "min": round(float(np.min(values)), 2),
        "max": round(float(np.max(values)), 2),
        "median": round(float(median(values)), 2),
        "std": round(float(np.std(values)), 2),
        "count": len(values)
    }


@dataclass
class EnhancedGaitAnalysisResult:
    """Complete results with angle and knee flexion analysis."""
    video_path: str
    total_frames: int
    fps: float
    duration_seconds: float
    output_video_path: Optional[str] = None
    
    left_steps: list[EnhancedStep] = field(default_factory=list)
    right_steps: list[EnhancedStep] = field(default_factory=list)
    
    # Summary stats
    total_steps: int = 0
    correct_steps: int = 0      # heel_strike
    incorrect_steps: int = 0    # toe_strike
    flat_foot_steps: int = 0    # flat_foot
    correct_percentage: float = 0.0
    incorrect_percentage: float = 0.0
    flat_foot_percentage: float = 0.0
    average_step_length_px: float = 0.0  # Average distance between consecutive contacts (pixels)
    
    def _step_lengths_from_steps(self, steps: list) -> list[float]:
        """Compute step lengths (pixels) between consecutive same-foot contacts."""
        lengths = []
        for i in range(1, len(steps)):
            a, b = steps[i - 1], steps[i]
            ax, ay = getattr(a, "contact_ankle_x", None), getattr(a, "contact_ankle_y", None)
            bx, by = getattr(b, "contact_ankle_x", None), getattr(b, "contact_ankle_y", None)
            if ax is not None and ay is not None and bx is not None and by is not None:
                lengths.append(float(np.hypot(bx - ax, by - ay)))
        return lengths
    
    def _compute_step_metrics(self, steps: list[EnhancedStep]) -> dict:
        """Compute metrics for a list of steps."""
        return {
            "angles": calculate_statistics([s.contact_angle_degrees for s in steps]),
            "stance_time_ms": calculate_statistics([s.stance_time_ms for s in steps]),
            "knee_flexion": {
                "at_contact": calculate_statistics([s.knee_flexion_at_contact for s in steps]),
                "min_during_stance": calculate_statistics([s.min_knee_flexion for s in steps]),
                "max_during_stance": calculate_statistics([s.max_knee_flexion for s in steps])
            }
        }
    
    def compute_metrics(self) -> dict:
        """Compute all metrics including angle and knee flexion statistics."""
        all_steps = self.left_steps + self.right_steps
        self.total_steps = len(all_steps)
        
        # Categorize by step type
        correct = [s for s in all_steps if s.step_type == StepType.HEEL_STRIKE]
        incorrect = [s for s in all_steps if s.step_type == StepType.TOE_STRIKE]
        flat_foot = [s for s in all_steps if s.step_type == StepType.FLAT_FOOT]
        
        self.correct_steps = len(correct)
        self.incorrect_steps = len(incorrect)
        self.flat_foot_steps = len(flat_foot)
        
        self.correct_percentage = (self.correct_steps / self.total_steps * 100) if self.total_steps > 0 else 0
        self.incorrect_percentage = (self.incorrect_steps / self.total_steps * 100) if self.total_steps > 0 else 0
        self.flat_foot_percentage = (self.flat_foot_steps / self.total_steps * 100) if self.total_steps > 0 else 0
        
        # Average step length (pixels) from consecutive same-foot contacts
        left_lengths = self._step_lengths_from_steps(self.left_steps)
        right_lengths = self._step_lengths_from_steps(self.right_steps)
        all_lengths = left_lengths + right_lengths
        self.average_step_length_px = float(np.mean(all_lengths)) if all_lengths else 0.0
        
        # Per-foot step categorization
        left_correct = [s for s in self.left_steps if s.step_type == StepType.HEEL_STRIKE]
        left_incorrect = [s for s in self.left_steps if s.step_type == StepType.TOE_STRIKE]
        left_flat_foot = [s for s in self.left_steps if s.step_type == StepType.FLAT_FOOT]
        
        right_correct = [s for s in self.right_steps if s.step_type == StepType.HEEL_STRIKE]
        right_incorrect = [s for s in self.right_steps if s.step_type == StepType.TOE_STRIKE]
        right_flat_foot = [s for s in self.right_steps if s.step_type == StepType.FLAT_FOOT]
        
        return {
            "left_foot": {
                "correct_steps": self._compute_step_metrics(left_correct),
                "incorrect_steps": self._compute_step_metrics(left_incorrect),
                "flat_foot_steps": self._compute_step_metrics(left_flat_foot),
                "all_steps": self._compute_step_metrics(self.left_steps)
            },
            "right_foot": {
                "correct_steps": self._compute_step_metrics(right_correct),
                "incorrect_steps": self._compute_step_metrics(right_incorrect),
                "flat_foot_steps": self._compute_step_metrics(right_flat_foot),
                "all_steps": self._compute_step_metrics(self.right_steps)
            },
            "overall": {
                "correct_steps": self._compute_step_metrics(correct),
                "incorrect_steps": self._compute_step_metrics(incorrect),
                "flat_foot_steps": self._compute_step_metrics(flat_foot),
                "all_steps": self._compute_step_metrics(all_steps)
            }
        }
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        metrics = self.compute_metrics()
        
        return {
            "video_info": {
                "path": self.video_path,
                "total_frames": self.total_frames,
                "fps": round(self.fps, 2),
                "duration_seconds": round(self.duration_seconds, 2),
                "output_video_path": self.output_video_path
            },
            "summary": {
                "total_steps": self.total_steps,
                "correct_steps": self.correct_steps,
                "incorrect_steps": self.incorrect_steps,
                "flat_foot_steps": self.flat_foot_steps,
                "correct_percentage": round(self.correct_percentage, 1),
                "incorrect_percentage": round(self.incorrect_percentage, 1),
                "flat_foot_percentage": round(self.flat_foot_percentage, 1),
                "average_step_length_px": round(self.average_step_length_px, 2)
            },
            "metrics": metrics,
            "detailed_steps": {
                "left": [s.to_dict() for s in self.left_steps],
                "right": [s.to_dict() for s in self.right_steps]
            }
        }


class EnhancedGaitAnalyzer:
    """Enhanced gait analyzer with angle calculations and robust foot identification."""
    
    def __init__(self):
        self.left_tracker = EnhancedFootTracker(side="left")
        self.right_tracker = EnhancedFootTracker(side="right")
        self.foot_identifier = FootIdentifier()
        self.keypoint_history: list[FrameKeypoints] = []
    
    def reset(self):
        """Reset for a new video."""
        self.left_tracker = EnhancedFootTracker(side="left")
        self.right_tracker = EnhancedFootTracker(side="right")
        self.foot_identifier.reset()
        self.keypoint_history = []
    
    def process_frame(self, keypoints: FrameKeypoints) -> dict:
        """Process a single frame with foot identification validation."""
        self.keypoint_history.append(keypoints)
        
        # Validate and potentially correct foot assignments
        ankles, knees, hips, confidence = self.foot_identifier.validate_and_correct(keypoints)
        
        # Update left foot tracker
        if ankles["left"] is not None:
            ax, ay = ankles["left"]
            kx, ky = knees["left"] if knees["left"] else (None, None)
            hx, hy = hips["left"] if hips["left"] else (None, None)
            self.left_tracker.add_position(
                keypoints.frame_number,
                keypoints.timestamp_ms,
                ax, ay, kx, ky, hx, hy
            )
        
        # Update right foot tracker
        if ankles["right"] is not None:
            ax, ay = ankles["right"]
            kx, ky = knees["right"] if knees["right"] else (None, None)
            hx, hy = hips["right"] if hips["right"] else (None, None)
            self.right_tracker.add_position(
                keypoints.frame_number,
                keypoints.timestamp_ms,
                ax, ay, kx, ky, hx, hy
            )
        
        # Get current knee angles for visualization
        left_knee_angle = self.left_tracker.knee_angles[-1] if self.left_tracker.knee_angles else None
        right_knee_angle = self.right_tracker.knee_angles[-1] if self.right_tracker.knee_angles else None
        
        # Return current state for visualization
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
            "foot_id_confidence": confidence,
        }
    
    def get_results(self, video_path: str, total_frames: int, fps: float, output_video: str = None) -> EnhancedGaitAnalysisResult:
        """Get complete analysis results."""
        # Finalize stance times
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

