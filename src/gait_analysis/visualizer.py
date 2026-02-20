"""
Visualization module for rendering keypoints and gait analysis results.
"""

import cv2
import numpy as np
from typing import Optional

from .detector import FrameKeypoints, ALL_KEYPOINTS
from .analyzer import StepType


# Color scheme (BGR format)
COLORS = {
    "skeleton": (200, 200, 200),
    "left_ankle": (255, 100, 100),   # Blue-ish for left
    "right_ankle": (100, 100, 255),  # Red-ish for right
    "heel_strike": (0, 255, 0),      # Green for correct
    "toe_strike": (0, 0, 255),       # Red for incorrect
    "flat_foot": (0, 255, 255),      # Yellow for neutral
    "unknown": (128, 128, 128),      # Gray for unknown
    "text_bg": (0, 0, 0),
    "text": (255, 255, 255),
}

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    # Torso
    (5, 6),   # shoulders
    (5, 11),  # left shoulder to hip
    (6, 12),  # right shoulder to hip
    (11, 12), # hips
    # Left leg
    (11, 13), # left hip to knee
    (13, 15), # left knee to ankle
    # Right leg
    (12, 14), # right hip to knee
    (14, 16), # right knee to ankle
    # Arms (optional)
    (5, 7),   # left shoulder to elbow
    (7, 9),   # left elbow to wrist
    (6, 8),   # right shoulder to elbow
    (8, 10),  # right elbow to wrist
]


class GaitVisualizer:
    """
    Visualizes keypoints and gait analysis on video frames.
    """
    
    def __init__(self, show_skeleton: bool = True, show_stats: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            show_skeleton: Whether to draw the skeleton connections
            show_stats: Whether to show statistics overlay
        """
        self.show_skeleton = show_skeleton
        self.show_stats = show_stats
        
    def draw_frame(
        self,
        frame: np.ndarray,
        keypoints: Optional[FrameKeypoints],
        analysis_state: Optional[dict] = None,
        yolo_results = None
    ) -> np.ndarray:
        """
        Draw visualization on a frame.
        
        Args:
            frame: Original BGR frame
            keypoints: Detected keypoints
            analysis_state: Current state from GaitAnalyzer
            yolo_results: Raw YOLO results for built-in visualization
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw YOLO's built-in visualization if available
        if yolo_results is not None and len(yolo_results) > 0:
            annotated = yolo_results[0].plot()
        
        # Draw custom overlays
        if keypoints is not None and keypoints.raw_keypoints is not None:
            self._draw_custom_keypoints(annotated, keypoints, analysis_state)
        
        # Draw statistics overlay
        if self.show_stats and analysis_state is not None:
            self._draw_stats_overlay(annotated, analysis_state)
        
        return annotated
    
    def _draw_custom_keypoints(
        self,
        frame: np.ndarray,
        keypoints: FrameKeypoints,
        analysis_state: Optional[dict]
    ):
        """Draw custom keypoint visualization focused on ankles."""
        raw_kpts = keypoints.raw_keypoints  # Shape: (17, 3) - x, y, conf
        
        # Draw skeleton if enabled
        if self.show_skeleton:
            self._draw_skeleton(frame, raw_kpts)
        
        # Highlight ankles with color based on last step type
        left_color = self._get_step_color(analysis_state.get("last_left_step") if analysis_state else None)
        right_color = self._get_step_color(analysis_state.get("last_right_step") if analysis_state else None)
        
        # Left ankle (index 15)
        left_ankle = raw_kpts[15]
        if left_ankle[2] > 0.5:  # confidence check
            self._draw_ankle_marker(frame, int(left_ankle[0]), int(left_ankle[1]), left_color, "L")
        
        # Right ankle (index 16)
        right_ankle = raw_kpts[16]
        if right_ankle[2] > 0.5:
            self._draw_ankle_marker(frame, int(right_ankle[0]), int(right_ankle[1]), right_color, "R")
    
    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray):
        """Draw skeleton connections."""
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            start_kpt = keypoints[start_idx]
            end_kpt = keypoints[end_idx]
            
            # Only draw if both points are confident
            if start_kpt[2] > 0.5 and end_kpt[2] > 0.5:
                start_point = (int(start_kpt[0]), int(start_kpt[1]))
                end_point = (int(end_kpt[0]), int(end_kpt[1]))
                cv2.line(frame, start_point, end_point, COLORS["skeleton"], 2)
    
    def _draw_ankle_marker(self, frame: np.ndarray, x: int, y: int, color: tuple, label: str):
        """Draw a highlighted marker at ankle position."""
        # Outer circle
        cv2.circle(frame, (x, y), 15, color, 3)
        # Inner filled circle
        cv2.circle(frame, (x, y), 8, color, -1)
        # Label
        cv2.putText(frame, label, (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 2)
    
    def _get_step_color(self, step_type: Optional[str]) -> tuple:
        """Get color based on step type."""
        if step_type == "heel_strike":
            return COLORS["heel_strike"]
        elif step_type == "toe_strike":
            return COLORS["toe_strike"]
        elif step_type == "flat_foot":
            return COLORS["flat_foot"]
        else:
            return COLORS["unknown"]
    
    def _draw_stats_overlay(self, frame: np.ndarray, analysis_state: dict):
        """Draw statistics overlay on the frame."""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 180), COLORS["text_bg"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        y_offset = 35
        line_height = 25
        
        stats_lines = [
            f"Left Foot: {analysis_state.get('left_phase', 'N/A')}",
            f"Right Foot: {analysis_state.get('right_phase', 'N/A')}",
            f"Left Steps: {analysis_state.get('left_steps', 0)}",
            f"Right Steps: {analysis_state.get('right_steps', 0)}",
            f"Last Left: {analysis_state.get('last_left_step', 'N/A')}",
            f"Last Right: {analysis_state.get('last_right_step', 'N/A')}",
        ]
        
        for line in stats_lines:
            cv2.putText(
                frame, line, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 1
            )
            y_offset += line_height
        
        # Draw legend at bottom
        legend_y = h - 40
        cv2.putText(frame, "GREEN=Correct(Heel)", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["heel_strike"], 2)
        cv2.putText(frame, "RED=Incorrect(Toe)", (200, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["toe_strike"], 2)


def create_visualization_window(window_name: str = "Gait Analysis"):
    """Create a named window for visualization."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    return window_name


def show_frame(window_name: str, frame: np.ndarray, wait_ms: int = 1) -> bool:
    """
    Display a frame and handle keyboard input.
    
    Returns:
        False if user pressed 'q' to quit, True otherwise
    """
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(wait_ms) & 0xFF
    
    if key == ord('q'):
        return False
    return True


def destroy_windows():
    """Clean up all OpenCV windows."""
    cv2.destroyAllWindows()

