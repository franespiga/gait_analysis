"""
Advanced visualizer with angle display and video output support.
"""

import cv2
import numpy as np
from typing import Optional

from .detector import FrameKeypoints
from .analyzer import StepType


# Color scheme (BGR format)
COLORS = {
    "skeleton": (200, 200, 200),
    "left_ankle": (255, 100, 100),
    "right_ankle": (100, 100, 255),
    "heel_strike": (0, 255, 0),       # Green - correct gait
    "toe_strike": (0, 0, 255),        # Red - incorrect gait
    "flat_foot": (0, 165, 255),       # Orange - suboptimal gait
    "unknown": (128, 128, 128),
    "text_bg": (0, 0, 0),
    "text": (255, 255, 255),
    "angle_text": (0, 255, 255),      # Yellow - contact angle
    "knee_angle": (255, 0, 255),      # Magenta - knee flexion
}

SKELETON_CONNECTIONS = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 7), (7, 9), (6, 8), (8, 10),
]

# Lower body skeleton connections (10 keypoints: hip, knee, ankle, heel, toe)
# Indices: 0-1 hips, 2-3 knees, 4-5 ankles, 6-7 heels, 8-9 toes
SKELETON_CONNECTIONS_LOWER_BODY = [
    (0, 1),   # Left hip to right hip
    (0, 2),   # Left hip to left knee
    (1, 3),   # Right hip to right knee
    (2, 4),   # Left knee to left ankle
    (3, 5),   # Right knee to right ankle
    (4, 6),   # Left ankle to left heel
    (5, 7),   # Right ankle to right heel
    (6, 8),   # Left heel to left toe
    (7, 9),   # Right heel to right toe
    (4, 8),   # Left ankle to left toe
    (5, 9),   # Right ankle to right toe
]


class AdvancedGaitVisualizer:
    """Visualizer with angle display and video writing capabilities."""
    
    def __init__(
        self, 
        show_skeleton: bool = True, 
        show_stats: bool = True,
        show_angles: bool = True
    ):
        self.show_skeleton = show_skeleton
        self.show_stats = show_stats
        self.show_angles = show_angles
        self.video_writer: Optional[cv2.VideoWriter] = None
    
    def init_video_writer(self, output_path: str, fps: float, width: int, height: int):
        """Initialize video writer for output."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        return self.video_writer.isOpened()
    
    def release_video_writer(self):
        """Release the video writer."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
    
    def write_frame(self, frame: np.ndarray):
        """Write a frame to the output video."""
        if self.video_writer is not None:
            self.video_writer.write(frame)
    
    def draw_frame(
        self,
        frame: np.ndarray,
        keypoints: Optional[FrameKeypoints],
        analysis_state: Optional[dict] = None,
        yolo_results=None,
        frame_number: int = 0,
        total_frames: int = 0,
        backend: str = "yolo_coco"
    ) -> np.ndarray:
        """Draw visualization on a frame with angle information."""
        annotated = frame.copy()
        
        # Draw YOLO's built-in visualization (only for standard COCO model)
        # The fine-tuned lower body model may not have skeleton metadata configured
        if yolo_results is not None and len(yolo_results) > 0 and backend == "yolo_coco":
            annotated = yolo_results[0].plot()
        
        # Draw custom overlays (includes full skeleton for lower body model)
        if keypoints is not None and keypoints.raw_keypoints is not None:
            self._draw_custom_keypoints(annotated, keypoints, analysis_state, backend)
        
        # Draw statistics overlay with angles
        if self.show_stats and analysis_state is not None:
            self._draw_stats_overlay(annotated, analysis_state, frame_number, total_frames)
        
        return annotated
    
    def _draw_custom_keypoints(
        self,
        frame: np.ndarray,
        keypoints: FrameKeypoints,
        analysis_state: Optional[dict],
        backend: str = "yolo_coco"
    ):
        """Draw custom keypoint visualization with angle and knee flexion indicators."""
        raw_kpts = keypoints.raw_keypoints
        num_kpts = len(raw_kpts)
        
        # Determine keypoint indices based on backend
        if backend == "yolo_lower" or num_kpts == 10:
            left_ankle_idx, right_ankle_idx = 4, 5
            left_knee_idx, right_knee_idx = 2, 3
            left_heel_idx, right_heel_idx = 6, 7
            left_toe_idx, right_toe_idx = 8, 9
            skeleton_connections = SKELETON_CONNECTIONS_LOWER_BODY
            # Draw all keypoints for lower body model since plot() may not work
            self._draw_all_keypoints_lower_body(frame, raw_kpts)
        else:
            left_ankle_idx, right_ankle_idx = 15, 16
            left_knee_idx, right_knee_idx = 13, 14
            left_heel_idx, right_heel_idx = None, None
            left_toe_idx, right_toe_idx = None, None
            skeleton_connections = SKELETON_CONNECTIONS
        
        if self.show_skeleton:
            self._draw_skeleton(frame, raw_kpts, skeleton_connections)
        
        # Get step type colors
        left_color = self._get_step_color(analysis_state.get("last_left_step") if analysis_state else None)
        right_color = self._get_step_color(analysis_state.get("last_right_step") if analysis_state else None)
        
        # Left ankle
        if left_ankle_idx < num_kpts:
            left_ankle = raw_kpts[left_ankle_idx]
            if left_ankle[2] > 0.5:
                x, y = int(left_ankle[0]), int(left_ankle[1])
                self._draw_ankle_marker(frame, x, y, left_color, "L")
                
                # Draw contact angle if available
                if analysis_state and analysis_state.get("last_left_angle") is not None:
                    angle = analysis_state["last_left_angle"]
                    self._draw_angle_indicator(frame, x, y - 30, angle, COLORS["angle_text"])
        
        # Right ankle
        if right_ankle_idx < num_kpts:
            right_ankle = raw_kpts[right_ankle_idx]
            if right_ankle[2] > 0.5:
                x, y = int(right_ankle[0]), int(right_ankle[1])
                self._draw_ankle_marker(frame, x, y, right_color, "R")
                
                if analysis_state and analysis_state.get("last_right_angle") is not None:
                    angle = analysis_state["last_right_angle"]
                    self._draw_angle_indicator(frame, x, y - 30, angle, COLORS["angle_text"])
        
        # Draw heel markers for lower body model
        if left_heel_idx is not None and left_heel_idx < num_kpts:
            left_heel = raw_kpts[left_heel_idx]
            if left_heel[2] > 0.5:
                hx, hy = int(left_heel[0]), int(left_heel[1])
                cv2.circle(frame, (hx, hy), 6, COLORS["heel_strike"], -1)
        
        if right_heel_idx is not None and right_heel_idx < num_kpts:
            right_heel = raw_kpts[right_heel_idx]
            if right_heel[2] > 0.5:
                hx, hy = int(right_heel[0]), int(right_heel[1])
                cv2.circle(frame, (hx, hy), 6, COLORS["heel_strike"], -1)
        
        # Draw toe markers for lower body model
        if left_toe_idx is not None and left_toe_idx < num_kpts:
            left_toe = raw_kpts[left_toe_idx]
            if left_toe[2] > 0.5:
                tx, ty = int(left_toe[0]), int(left_toe[1])
                cv2.circle(frame, (tx, ty), 6, COLORS["toe_strike"], -1)
        
        if right_toe_idx is not None and right_toe_idx < num_kpts:
            right_toe = raw_kpts[right_toe_idx]
            if right_toe[2] > 0.5:
                tx, ty = int(right_toe[0]), int(right_toe[1])
                cv2.circle(frame, (tx, ty), 6, COLORS["toe_strike"], -1)
        
        # Draw knee flexion angles near knees
        if left_knee_idx < num_kpts:
            left_knee = raw_kpts[left_knee_idx]
            if left_knee[2] > 0.5 and analysis_state:
                kx, ky = int(left_knee[0]), int(left_knee[1])
                left_knee_angle = analysis_state.get("left_knee_angle")
                if left_knee_angle is not None:
                    self._draw_knee_angle(frame, kx, ky, left_knee_angle)
        
        if right_knee_idx < num_kpts:
            right_knee = raw_kpts[right_knee_idx]
            if right_knee[2] > 0.5 and analysis_state:
                kx, ky = int(right_knee[0]), int(right_knee[1])
                right_knee_angle = analysis_state.get("right_knee_angle")
                if right_knee_angle is not None:
                    self._draw_knee_angle(frame, kx, ky, right_knee_angle)
    
    def _draw_knee_angle(self, frame: np.ndarray, x: int, y: int, angle: float):
        """Draw knee flexion angle near the knee."""
        text = f"{angle:.0f}°"
        cv2.putText(frame, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["knee_angle"], 1)
    
    def _draw_all_keypoints_lower_body(self, frame: np.ndarray, keypoints: np.ndarray):
        """
        Draw all keypoints for the lower body model with labels.
        
        Lower body keypoints (10 total):
        0: Left Hip, 1: Right Hip
        2: Left Knee, 3: Right Knee
        4: Left Ankle, 5: Right Ankle
        6: Left Heel, 7: Right Heel
        8: Left Toe, 9: Right Toe
        """
        # Keypoint names and colors
        keypoint_info = [
            ("L_Hip", (255, 200, 100)),    # 0
            ("R_Hip", (100, 200, 255)),    # 1
            ("L_Knee", (255, 150, 100)),   # 2
            ("R_Knee", (100, 150, 255)),   # 3
            ("L_Ankle", (255, 100, 100)),  # 4
            ("R_Ankle", (100, 100, 255)),  # 5
            ("L_Heel", (0, 255, 0)),       # 6 - Green for heel
            ("R_Heel", (0, 255, 0)),       # 7 - Green for heel
            ("L_Toe", (0, 165, 255)),      # 8 - Orange for toe
            ("R_Toe", (0, 165, 255)),      # 9 - Orange for toe
        ]
        
        for idx, (name, color) in enumerate(keypoint_info):
            if idx >= len(keypoints):
                break
            
            kpt = keypoints[idx]
            conf = kpt[2] if len(kpt) > 2 else 1.0
            
            if conf > 0.3 and kpt[0] > 0 and kpt[1] > 0:
                x, y = int(kpt[0]), int(kpt[1])
                
                # Draw keypoint circle
                radius = 8 if "Heel" in name or "Toe" in name else 6
                cv2.circle(frame, (x, y), radius, color, -1)
                cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)  # White border
                
                # Draw label (offset to avoid overlap)
                label_offset = 12
                cv2.putText(frame, name, (x + label_offset, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, connections=None):
        """Draw skeleton connections."""
        if connections is None:
            connections = SKELETON_CONNECTIONS
        
        for start_idx, end_idx in connections:
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue
            start_kpt = keypoints[start_idx]
            end_kpt = keypoints[end_idx]
            
            if start_kpt[2] > 0.5 and end_kpt[2] > 0.5:
                start_point = (int(start_kpt[0]), int(start_kpt[1]))
                end_point = (int(end_kpt[0]), int(end_kpt[1]))
                cv2.line(frame, start_point, end_point, COLORS["skeleton"], 2)
    
    def _draw_ankle_marker(self, frame: np.ndarray, x: int, y: int, color: tuple, label: str):
        """Draw highlighted ankle marker."""
        cv2.circle(frame, (x, y), 15, color, 3)
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.putText(frame, label, (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 2)
    
    def _draw_angle_indicator(self, frame: np.ndarray, x: int, y: int, angle: float, color: tuple):
        """Draw angle value near the ankle."""
        text = f"{angle:.1f}°"
        cv2.putText(frame, text, (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["angle_text"], 2)
    
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
    
    def _draw_stats_overlay(
        self, 
        frame: np.ndarray, 
        analysis_state: dict,
        frame_number: int,
        total_frames: int
    ):
        """Draw statistics overlay with angle and knee flexion information."""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay (larger for more stats)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 280), COLORS["text_bg"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 28
        line_height = 20
        
        # Progress
        progress = (frame_number / total_frames * 100) if total_frames > 0 else 0
        
        # Format knee angles
        left_knee = analysis_state.get('left_knee_angle')
        right_knee = analysis_state.get('right_knee_angle')
        left_knee_str = f"{left_knee:.0f}°" if left_knee is not None else "N/A"
        right_knee_str = f"{right_knee:.0f}°" if right_knee is not None else "N/A"
        
        # Format contact angles
        left_angle = analysis_state.get('last_left_angle')
        right_angle = analysis_state.get('last_right_angle')
        left_angle_str = f"{left_angle:.1f}°" if left_angle is not None else "N/A"
        right_angle_str = f"{right_angle:.1f}°" if right_angle is not None else "N/A"
        
        # Foot ID confidence
        confidence = analysis_state.get('foot_id_confidence', 1.0)
        
        stats_lines = [
            f"Progress: {progress:.1f}% ({frame_number}/{total_frames})",
            f"Foot ID Confidence: {confidence:.0%}",
            f"",
            f"LEFT FOOT: {analysis_state.get('left_phase', 'N/A')}",
            f"  Steps: {analysis_state.get('left_steps', 0)} | Type: {analysis_state.get('last_left_step', 'N/A')}",
            f"  Contact Angle: {left_angle_str} | Knee: {left_knee_str}",
            f"",
            f"RIGHT FOOT: {analysis_state.get('right_phase', 'N/A')}",
            f"  Steps: {analysis_state.get('right_steps', 0)} | Type: {analysis_state.get('last_right_step', 'N/A')}",
            f"  Contact Angle: {right_angle_str} | Knee: {right_knee_str}",
        ]
        
        for line in stats_lines:
            cv2.putText(
                frame, line, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text"], 1
            )
            y_offset += line_height
        
        # Legend at bottom (two lines)
        legend_y = h - 50
        cv2.putText(frame, "GREEN=Heel", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["heel_strike"], 2)
        cv2.putText(frame, "RED=Toe", (120, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["toe_strike"], 2)
        cv2.putText(frame, "ORANGE=Flat", (210, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["flat_foot"], 2)
        
        legend_y2 = h - 25
        cv2.putText(frame, "YELLOW=Contact Angle", (10, legend_y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["angle_text"], 2)
        cv2.putText(frame, "MAGENTA=Knee Angle", (200, legend_y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["knee_angle"], 2)


def create_visualization_window(window_name: str = "Advanced Gait Analysis"):
    """Create a named window for visualization."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    return window_name


def show_frame(window_name: str, frame: np.ndarray, wait_ms: int = 1) -> bool:
    """Display a frame and handle keyboard input."""
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(wait_ms) & 0xFF
    return key != ord('q')


def destroy_windows():
    """Clean up all OpenCV windows."""
    cv2.destroyAllWindows()

