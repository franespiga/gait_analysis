"""
Keypoint detection module using Ultralytics YOLO pose estimation.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from ultralytics import YOLO


# COCO keypoint indices for lower body
KEYPOINT_INDICES = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# All keypoint names for reference
ALL_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


@dataclass
class FrameKeypoints:
    """Keypoints detected in a single frame."""
    frame_number: int
    timestamp_ms: float
    keypoints: dict[str, tuple[float, float, float]]  # name -> (x, y, confidence)
    raw_keypoints: Optional[np.ndarray] = None  # Full keypoint array for visualization
    
    def get_ankle_positions(self) -> dict[str, Optional[tuple[float, float]]]:
        """Get ankle positions for gait analysis."""
        result = {}
        for side in ["left", "right"]:
            key = f"{side}_ankle"
            if key in self.keypoints:
                x, y, conf = self.keypoints[key]
                if conf > 0.5:  # Only return if confident
                    result[side] = (x, y)
                else:
                    result[side] = None
            else:
                result[side] = None
        return result
    
    def get_knee_positions(self) -> dict[str, Optional[tuple[float, float]]]:
        """Get knee positions for gait analysis."""
        result = {}
        for side in ["left", "right"]:
            key = f"{side}_knee"
            if key in self.keypoints:
                x, y, conf = self.keypoints[key]
                if conf > 0.5:
                    result[side] = (x, y)
                else:
                    result[side] = None
            else:
                result[side] = None
        return result
    
    def get_hip_positions(self) -> dict[str, Optional[tuple[float, float]]]:
        """Get hip positions for body orientation and knee angle calculation."""
        result = {}
        for side in ["left", "right"]:
            key = f"{side}_hip"
            if key in self.keypoints:
                x, y, conf = self.keypoints[key]
                if conf > 0.5:
                    result[side] = (x, y)
                else:
                    result[side] = None
            else:
                result[side] = None
        return result
    
    def get_body_center(self) -> Optional[tuple[float, float]]:
        """Get the body center (midpoint between hips) for orientation."""
        hips = self.get_hip_positions()
        if hips["left"] is not None and hips["right"] is not None:
            center_x = (hips["left"][0] + hips["right"][0]) / 2
            center_y = (hips["left"][1] + hips["right"][1]) / 2
            return (center_x, center_y)
        return None


class KeypointDetector:
    """
    Detects body keypoints in video frames using YOLO pose estimation.
    """
    
    def __init__(self, model_name: str = "yolov8n-pose.pt", device: str = "auto"):
        """
        Initialize the keypoint detector.
        
        Args:
            model_name: YOLO pose model to use (default: yolov8n-pose.pt)
            device: Device to run inference on ('auto', 'cpu', 'cuda', etc.)
        """
        self.model = YOLO(model_name)
        self.device = device
        
    def detect_frame(
        self, 
        frame: np.ndarray, 
        frame_number: int, 
        timestamp_ms: float,
        conf_threshold: float = 0.5
    ) -> Optional[FrameKeypoints]:
        """
        Detect keypoints in a single frame.
        
        Args:
            frame: BGR image as numpy array
            frame_number: Frame index in the video
            timestamp_ms: Timestamp in milliseconds
            conf_threshold: Minimum confidence threshold for person detection
            
        Returns:
            FrameKeypoints object or None if no person detected
        """
        results = self.model(frame, verbose=False, device=self.device)
        
        if len(results) == 0 or results[0].keypoints is None:
            return None
            
        # Get keypoints for the first detected person (highest confidence)
        keypoints_data = results[0].keypoints
        
        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return None
        
        # Get the first person's keypoints
        kpts_xy = keypoints_data.xy[0].cpu().numpy()  # Shape: (17, 2)
        kpts_conf = keypoints_data.conf[0].cpu().numpy() if keypoints_data.conf is not None else np.ones(17)
        
        # Build keypoints dictionary
        keypoints = {}
        for name, idx in KEYPOINT_INDICES.items():
            x, y = kpts_xy[idx]
            conf = kpts_conf[idx]
            keypoints[name] = (float(x), float(y), float(conf))
        
        # Store raw keypoints for visualization
        raw_keypoints = np.concatenate([kpts_xy, kpts_conf.reshape(-1, 1)], axis=1)
        
        return FrameKeypoints(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            keypoints=keypoints,
            raw_keypoints=raw_keypoints
        )
    
    def get_results_with_visualization(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5
    ):
        """
        Get detection results with visualization data for rendering.
        
        Returns the raw YOLO results object for custom visualization.
        """
        return self.model(frame, verbose=False, device=self.device, conf=conf_threshold)

