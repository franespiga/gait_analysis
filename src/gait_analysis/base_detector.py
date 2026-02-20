"""
Base detector interface and common data structures for pose estimation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np

from .keypoint_config import KeypointIndices, DetectorBackend


@dataclass
class GaitKeypoints:
    """
    Standardized keypoint data for gait analysis.
    
    All positions are (x, y) tuples or None if not detected with sufficient confidence.
    This provides a unified interface regardless of which detector backend is used.
    """
    frame_number: int
    timestamp_ms: float
    
    # Hip positions
    left_hip: Optional[tuple[float, float]] = None
    right_hip: Optional[tuple[float, float]] = None
    
    # Knee positions
    left_knee: Optional[tuple[float, float]] = None
    right_knee: Optional[tuple[float, float]] = None
    
    # Ankle positions
    left_ankle: Optional[tuple[float, float]] = None
    right_ankle: Optional[tuple[float, float]] = None
    
    # Heel positions (if available - greatly improves gait analysis)
    left_heel: Optional[tuple[float, float]] = None
    right_heel: Optional[tuple[float, float]] = None
    
    # Toe/foot front positions (if available)
    left_toe: Optional[tuple[float, float]] = None
    right_toe: Optional[tuple[float, float]] = None
    
    # Raw keypoints array for visualization (model-specific format)
    raw_keypoints: Optional[np.ndarray] = None
    
    # Confidence scores for key joints
    confidences: Optional[dict[str, float]] = None
    
    def has_heel_toe(self) -> bool:
        """Check if heel and toe keypoints are available."""
        return (self.left_heel is not None or self.right_heel is not None or
                self.left_toe is not None or self.right_toe is not None)
    
    def get_ankle_positions(self) -> dict[str, Optional[tuple[float, float]]]:
        """Get ankle positions (for compatibility with existing code)."""
        return {
            "left": self.left_ankle,
            "right": self.right_ankle
        }
    
    def get_knee_positions(self) -> dict[str, Optional[tuple[float, float]]]:
        """Get knee positions."""
        return {
            "left": self.left_knee,
            "right": self.right_knee
        }
    
    def get_hip_positions(self) -> dict[str, Optional[tuple[float, float]]]:
        """Get hip positions."""
        return {
            "left": self.left_hip,
            "right": self.right_hip
        }
    
    def get_heel_positions(self) -> dict[str, Optional[tuple[float, float]]]:
        """Get heel positions (may be None for models without heel keypoints)."""
        return {
            "left": self.left_heel,
            "right": self.right_heel
        }
    
    def get_toe_positions(self) -> dict[str, Optional[tuple[float, float]]]:
        """Get toe/foot front positions (may be None for models without toe keypoints)."""
        return {
            "left": self.left_toe,
            "right": self.right_toe
        }
    
    def get_body_center(self) -> Optional[tuple[float, float]]:
        """Get the body center (midpoint between hips)."""
        if self.left_hip is not None and self.right_hip is not None:
            center_x = (self.left_hip[0] + self.right_hip[0]) / 2
            center_y = (self.left_hip[1] + self.right_hip[1]) / 2
            return (center_x, center_y)
        return None


class BaseDetector(ABC):
    """
    Abstract base class for pose estimation detectors.
    
    All detector implementations should inherit from this class and implement
    the required methods to provide a consistent interface.
    """
    
    @property
    @abstractmethod
    def backend(self) -> DetectorBackend:
        """Return the detector backend type."""
        pass
    
    @property
    @abstractmethod
    def keypoint_config(self) -> KeypointIndices:
        """Return the keypoint configuration for this detector."""
        pass
    
    @property
    def has_heel_toe(self) -> bool:
        """Check if this detector provides heel and toe keypoints."""
        return self.keypoint_config.has_heel_toe()
    
    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_ms: float,
        conf_threshold: float = 0.5
    ) -> Optional[GaitKeypoints]:
        """
        Detect keypoints in a single frame.
        
        Args:
            frame: BGR image as numpy array
            frame_number: Frame index in the video
            timestamp_ms: Timestamp in milliseconds
            conf_threshold: Minimum confidence threshold
            
        Returns:
            GaitKeypoints object or None if no person detected
        """
        pass
    
    @abstractmethod
    def get_visualization_results(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5
    ):
        """
        Get detection results suitable for visualization.
        
        Returns model-specific results that can be used for drawing.
        """
        pass
    
    def _extract_point(
        self,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        index: Optional[int],
        conf_threshold: float = 0.5
    ) -> Optional[tuple[float, float]]:
        """
        Extract a single keypoint position if confidence is sufficient.
        
        Args:
            keypoints_xy: Array of (x, y) positions
            keypoints_conf: Array of confidence scores
            index: Keypoint index (None if not available in this model)
            conf_threshold: Minimum confidence threshold
            
        Returns:
            (x, y) tuple or None
        """
        if index is None:
            return None
        
        if index >= len(keypoints_xy) or index >= len(keypoints_conf):
            return None
        
        x, y = keypoints_xy[index]
        conf = keypoints_conf[index]
        
        if conf >= conf_threshold and x > 0 and y > 0:
            return (float(x), float(y))
        
        return None

