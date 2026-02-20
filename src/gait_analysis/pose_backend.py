"""
Unified Pose Backend Interface and PoseResult dataclass.

This module defines the canonical interface that all pose estimation
backends must implement. It provides:

1. PoseResult - Standardized output format for pose estimation
2. PoseBackend - Abstract protocol that all backends implement
3. Dependency checking utilities

All backends (YOLOv8, OpenPose, PocketPose, SDPose, AlphaPose) conform
to this interface for seamless interchangeability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Protocol, List, Dict, Any, runtime_checkable
import numpy as np


@dataclass
class PoseResult:
    """
    Standardized pose estimation result.
    
    This is the canonical output format for all pose backends.
    Contains keypoints, confidence scores, bounding boxes, and metadata.
    
    Attributes:
        keypoints: Array of shape (N, K, 2) - N persons, K keypoints, (x, y)
        scores: Array of shape (N, K) - confidence scores per keypoint
        bboxes: Array of shape (N, 4) - bounding boxes [x1, y1, x2, y2]
        skeleton_name: Name of the keypoint schema (e.g., "coco17", "halpe136")
        track_ids: Optional array of shape (N,) - tracking IDs for each person
        num_persons: Number of detected persons
        num_keypoints: Number of keypoints per person
        raw_output: Optional raw model output for debugging/visualization
    """
    keypoints: np.ndarray  # (N, K, 2)
    scores: np.ndarray     # (N, K)
    bboxes: np.ndarray     # (N, 4)
    skeleton_name: str
    track_ids: Optional[np.ndarray] = None  # (N,)
    num_persons: int = 0
    num_keypoints: int = 0
    raw_output: Optional[Any] = None
    
    def __post_init__(self):
        """Compute derived fields."""
        if self.keypoints is not None and len(self.keypoints) > 0:
            self.num_persons = self.keypoints.shape[0]
            self.num_keypoints = self.keypoints.shape[1]
    
    def get_person(self, idx: int = 0) -> Optional['PersonPose']:
        """Get pose for a specific person."""
        if idx >= self.num_persons:
            return None
        return PersonPose(
            keypoints=self.keypoints[idx],
            scores=self.scores[idx],
            bbox=self.bboxes[idx] if len(self.bboxes) > idx else None,
            track_id=self.track_ids[idx] if self.track_ids is not None else None,
            skeleton_name=self.skeleton_name
        )
    
    def filter_by_confidence(self, min_score: float = 0.3) -> 'PoseResult':
        """Return a new PoseResult with low-confidence keypoints zeroed."""
        filtered_kpts = self.keypoints.copy()
        mask = self.scores < min_score
        filtered_kpts[mask] = 0
        return PoseResult(
            keypoints=filtered_kpts,
            scores=self.scores,
            bboxes=self.bboxes,
            skeleton_name=self.skeleton_name,
            track_ids=self.track_ids,
            raw_output=self.raw_output
        )
    
    @classmethod
    def empty(cls, skeleton_name: str = "unknown") -> 'PoseResult':
        """Create an empty result with no detections."""
        return cls(
            keypoints=np.empty((0, 0, 2)),
            scores=np.empty((0, 0)),
            bboxes=np.empty((0, 4)),
            skeleton_name=skeleton_name
        )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "num_persons": self.num_persons,
            "num_keypoints": self.num_keypoints,
            "skeleton_name": self.skeleton_name,
            "keypoints_shape": list(self.keypoints.shape),
            "has_tracking": self.track_ids is not None
        }


@dataclass
class PersonPose:
    """Single person pose data."""
    keypoints: np.ndarray  # (K, 2)
    scores: np.ndarray     # (K,)
    bbox: Optional[np.ndarray] = None  # (4,)
    track_id: Optional[int] = None
    skeleton_name: str = "unknown"
    
    def get_keypoint(self, idx: int, min_score: float = 0.0) -> Optional[tuple]:
        """Get a specific keypoint if above threshold."""
        if idx >= len(self.keypoints) or idx >= len(self.scores):
            return None
        if self.scores[idx] < min_score:
            return None
        x, y = self.keypoints[idx]
        if x <= 0 or y <= 0:
            return None
        return (float(x), float(y))
    
    def get_keypoint_with_conf(self, idx: int) -> tuple:
        """Get keypoint with confidence: (x, y, conf)."""
        if idx >= len(self.keypoints):
            return (0.0, 0.0, 0.0)
        x, y = self.keypoints[idx]
        conf = self.scores[idx] if idx < len(self.scores) else 0.0
        return (float(x), float(y), float(conf))


@runtime_checkable
class PoseBackend(Protocol):
    """
    Protocol defining the interface for pose estimation backends.
    
    All pose backends must implement this interface to be used
    interchangeably in the gait analysis pipeline.
    """
    
    @property
    def name(self) -> str:
        """Return the backend name."""
        ...
    
    @property
    def skeleton_name(self) -> str:
        """Return the skeleton/keypoint schema name."""
        ...
    
    def predict(self, image: np.ndarray) -> PoseResult:
        """
        Run pose estimation on an image.
        
        Args:
            image: BGR image as numpy array (H, W, 3)
            
        Returns:
            PoseResult with detected poses
        """
        ...
    
    def warmup(self, image_size: tuple = (640, 480)) -> None:
        """
        Warm up the model with a dummy inference.
        
        Args:
            image_size: (width, height) for dummy image
        """
        ...


class BasePoseBackend(ABC):
    """
    Abstract base class for pose backends.
    
    Provides common functionality and enforces the PoseBackend protocol.
    Subclasses must implement predict() and the name/skeleton properties.
    """
    
    def __init__(self, device: str = "auto", conf_threshold: float = 0.3):
        """
        Initialize the backend.
        
        Args:
            device: Inference device ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            conf_threshold: Default confidence threshold for keypoints
        """
        self.device = self._resolve_device(device)
        self.conf_threshold = conf_threshold
        self._is_initialized = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier name."""
        pass
    
    @property
    @abstractmethod
    def skeleton_name(self) -> str:
        """Keypoint schema name."""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> PoseResult:
        """Run pose estimation."""
        pass
    
    def warmup(self, image_size: tuple = (640, 480)) -> None:
        """Warm up with dummy inference."""
        dummy = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        self.predict(dummy)
        self._is_initialized = True
    
    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device != "auto":
            return device
        
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    
    def _bbox_from_keypoints(
        self, 
        keypoints: np.ndarray, 
        scores: np.ndarray,
        min_score: float = 0.1,
        padding: float = 0.1
    ) -> np.ndarray:
        """
        Compute bounding box from keypoints.
        
        Args:
            keypoints: (K, 2) array of keypoints
            scores: (K,) array of confidence scores
            min_score: Minimum confidence to include keypoint
            padding: Relative padding to add
            
        Returns:
            (4,) array [x1, y1, x2, y2]
        """
        valid_mask = (scores > min_score) & (keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)
        
        if not np.any(valid_mask):
            return np.array([0, 0, 0, 0], dtype=np.float32)
        
        valid_kpts = keypoints[valid_mask]
        x1, y1 = valid_kpts.min(axis=0)
        x2, y2 = valid_kpts.max(axis=0)
        
        # Add padding
        w, h = x2 - x1, y2 - y1
        x1 -= w * padding
        y1 -= h * padding
        x2 += w * padding
        y2 += h * padding
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)


class DependencyError(Exception):
    """Raised when a required dependency is missing."""
    
    def __init__(self, backend_name: str, package: str, install_cmd: str):
        self.backend_name = backend_name
        self.package = package
        self.install_cmd = install_cmd
        message = (
            f"Backend '{backend_name}' requires '{package}'. "
            f"Install with: {install_cmd}"
        )
        super().__init__(message)


def check_dependency(package: str, backend_name: str, extras_name: str) -> bool:
    """
    Check if a package is available, raise helpful error if not.
    
    Args:
        package: Python package name to import
        backend_name: Name of the backend requiring this dependency
        extras_name: Name of the pip extras to install
        
    Returns:
        True if available
        
    Raises:
        DependencyError: If package is not available
    """
    try:
        __import__(package)
        return True
    except ImportError:
        raise DependencyError(
            backend_name=backend_name,
            package=package,
            install_cmd=f"pip install gait-analysis[{extras_name}]"
        )


# Skeleton definitions for reference
SKELETON_INFO = {
    "coco17": {
        "num_keypoints": 17,
        "description": "COCO 17-keypoint format (standard YOLO)",
        "has_feet": False
    },
    "coco_lower10": {
        "num_keypoints": 10,
        "description": "Lower body 10 keypoints with heel/toe",
        "has_feet": True
    },
    "body25": {
        "num_keypoints": 25,
        "description": "OpenPose Body_25 format",
        "has_feet": True
    },
    "halpe26": {
        "num_keypoints": 26,
        "description": "HALPE 26-keypoint body format",
        "has_feet": True
    },
    "halpe136": {
        "num_keypoints": 136,
        "description": "HALPE 136-keypoint whole-body format",
        "has_feet": True
    },
    "wholebody133": {
        "num_keypoints": 133,
        "description": "COCO-WholeBody 133 keypoints",
        "has_feet": True
    },
    "sdpose_wholebody": {
        "num_keypoints": 133,
        "description": "SDPose whole-body keypoints",
        "has_feet": True
    }
}
