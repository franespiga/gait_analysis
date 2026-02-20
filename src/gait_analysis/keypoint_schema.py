"""
Keypoint Schema Mapping Layer.

Maps different pose model keypoint formats to a canonical gait analysis schema.
This allows any pose backend to be used for gait analysis, regardless of its
native keypoint format.

Canonical Gait Schema:
- L/R Hip
- L/R Knee  
- L/R Ankle
- L/R Heel (optional)
- L/R Toe / Big Toe (optional)
- L/R Small Toe (optional, for models with detailed feet)

Supported source schemas:
- COCO-17 (standard YOLO)
- COCO Lower Body 10 (fine-tuned YOLO)
- OpenPose Body_25
- HALPE-26 (AlphaPose body)
- HALPE-136 (AlphaPose whole-body)
- COCO-WholeBody 133 (MMPose/SDPose)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np

from .pose_backend import PoseResult, PersonPose


class SkeletonType(Enum):
    """Supported skeleton/keypoint formats."""
    COCO_17 = "coco17"                    # Standard COCO (YOLO default)
    COCO_LOWER_10 = "coco_lower10"        # Lower body model
    BODY_25 = "body25"                    # OpenPose Body_25
    HALPE_26 = "halpe26"                  # AlphaPose body
    HALPE_136 = "halpe136"                # AlphaPose whole-body
    WHOLEBODY_133 = "wholebody133"        # COCO-WholeBody (MMPose/SDPose)
    POCKETPOSE = "pocketpose"             # PocketPose format


@dataclass
class CanonicalGaitKeypoints:
    """
    Canonical gait analysis keypoints.
    
    All coordinates are (x, y) tuples or None if not detected/available.
    Confidence scores range from 0 to 1.
    """
    # Core lower body (always available in any pose model)
    left_hip: Optional[tuple] = None
    right_hip: Optional[tuple] = None
    left_knee: Optional[tuple] = None
    right_knee: Optional[tuple] = None
    left_ankle: Optional[tuple] = None
    right_ankle: Optional[tuple] = None
    
    # Foot keypoints (optional - dramatically improve gait analysis)
    left_heel: Optional[tuple] = None
    right_heel: Optional[tuple] = None
    left_toe: Optional[tuple] = None       # Big toe or foot front
    right_toe: Optional[tuple] = None      # Big toe or foot front
    left_small_toe: Optional[tuple] = None # Small toe (when available)
    right_small_toe: Optional[tuple] = None
    
    # Additional foot points (for detailed models like HALPE-136)
    left_foot_index: Optional[tuple] = None   # Second toe
    right_foot_index: Optional[tuple] = None
    
    # Confidence scores
    confidences: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    source_skeleton: str = "unknown"
    frame_number: int = 0
    timestamp_ms: float = 0.0
    bbox: Optional[np.ndarray] = None
    track_id: Optional[int] = None
    
    def has_feet(self) -> bool:
        """Check if detailed foot keypoints are available."""
        return any([
            self.left_heel, self.right_heel,
            self.left_toe, self.right_toe
        ])
    
    def has_detailed_feet(self) -> bool:
        """Check if small toe / additional foot points are available."""
        return any([
            self.left_small_toe, self.right_small_toe,
            self.left_foot_index, self.right_foot_index
        ])
    
    def get_available_keypoints(self) -> List[str]:
        """Return list of available (non-None) keypoint names."""
        available = []
        keypoint_attrs = [
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_toe', 'right_toe', 'left_small_toe', 'right_small_toe',
            'left_foot_index', 'right_foot_index'
        ]
        for attr in keypoint_attrs:
            if getattr(self, attr) is not None:
                available.append(attr)
        return available
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "left_hip": self.left_hip,
            "right_hip": self.right_hip,
            "left_knee": self.left_knee,
            "right_knee": self.right_knee,
            "left_ankle": self.left_ankle,
            "right_ankle": self.right_ankle,
            "left_heel": self.left_heel,
            "right_heel": self.right_heel,
            "left_toe": self.left_toe,
            "right_toe": self.right_toe,
            "left_small_toe": self.left_small_toe,
            "right_small_toe": self.right_small_toe,
            "has_feet": self.has_feet(),
            "has_detailed_feet": self.has_detailed_feet(),
            "source_skeleton": self.source_skeleton,
            "available_keypoints": self.get_available_keypoints()
        }


# ============================================================================
# Keypoint Index Mappings for Each Skeleton Type
# ============================================================================

# COCO-17 (Standard YOLO/MMPose)
# No feet keypoints - only ankle
COCO_17_INDICES = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    # No heel/toe in COCO-17
}

# COCO Lower Body 10 (Fine-tuned YOLO)
# Order: L_Hip, R_Hip, L_Knee, R_Knee, L_Ankle, R_Ankle, L_Heel, R_Heel, L_Foot, R_Foot
COCO_LOWER_10_INDICES = {
    "left_hip": 0,
    "right_hip": 1,
    "left_knee": 2,
    "right_knee": 3,
    "left_ankle": 4,
    "right_ankle": 5,
    "left_heel": 6,
    "right_heel": 7,
    "left_toe": 8,
    "right_toe": 9,
}

# OpenPose Body_25
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
BODY_25_INDICES = {
    "left_hip": 12,
    "right_hip": 9,
    "left_knee": 13,
    "right_knee": 10,
    "left_ankle": 14,
    "right_ankle": 11,
    "left_heel": 21,
    "right_heel": 24,
    "left_toe": 19,       # LBigToe
    "right_toe": 22,      # RBigToe
    "left_small_toe": 20, # LSmallToe
    "right_small_toe": 23, # RSmallToe
}

# HALPE-26 (AlphaPose body format)
# https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
HALPE_26_INDICES = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    "left_heel": 19,
    "right_heel": 22,
    "left_toe": 20,       # LBigToe
    "right_toe": 23,      # RBigToe
    "left_small_toe": 21,
    "right_small_toe": 24,
}

# HALPE-136 Whole-Body (AlphaPose)
# Body: 0-25, Face: 26-93, Left Hand: 94-114, Right Hand: 115-135
# Feet are same indices as HALPE-26
HALPE_136_INDICES = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    "left_heel": 19,
    "right_heel": 22,
    "left_toe": 20,
    "right_toe": 23,
    "left_small_toe": 21,
    "right_small_toe": 24,
}

# COCO-WholeBody 133 (MMPose, SDPose)
# Body: 0-16, Foot: 17-22, Face: 23-90, Left Hand: 91-111, Right Hand: 112-132
# Foot keypoints: 17=L_BigToe, 18=L_SmallToe, 19=L_Heel, 20=R_BigToe, 21=R_SmallToe, 22=R_Heel
WHOLEBODY_133_INDICES = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    "left_toe": 17,        # L_BigToe
    "left_small_toe": 18,  # L_SmallToe
    "left_heel": 19,
    "right_toe": 20,       # R_BigToe
    "right_small_toe": 21, # R_SmallToe
    "right_heel": 22,
}

# PocketPose (similar to COCO-WholeBody but may vary by model)
# Using COCO-WholeBody mapping as default
POCKETPOSE_INDICES = WHOLEBODY_133_INDICES.copy()


# Mapping from skeleton type to indices
SKELETON_MAPPINGS: Dict[SkeletonType, Dict[str, int]] = {
    SkeletonType.COCO_17: COCO_17_INDICES,
    SkeletonType.COCO_LOWER_10: COCO_LOWER_10_INDICES,
    SkeletonType.BODY_25: BODY_25_INDICES,
    SkeletonType.HALPE_26: HALPE_26_INDICES,
    SkeletonType.HALPE_136: HALPE_136_INDICES,
    SkeletonType.WHOLEBODY_133: WHOLEBODY_133_INDICES,
    SkeletonType.POCKETPOSE: POCKETPOSE_INDICES,
}

# Also allow string-based lookup
SKELETON_MAPPINGS_BY_NAME: Dict[str, Dict[str, int]] = {
    "coco17": COCO_17_INDICES,
    "coco_lower10": COCO_LOWER_10_INDICES,
    "body25": BODY_25_INDICES,
    "halpe26": HALPE_26_INDICES,
    "halpe136": HALPE_136_INDICES,
    "wholebody133": WHOLEBODY_133_INDICES,
    "sdpose_wholebody": WHOLEBODY_133_INDICES,  # Same as wholebody133
    "pocketpose": POCKETPOSE_INDICES,
}


class KeypointSchemaMapper:
    """
    Maps keypoints from any supported skeleton format to canonical gait keypoints.
    """
    
    def __init__(self, source_skeleton: str, conf_threshold: float = 0.3):
        """
        Initialize the mapper.
        
        Args:
            source_skeleton: Name of the source skeleton format
            conf_threshold: Minimum confidence to consider keypoint valid
        """
        self.source_skeleton = source_skeleton.lower()
        self.conf_threshold = conf_threshold
        
        if self.source_skeleton not in SKELETON_MAPPINGS_BY_NAME:
            raise ValueError(
                f"Unknown skeleton type: {source_skeleton}. "
                f"Supported: {list(SKELETON_MAPPINGS_BY_NAME.keys())}"
            )
        
        self.indices = SKELETON_MAPPINGS_BY_NAME[self.source_skeleton]
    
    def map_pose_result(
        self,
        result: PoseResult,
        person_idx: int = 0,
        frame_number: int = 0,
        timestamp_ms: float = 0.0
    ) -> Optional[CanonicalGaitKeypoints]:
        """
        Map a PoseResult to canonical gait keypoints.
        
        Args:
            result: PoseResult from any backend
            person_idx: Which person to extract (default: first/most confident)
            frame_number: Current frame number
            timestamp_ms: Current timestamp in ms
            
        Returns:
            CanonicalGaitKeypoints or None if no valid person detected
        """
        person = result.get_person(person_idx)
        if person is None:
            return None
        
        return self.map_person_pose(person, frame_number, timestamp_ms)
    
    def map_person_pose(
        self,
        person: PersonPose,
        frame_number: int = 0,
        timestamp_ms: float = 0.0
    ) -> CanonicalGaitKeypoints:
        """
        Map a single PersonPose to canonical gait keypoints.
        """
        def get_kpt(name: str) -> Optional[tuple]:
            """Extract keypoint if in mapping and above threshold."""
            if name not in self.indices:
                return None
            idx = self.indices[name]
            return person.get_keypoint(idx, self.conf_threshold)
        
        def get_conf(name: str) -> float:
            """Get confidence score for a keypoint."""
            if name not in self.indices:
                return 0.0
            idx = self.indices[name]
            _, _, conf = person.get_keypoint_with_conf(idx)
            return conf
        
        # Build confidence dict
        confidences = {
            "left_hip": get_conf("left_hip"),
            "right_hip": get_conf("right_hip"),
            "left_knee": get_conf("left_knee"),
            "right_knee": get_conf("right_knee"),
            "left_ankle": get_conf("left_ankle"),
            "right_ankle": get_conf("right_ankle"),
            "left_heel": get_conf("left_heel"),
            "right_heel": get_conf("right_heel"),
            "left_toe": get_conf("left_toe"),
            "right_toe": get_conf("right_toe"),
        }
        
        return CanonicalGaitKeypoints(
            # Core joints
            left_hip=get_kpt("left_hip"),
            right_hip=get_kpt("right_hip"),
            left_knee=get_kpt("left_knee"),
            right_knee=get_kpt("right_knee"),
            left_ankle=get_kpt("left_ankle"),
            right_ankle=get_kpt("right_ankle"),
            # Feet
            left_heel=get_kpt("left_heel"),
            right_heel=get_kpt("right_heel"),
            left_toe=get_kpt("left_toe"),
            right_toe=get_kpt("right_toe"),
            left_small_toe=get_kpt("left_small_toe"),
            right_small_toe=get_kpt("right_small_toe"),
            left_foot_index=get_kpt("left_foot_index"),
            right_foot_index=get_kpt("right_foot_index"),
            # Metadata
            confidences=confidences,
            source_skeleton=self.source_skeleton,
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            bbox=person.bbox,
            track_id=person.track_id
        )
    
    def map_raw_keypoints(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        frame_number: int = 0,
        timestamp_ms: float = 0.0,
        bbox: Optional[np.ndarray] = None,
        track_id: Optional[int] = None
    ) -> CanonicalGaitKeypoints:
        """
        Map raw keypoint arrays to canonical gait keypoints.
        
        Args:
            keypoints: Array of shape (K, 2) - K keypoints, (x, y)
            scores: Array of shape (K,) - confidence scores
            frame_number: Frame number
            timestamp_ms: Timestamp
            bbox: Optional bounding box
            track_id: Optional tracking ID
            
        Returns:
            CanonicalGaitKeypoints
        """
        person = PersonPose(
            keypoints=keypoints,
            scores=scores,
            bbox=bbox,
            track_id=track_id,
            skeleton_name=self.source_skeleton
        )
        return self.map_person_pose(person, frame_number, timestamp_ms)
    
    @property
    def has_feet_mapping(self) -> bool:
        """Check if this skeleton has foot keypoint mappings."""
        return "left_heel" in self.indices or "left_toe" in self.indices
    
    def get_expected_keypoints(self) -> List[str]:
        """Get list of keypoint names available in this skeleton."""
        return list(self.indices.keys())


def create_schema_mapper(skeleton_name: str, conf_threshold: float = 0.3) -> KeypointSchemaMapper:
    """
    Factory function to create a KeypointSchemaMapper.
    
    Args:
        skeleton_name: Name of the source skeleton format
        conf_threshold: Confidence threshold for valid keypoints
        
    Returns:
        Configured KeypointSchemaMapper
    """
    return KeypointSchemaMapper(skeleton_name, conf_threshold)


def get_supported_skeletons() -> List[str]:
    """Get list of supported skeleton format names."""
    return list(SKELETON_MAPPINGS_BY_NAME.keys())


def skeleton_has_feet(skeleton_name: str) -> bool:
    """Check if a skeleton format includes foot keypoints."""
    if skeleton_name.lower() not in SKELETON_MAPPINGS_BY_NAME:
        return False
    indices = SKELETON_MAPPINGS_BY_NAME[skeleton_name.lower()]
    return "left_heel" in indices or "left_toe" in indices
