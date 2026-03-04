"""
Keypoint configuration for different pose estimation models.

Supported models:
1. YOLO COCO - Standard YOLOv8-pose with 17 COCO keypoints
2. YOLO Lower Body - Fine-tuned model with 10 lower-body keypoints including heel/toe

References:
- YOLO Lower Body: https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DetectorBackend(Enum):
    """Available pose estimation backends (YOLO-based only)."""
    YOLO_COCO = "yolo_coco"           # Standard YOLO with COCO keypoints (17 points)
    YOLOV8 = "yolov8"                  # Alias for YOLO_COCO
    YOLO_LOWER_BODY = "yolo_lower"    # Fine-tuned YOLO for lower body (10 points)


@dataclass
class KeypointIndices:
    """Keypoint indices for a specific model."""
    # Lower body keypoints (required for gait analysis)
    left_hip: int
    right_hip: int
    left_knee: int
    right_knee: int
    left_ankle: int
    right_ankle: int
    
    # Heel and toe keypoints (optional - dramatically improves gait analysis)
    left_heel: Optional[int] = None
    right_heel: Optional[int] = None
    left_toe: Optional[int] = None      # Big toe or foot front
    right_toe: Optional[int] = None     # Big toe or foot front
    left_small_toe: Optional[int] = None
    right_small_toe: Optional[int] = None
    
    # Total keypoints in the model
    total_keypoints: int = 17
    
    def has_heel_toe(self) -> bool:
        """Check if this model has heel and toe keypoints."""
        return (self.left_heel is not None and 
                self.right_heel is not None and
                self.left_toe is not None and 
                self.right_toe is not None)


# Standard YOLO COCO keypoints (17 points)
# No heel or toe - only ankle
YOLO_COCO_KEYPOINTS = KeypointIndices(
    left_hip=11,
    right_hip=12,
    left_knee=13,
    right_knee=14,
    left_ankle=15,
    right_ankle=16,
    # No heel/toe in COCO
    left_heel=None,
    right_heel=None,
    left_toe=None,
    right_toe=None,
    total_keypoints=17
)


# Fine-tuned YOLO Lower Body keypoints (10 points)
# Keypoint order: L_Hip, R_Hip, L_Knee, R_Knee, L_Ankle, R_Ankle, L_Heel, R_Heel, L_Foot, R_Foot
YOLO_LOWER_BODY_KEYPOINTS = KeypointIndices(
    left_hip=0,
    right_hip=1,
    left_knee=2,
    right_knee=3,
    left_ankle=4,
    right_ankle=5,
    left_heel=6,
    right_heel=7,
    left_toe=8,      # "Foot" in the model = front of foot/toes
    right_toe=9,
    total_keypoints=10
)


def get_keypoint_config(backend: DetectorBackend) -> KeypointIndices:
    """Get keypoint configuration for a specific backend."""
    configs = {
        DetectorBackend.YOLO_COCO: YOLO_COCO_KEYPOINTS,
        DetectorBackend.YOLOV8: YOLO_COCO_KEYPOINTS,  # Alias
        DetectorBackend.YOLO_LOWER_BODY: YOLO_LOWER_BODY_KEYPOINTS,
    }
    return configs[backend]


# Keypoint names for each model (for visualization/debugging)
YOLO_COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

YOLO_LOWER_BODY_NAMES = [
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot", "right_foot"
]

# COCO-WholeBody style foot keypoints (6 points) for production gait pipeline
# L_HEEL, L_BIG_TOE, L_SMALL_TOE, R_HEEL, R_BIG_TOE, R_SMALL_TOE
FOOT_6_KEYPOINT_NAMES = [
    "L_HEEL", "L_BIG_TOE", "L_SMALL_TOE",
    "R_HEEL", "R_BIG_TOE", "R_SMALL_TOE",
]
