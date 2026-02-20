"""
Keypoint configuration for different pose estimation models.

Supported models:
1. YOLO COCO - Standard YOLOv8-pose with 17 COCO keypoints
2. YOLO Lower Body - Fine-tuned model with 10 lower-body keypoints including heel/toe
3. OpenPose - CMU OpenPose with 25 body keypoints + foot keypoints

References:
- YOLO Lower Body: https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints
- OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose
- OpenPose keypoints: https://chingswy.github.io/easymocap-public-doc/database/2_keypoints.html
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DetectorBackend(Enum):
    """Available pose estimation backends."""
    YOLO_COCO = "yolo_coco"           # Standard YOLO with COCO keypoints (17 points)
    YOLOV8 = "yolov8"                  # Alias for YOLO_COCO
    YOLO_LOWER_BODY = "yolo_lower"    # Fine-tuned YOLO for lower body (10 points)
    OPENPOSE = "openpose"              # OpenPose (25 body + foot keypoints)
    POCKETPOSE = "pocketpose"          # PocketPose (whole-body, ONNX)
    SDPOSE = "sdpose"                  # SDPose whole-body from HuggingFace
    ALPHAPOSE = "alphapose"            # AlphaPose with HALPE-136 keypoints
    ALPHAPOSE_BODY = "alphapose_body"  # AlphaPose with HALPE-26 keypoints


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
    left_small_toe: Optional[int] = None   # Small toe (OpenPose only)
    right_small_toe: Optional[int] = None  # Small toe (OpenPose only)
    
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


# OpenPose Body_25 + Foot keypoints
# Body keypoints: https://chingswy.github.io/easymocap-public-doc/database/2_keypoints.html
# Hips: 9 (right), 12 (left)
# Knees: 10 (right), 13 (left)
# Ankles: 11 (right), 14 (left)
# Heels: 21 (left), 24 (right)
# Big toe: 19 (left), 22 (right)
# Small toe: 20 (left), 23 (right)
OPENPOSE_KEYPOINTS = KeypointIndices(
    left_hip=12,
    right_hip=9,
    left_knee=13,
    right_knee=10,
    left_ankle=14,
    right_ankle=11,
    left_heel=21,
    right_heel=24,
    left_toe=19,       # Big toe
    right_toe=22,      # Big toe
    left_small_toe=20,
    right_small_toe=23,
    total_keypoints=25  # Body_25 format (foot keypoints may be separate)
)


def get_keypoint_config(backend: DetectorBackend) -> KeypointIndices:
    """Get keypoint configuration for a specific backend."""
    configs = {
        DetectorBackend.YOLO_COCO: YOLO_COCO_KEYPOINTS,
        DetectorBackend.YOLOV8: YOLO_COCO_KEYPOINTS,  # Alias
        DetectorBackend.YOLO_LOWER_BODY: YOLO_LOWER_BODY_KEYPOINTS,
        DetectorBackend.OPENPOSE: OPENPOSE_KEYPOINTS,
        DetectorBackend.POCKETPOSE: WHOLEBODY_133_KEYPOINTS,
        DetectorBackend.SDPOSE: WHOLEBODY_133_KEYPOINTS,
        DetectorBackend.ALPHAPOSE: HALPE_136_KEYPOINTS,
        DetectorBackend.ALPHAPOSE_BODY: HALPE_26_KEYPOINTS,
    }
    return configs[backend]


# HALPE-26 keypoints (AlphaPose body format)
HALPE_26_KEYPOINTS = KeypointIndices(
    left_hip=11,
    right_hip=12,
    left_knee=13,
    right_knee=14,
    left_ankle=15,
    right_ankle=16,
    left_heel=19,
    right_heel=22,
    left_toe=20,       # Big toe
    right_toe=23,      # Big toe
    left_small_toe=21,
    right_small_toe=24,
    total_keypoints=26
)


# HALPE-136 whole-body keypoints (AlphaPose)
HALPE_136_KEYPOINTS = KeypointIndices(
    left_hip=11,
    right_hip=12,
    left_knee=13,
    right_knee=14,
    left_ankle=15,
    right_ankle=16,
    left_heel=19,
    right_heel=22,
    left_toe=20,
    right_toe=23,
    left_small_toe=21,
    right_small_toe=24,
    total_keypoints=136
)


# COCO-WholeBody 133 keypoints (MMPose/SDPose/PocketPose)
# Foot keypoints: 17=L_BigToe, 18=L_SmallToe, 19=L_Heel, 20=R_BigToe, 21=R_SmallToe, 22=R_Heel
WHOLEBODY_133_KEYPOINTS = KeypointIndices(
    left_hip=11,
    right_hip=12,
    left_knee=13,
    right_knee=14,
    left_ankle=15,
    right_ankle=16,
    left_toe=17,        # L_BigToe
    left_small_toe=18,  # L_SmallToe
    left_heel=19,
    right_toe=20,       # R_BigToe
    right_small_toe=21, # R_SmallToe
    right_heel=22,
    total_keypoints=133
)


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

OPENPOSE_BODY25_NAMES = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "mid_hip",
    "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "right_eye", "left_eye",
    "right_ear", "left_ear", "left_big_toe", "left_small_toe",
    "left_heel", "right_big_toe", "right_small_toe", "right_heel"
]

