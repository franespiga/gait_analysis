"""
YOLO-based pose estimation detectors.

Supports:
1. Standard YOLOv8-pose with COCO keypoints (17 points)
2. Fine-tuned YOLOv8-pose for lower body (10 points with heel/toe)

References:
- YOLOv8: https://docs.ultralytics.com/tasks/pose/
- Lower Body Model: https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints
"""

from typing import Optional
import numpy as np
from ultralytics import YOLO

from .base_detector import BaseDetector, GaitKeypoints
from .keypoint_config import (
    DetectorBackend, 
    KeypointIndices,
    YOLO_COCO_KEYPOINTS,
    YOLO_LOWER_BODY_KEYPOINTS,
    get_keypoint_config
)




class YOLODetector(BaseDetector):
    """
    YOLO-based pose detector supporting both COCO and lower-body models.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        backend: DetectorBackend = DetectorBackend.YOLO_COCO,
        device: str = "cpu"
    ):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
                - For COCO: "yolov8n-pose.pt", "yolov8s-pose.pt", etc.
                - For Lower Body: path to fine-tuned model (default: models/yolo_lower/best.pt)
            backend: Which keypoint configuration to use
            device: Inference device ('auto', 'cpu', 'cuda', etc.)
        """
        self._backend = backend
        self._keypoint_config = get_keypoint_config(backend)
        self.model = YOLO(model_path)
        self.device = device
        self.model_path = model_path
    
    @property
    def backend(self) -> DetectorBackend:
        return self._backend
    
    @property
    def keypoint_config(self) -> KeypointIndices:
        return self._keypoint_config
    
    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_ms: float,
        conf_threshold: float = 0.5
    ) -> Optional[GaitKeypoints]:
        """
        Detect keypoints in a single frame.
        """
        results = self.model(frame, verbose=False, device=self.device)
        
        if len(results) == 0 or results[0].keypoints is None:
            return None
        
        keypoints_data = results[0].keypoints
        
        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return None
        
        # Get the first person's keypoints (highest confidence)
        kpts_xy = keypoints_data.xy[0].cpu().numpy()
        kpts_conf = (keypoints_data.conf[0].cpu().numpy() 
                    if keypoints_data.conf is not None 
                    else np.ones(len(kpts_xy)))
        
        config = self._keypoint_config
        
        # Extract all relevant keypoints
        gait_kpts = GaitKeypoints(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            # Hips
            left_hip=self._extract_point(kpts_xy, kpts_conf, config.left_hip, conf_threshold),
            right_hip=self._extract_point(kpts_xy, kpts_conf, config.right_hip, conf_threshold),
            # Knees
            left_knee=self._extract_point(kpts_xy, kpts_conf, config.left_knee, conf_threshold),
            right_knee=self._extract_point(kpts_xy, kpts_conf, config.right_knee, conf_threshold),
            # Ankles
            left_ankle=self._extract_point(kpts_xy, kpts_conf, config.left_ankle, conf_threshold),
            right_ankle=self._extract_point(kpts_xy, kpts_conf, config.right_ankle, conf_threshold),
            # Heels (if available)
            left_heel=self._extract_point(kpts_xy, kpts_conf, config.left_heel, conf_threshold),
            right_heel=self._extract_point(kpts_xy, kpts_conf, config.right_heel, conf_threshold),
            # Toes (if available)
            left_toe=self._extract_point(kpts_xy, kpts_conf, config.left_toe, conf_threshold),
            right_toe=self._extract_point(kpts_xy, kpts_conf, config.right_toe, conf_threshold),
            # Raw keypoints for visualization
            raw_keypoints=np.concatenate([kpts_xy, kpts_conf.reshape(-1, 1)], axis=1),
            # Confidence scores
            confidences=self._extract_confidences(kpts_conf, config)
        )
        
        return gait_kpts
    
    def get_visualization_results(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5
    ):
        """Get YOLO results for visualization."""
        return self.model(frame, verbose=False, device=self.device, conf=conf_threshold)
    
    def _extract_confidences(
        self,
        kpts_conf: np.ndarray,
        config: KeypointIndices
    ) -> dict[str, float]:
        """Extract confidence scores for key joints."""
        def safe_get(idx: Optional[int]) -> float:
            if idx is None or idx >= len(kpts_conf):
                return 0.0
            return float(kpts_conf[idx])
        
        return {
            "left_hip": safe_get(config.left_hip),
            "right_hip": safe_get(config.right_hip),
            "left_knee": safe_get(config.left_knee),
            "right_knee": safe_get(config.right_knee),
            "left_ankle": safe_get(config.left_ankle),
            "right_ankle": safe_get(config.right_ankle),
            "left_heel": safe_get(config.left_heel),
            "right_heel": safe_get(config.right_heel),
            "left_toe": safe_get(config.left_toe),
            "right_toe": safe_get(config.right_toe),
        }


class YOLOCocoDetector(YOLODetector):
    """
    Standard YOLO detector with COCO keypoints (17 points).
    
    This is the default detector using pre-trained YOLOv8-pose models.
    Does NOT have heel or toe keypoints - only ankle.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        device: str = "cpu"
    ):
        """
        Initialize with a standard YOLO pose model.
        
        Args:
            model_name: YOLO model name (yolov8n-pose.pt, yolov8s-pose.pt, etc.)
            device: Inference device
        """
        super().__init__(
            model_path=model_name,
            backend=DetectorBackend.YOLO_COCO,
            device=device
        )


class YOLOLowerBodyDetector(YOLODetector):
    """
    Fine-tuned YOLO detector for lower body with 10 keypoints.
    
    Keypoints: L&R Hip, L&R Knee, L&R Ankle, L&R Heel, L&R Foot (toe)
    
    This model provides heel and toe keypoints which dramatically improve
    gait analysis accuracy for detecting heel strike vs toe walking.
    
    Reference: https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu"
    ):
        """
        Initialize with the fine-tuned lower body model.
        
        Args:
            model_path: Path to the fine-tuned model (default: models/yolo_lower/best.pt)
            device: Inference device
        """
        super().__init__(
            model_path=model_path,
            backend=DetectorBackend.YOLO_LOWER_BODY,
            device=device
        )


def create_yolo_detector(
    model_path: str = "yolov8n-pose.pt",
    backend: DetectorBackend = DetectorBackend.YOLO_COCO,
    device: str = "cpu"
) -> YOLODetector:
    """
    Factory function to create the appropriate YOLO detector.
    
    Args:
        model_path: Path to model weights
        backend: Keypoint configuration to use
        device: Inference device
        
    Returns:
        Configured YOLODetector instance
    """
    if backend == DetectorBackend.YOLO_LOWER_BODY:
        return YOLOLowerBodyDetector(model_path=model_path, device=device)
    else:
        return YOLOCocoDetector(model_name=model_path, device=device)

