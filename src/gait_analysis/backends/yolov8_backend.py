"""
YOLOv8 Pose Backend Implementation.

Supports:
1. Standard YOLOv8-pose with COCO-17 keypoints
2. Fine-tuned YOLOv8 for lower body (10 keypoints with heel/toe)

References:
- Ultralytics YOLOv8: https://docs.ultralytics.com/tasks/pose/
- Lower Body Model: https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints
"""

from typing import Optional
import numpy as np

from ..pose_backend import BasePoseBackend, PoseResult, check_dependency


class YOLOv8PoseBackend(BasePoseBackend):
    """
    YOLOv8 Pose backend with COCO-17 keypoints.
    
    This is the standard YOLO pose model with 17 keypoints.
    Does NOT include heel or toe keypoints - only ankle.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: str = "auto",
        conf_threshold: float = 0.3
    ):
        """
        Initialize YOLOv8 pose backend.
        
        Args:
            model_path: Path to YOLO model or model name
                       (yolov8n-pose.pt, yolov8s-pose.pt, etc.)
            device: Inference device ('auto', 'cpu', 'cuda')
            conf_threshold: Confidence threshold for keypoints
        """
        super().__init__(device=device, conf_threshold=conf_threshold)
        
        # Check dependency
        check_dependency("ultralytics", "yolov8", "yolo")
        
        from ultralytics import YOLO
        self.model_path = model_path
        self.model = YOLO(model_path)
        self._skeleton = "coco17"
    
    @property
    def name(self) -> str:
        return "yolov8"
    
    @property
    def skeleton_name(self) -> str:
        return self._skeleton
    
    def predict(self, image: np.ndarray) -> PoseResult:
        """
        Run pose estimation on an image.
        
        Args:
            image: BGR image as numpy array (H, W, 3)
            
        Returns:
            PoseResult with detected poses
        """
        results = self.model(
            image,
            verbose=False,
            device=self.device,
            conf=self.conf_threshold
        )
        
        if len(results) == 0 or results[0].keypoints is None:
            return PoseResult.empty(self.skeleton_name)
        
        keypoints_data = results[0].keypoints
        
        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return PoseResult.empty(self.skeleton_name)
        
        # Extract keypoints and scores for all persons
        kpts_xy = keypoints_data.xy.cpu().numpy()  # (N, K, 2)
        kpts_conf = (keypoints_data.conf.cpu().numpy()
                    if keypoints_data.conf is not None
                    else np.ones(kpts_xy.shape[:2]))  # (N, K)
        
        # Extract bounding boxes
        bboxes = np.empty((len(kpts_xy), 4), dtype=np.float32)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            bboxes = results[0].boxes.xyxy.cpu().numpy()
        else:
            # Compute from keypoints
            for i in range(len(kpts_xy)):
                bboxes[i] = self._bbox_from_keypoints(kpts_xy[i], kpts_conf[i])
        
        return PoseResult(
            keypoints=kpts_xy,
            scores=kpts_conf,
            bboxes=bboxes,
            skeleton_name=self.skeleton_name,
            raw_output=results
        )
    
    def get_raw_results(self, image: np.ndarray):
        """Get raw YOLO results for visualization."""
        return self.model(
            image,
            verbose=False,
            device=self.device,
            conf=self.conf_threshold
        )


class YOLOv8LowerBodyBackend(YOLOv8PoseBackend):
    """
    Fine-tuned YOLOv8 for lower body with 10 keypoints.
    
    Keypoints: L&R Hip, L&R Knee, L&R Ankle, L&R Heel, L&R Foot (toe)
    
    This model provides heel and toe keypoints which dramatically improve
    gait analysis accuracy for detecting heel strike vs toe walking.
    
    IMPORTANT: This model was trained on cropped lower-body images.
    For full-body video frames, we use a two-stage pipeline:
    1. Detect persons using YOLOv8 object detector
    2. Crop to lower body region  
    3. Run pose estimation on the crop
    4. Transform keypoints back to original coordinates
    
    Reference: https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints
    """
    
    def __init__(
        self,
        model_path: str = "best.pt",
        device: str = "auto",
        conf_threshold: float = 0.25,  # Lower default for this model
        use_person_detector: bool = True,  # Enable two-stage pipeline
        lower_body_ratio: float = 0.6  # Crop bottom 60% for lower body
    ):
        """
        Initialize lower body YOLO backend.
        
        Args:
            model_path: Path to fine-tuned model weights (e.g., "best.pt")
            device: Inference device
            conf_threshold: Confidence threshold (default lower for this model)
            use_person_detector: If True, detect persons first then crop
            lower_body_ratio: Ratio of person bbox to keep (from bottom)
        """
        super().__init__(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold
        )
        self._skeleton = "coco_lower10"
        self._use_person_detector = use_person_detector
        self._lower_body_ratio = lower_body_ratio
        self._person_detector = None
        
        if use_person_detector:
            self._init_person_detector()
    
    def _init_person_detector(self):
        """Initialize YOLOv8 person detector for two-stage pipeline."""
        try:
            from ultralytics import YOLO
            self._person_detector = YOLO("yolov8n.pt")
            print("YOLO Lower: Person detector initialized for two-stage pipeline")
        except Exception as e:
            print(f"YOLO Lower: Person detector init failed: {e}")
            print("YOLO Lower: Falling back to direct detection (may have low accuracy)")
            self._use_person_detector = False
    
    @property
    def name(self) -> str:
        return "yolo_lower"
    
    def predict(self, image: np.ndarray) -> PoseResult:
        """
        Run pose estimation with optional person detection preprocessing.
        
        For best results with the lower-body model, we:
        1. Detect persons in the full frame
        2. Crop to lower body region of each person
        3. Run pose estimation on crops
        4. Transform keypoints back to original coordinates
        """
        if not self._use_person_detector or self._person_detector is None:
            # Direct detection (original behavior)
            return super().predict(image)
        
        # Two-stage pipeline: detect persons first
        return self._predict_with_person_detection(image)
    
    def _predict_with_person_detection(self, image: np.ndarray) -> PoseResult:
        """Two-stage pipeline: detect persons, crop, then run pose."""
        import cv2
        
        # Stage 1: Detect persons
        person_results = self._person_detector(
            image, 
            verbose=False, 
            device=self.device, 
            classes=[0],  # class 0 = person
            conf=0.3  # Lower threshold for person detection
        )
        
        if len(person_results) == 0 or person_results[0].boxes is None:
            # Fallback: try direct detection on full frame
            return self._predict_direct(image)
        
        boxes = person_results[0].boxes
        if len(boxes) == 0:
            return self._predict_direct(image)
        
        person_bboxes = boxes.xyxy.cpu().numpy()
        
        all_keypoints = []
        all_scores = []
        all_bboxes = []
        
        h, w = image.shape[:2]
        
        # Stage 2 & 3: Process each person with multiple crop strategies
        for bbox in person_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure valid bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            person_height = y2 - y1
            person_width = x2 - x1
            
            # Try multiple cropping strategies
            crop_result = None
            crop_x1, crop_y1 = 0, 0
            
            # Strategy 1: Lower body only (original)
            for ratio in [self._lower_body_ratio, 0.75, 0.9, 1.0]:
                lower_body_top = y1 + int(person_height * (1 - ratio))
                
                # Add horizontal padding
                pad_x = int(person_width * 0.2)
                c_x1 = max(0, x1 - pad_x)
                c_x2 = min(w, x2 + pad_x)
                c_y1 = max(0, lower_body_top - int(person_height * 0.15))
                c_y2 = min(h, y2 + int(person_height * 0.1))
                
                crop = image[c_y1:c_y2, c_x1:c_x2]
                
                if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                    continue
                
                result = self._predict_on_crop(crop)
                
                if result.num_persons > 0:
                    crop_result = result
                    crop_x1, crop_y1 = c_x1, c_y1
                    break
            
            if crop_result is not None and crop_result.num_persons > 0:
                # Transform keypoints back to original image coordinates
                kpts = crop_result.keypoints[0].copy()
                scores = crop_result.scores[0].copy()
                
                # Offset by crop position
                valid_mask = (kpts[:, 0] > 0) | (kpts[:, 1] > 0)
                kpts[valid_mask, 0] += crop_x1
                kpts[valid_mask, 1] += crop_y1
                
                all_keypoints.append(kpts)
                all_scores.append(scores)
                all_bboxes.append(bbox)
        
        if not all_keypoints:
            # Fallback to direct detection if cropping failed
            return self._predict_direct(image)
        
        return PoseResult(
            keypoints=np.array(all_keypoints),
            scores=np.array(all_scores),
            bboxes=np.array(all_bboxes),
            skeleton_name=self.skeleton_name
        )
    
    def _predict_direct(self, image: np.ndarray) -> PoseResult:
        """Direct detection on full frame (fallback)."""
        results = self.model(
            image,
            verbose=False,
            device=self.device,
            conf=self.conf_threshold * 0.5  # Lower threshold for fallback
        )
        
        if len(results) == 0 or results[0].keypoints is None:
            return PoseResult.empty(self.skeleton_name)
        
        keypoints_data = results[0].keypoints
        
        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return PoseResult.empty(self.skeleton_name)
        
        kpts_xy = keypoints_data.xy.cpu().numpy()
        kpts_conf = (keypoints_data.conf.cpu().numpy()
                    if keypoints_data.conf is not None
                    else np.ones(kpts_xy.shape[:2]))
        
        bboxes = np.empty((len(kpts_xy), 4), dtype=np.float32)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            bboxes = results[0].boxes.xyxy.cpu().numpy()
        else:
            for i in range(len(kpts_xy)):
                bboxes[i] = self._bbox_from_keypoints(kpts_xy[i], kpts_conf[i])
        
        return PoseResult(
            keypoints=kpts_xy,
            scores=kpts_conf,
            bboxes=bboxes,
            skeleton_name=self.skeleton_name,
            raw_output=results
        )
    
    def _predict_on_crop(self, crop: np.ndarray) -> PoseResult:
        """Run pose estimation on a cropped image."""
        results = self.model(
            crop,
            verbose=False,
            device=self.device,
            conf=self.conf_threshold
        )
        
        if len(results) == 0 or results[0].keypoints is None:
            return PoseResult.empty(self.skeleton_name)
        
        keypoints_data = results[0].keypoints
        
        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return PoseResult.empty(self.skeleton_name)
        
        kpts_xy = keypoints_data.xy.cpu().numpy()
        kpts_conf = (keypoints_data.conf.cpu().numpy()
                    if keypoints_data.conf is not None
                    else np.ones(kpts_xy.shape[:2]))
        
        bboxes = np.empty((len(kpts_xy), 4), dtype=np.float32)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            bboxes = results[0].boxes.xyxy.cpu().numpy()
        else:
            for i in range(len(kpts_xy)):
                bboxes[i] = self._bbox_from_keypoints(kpts_xy[i], kpts_conf[i])
        
        return PoseResult(
            keypoints=kpts_xy,
            scores=kpts_conf,
            bboxes=bboxes,
            skeleton_name=self.skeleton_name,
            raw_output=results
        )
