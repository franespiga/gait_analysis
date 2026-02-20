"""
AlphaPose Backend Implementation.

AlphaPose is a real-time multi-person pose estimation system that supports
HALPE whole-body keypoints (136 points) including detailed foot keypoints.

References:
- AlphaPose: https://github.com/MVIG-SJTU/AlphaPose
- HALPE dataset: https://github.com/Fang-Haoshu/Halpe-FullBody
- Model Zoo: https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from ..pose_backend import BasePoseBackend, PoseResult, check_dependency


# HALPE keypoint configurations
HALPE_26_SKELETON = {
    "name": "halpe26",
    "num_keypoints": 26,
    "description": "HALPE 26-point body format with feet",
}

HALPE_136_SKELETON = {
    "name": "halpe136", 
    "num_keypoints": 136,
    "description": "HALPE 136-point whole-body format (body + face + hands + feet)",
}


class AlphaPoseBackend(BasePoseBackend):
    """
    AlphaPose backend with HALPE whole-body keypoints.
    
    Supports:
    - HALPE-26: 26 body keypoints including feet
    - HALPE-136: 136 whole-body keypoints (body, face, hands, feet)
    
    Both variants include foot keypoints (heel, big toe, small toe) which
    are essential for accurate gait analysis.
    """
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        checkpoint: Optional[str] = None,
        device: str = "auto",
        conf_threshold: float = 0.3,
        detector: str = "yolov8",
        use_wholebody: bool = True
    ):
        """
        Initialize AlphaPose backend.
        
        Args:
            config_file: Path to AlphaPose config file or preset name
                        ("halpe26", "halpe136", or custom path)
            checkpoint: Path to model checkpoint
            device: Inference device ('auto', 'cpu', 'cuda')
            conf_threshold: Confidence threshold for keypoints
            detector: Person detector type ("yolov8", "yolox", "fasterrcnn")
            use_wholebody: If True, use HALPE-136; else use HALPE-26
        """
        super().__init__(device=device, conf_threshold=conf_threshold)
        
        # Check dependencies
        check_dependency("torch", "alphapose", "alphapose")
        check_dependency("torchvision", "alphapose", "alphapose")
        
        self.config_file = config_file
        self.checkpoint = checkpoint
        self.detector_type = detector
        self.use_wholebody = use_wholebody
        
        self._model = None
        self._detector = None
        self._transform = None
        self._config = None
        self._using_fallback = False  # Track if we're using YOLO fallback
        
        self._skeleton_info = HALPE_136_SKELETON if use_wholebody else HALPE_26_SKELETON
        
        self._load_model()
    
    def _load_model(self):
        """Load AlphaPose model and detector."""
        import torch
        
        # Resolve device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load detector first
        self._load_detector()
        
        # Try to load AlphaPose model
        try:
            self._load_alphapose_model()
        except Exception as e:
            print(f"AlphaPose model loading failed: {e}")
            print("Using fallback pose estimation")
            self._use_fallback_model()
        
        self._is_initialized = True
    
    def _load_detector(self):
        """Load person detector."""
        try:
            if self.detector_type == "yolov8":
                from ultralytics import YOLO
                self._detector = YOLO("yolov8n.pt")
                self._detector_type = "yolov8"
            else:
                # Use YOLO as default fallback
                from ultralytics import YOLO
                self._detector = YOLO("yolov8n.pt")
                self._detector_type = "yolov8"
        except ImportError:
            print("Warning: Person detector not available. Using full image.")
            self._detector = None
    
    def _load_alphapose_model(self):
        """Load the actual AlphaPose model."""
        import torch
        
        # Check if AlphaPose is installed
        try:
            from alphapose.models import builder
            from alphapose.utils.config import update_config
            from alphapose.utils.transforms import get_affine_transform
        except ImportError:
            raise ImportError(
                "AlphaPose package not found. Install from: "
                "https://github.com/MVIG-SJTU/AlphaPose"
            )
        
        # Find AlphaPose directory
        alphapose_dir = self._find_alphapose_dir()
        
        # Load config
        if self.config_file is None:
            # Use default config based on model type
            if self.use_wholebody:
                self.config_file = alphapose_dir / "configs" / "halpe_136" / "resnet" / "256x192_res50_lr1e-3_2x-regression.yaml"
            else:
                self.config_file = alphapose_dir / "configs" / "halpe_26" / "resnet" / "256x192_res50_lr1e-3_1x.yaml"
        
        if not Path(self.config_file).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        # Auto-find checkpoint if not specified
        if self.checkpoint is None:
            pretrained_dir = alphapose_dir / "pretrained_models"
            if self.use_wholebody:
                default_checkpoint = pretrained_dir / "halpe136_fast50_regression_256x192.pth"
            else:
                default_checkpoint = pretrained_dir / "halpe26_fast_res50_256x192.pth"
            
            if default_checkpoint.exists():
                self.checkpoint = str(default_checkpoint)
                print(f"AlphaPose: Using checkpoint {self.checkpoint}")
            else:
                # Check for any .pth file in pretrained_models
                pth_files = list(pretrained_dir.glob("*.pth")) if pretrained_dir.exists() else []
                if pth_files:
                    self.checkpoint = str(pth_files[0])
                    print(f"AlphaPose: Using checkpoint {self.checkpoint}")
        
        # Load model
        self._config = update_config(str(self.config_file))
        self._model = builder.build_sppe(self._config.MODEL, preset_cfg=self._config.DATA_PRESET)
        
        # Load checkpoint
        if self.checkpoint:
            checkpoint_path = Path(self.checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            self._model.load_state_dict(checkpoint)
        else:
            print("AlphaPose: Warning - No checkpoint loaded, using random weights")
        
        self._model = self._model.to(self.device)
        self._model.eval()
    
    def _find_alphapose_dir(self) -> Path:
        """Find AlphaPose installation directory."""
        # Check common locations (in priority order)
        candidates = [
            # 1. Environment variable (highest priority)
            Path(os.environ.get("ALPHAPOSE_DIR", "")),
            # 2. Local models directory (bundled with gait_analysis)
            Path(__file__).parent.parent.parent.parent / "models" / "AlphaPose",
            # 3. Relative to current working directory
            Path("./models/AlphaPose"),
            # 4. User home directory
            Path.home() / "AlphaPose",
            # 5. System locations
            Path("/opt/AlphaPose"),
            Path("C:/AlphaPose"),
            Path("./AlphaPose"),
        ]
        
        for candidate in candidates:
            if candidate.exists() and (candidate / "configs").exists():
                return candidate
        
        raise FileNotFoundError(
            "AlphaPose directory not found. Either:\n"
            "  1. Set ALPHAPOSE_DIR environment variable\n"
            "  2. Place AlphaPose in models/AlphaPose/\n"
            "  3. Clone AlphaPose to your home directory"
        )
    
    def _use_fallback_model(self):
        """Use a fallback model when AlphaPose isn't available."""
        import warnings
        
        # Fall back to using YOLO for pose if available
        try:
            from ultralytics import YOLO
            self._model = "yolo_fallback"
            self._using_fallback = True
            self._fallback_pose = YOLO("yolov8n-pose.pt")
            
            # IMPORTANT: Update skeleton info to reflect actual capabilities
            # YOLO fallback only provides COCO-17 keypoints (no feet!)
            self._skeleton_info = {
                "name": "coco17",  # Changed from halpe136/halpe26
                "num_keypoints": 17,
                "description": "COCO-17 (YOLOv8 fallback - NO FEET KEYPOINTS)",
            }
            
            warnings.warn(
                "AlphaPose model not available. Using YOLOv8-pose fallback. "
                "WARNING: Feet keypoints (heel/toe) will NOT be available. "
                "For full AlphaPose with feet, install from: "
                "https://github.com/MVIG-SJTU/AlphaPose",
                UserWarning
            )
            print("AlphaPose: Using YOLOv8-pose as fallback (NO FEET KEYPOINTS)")
        except ImportError:
            self._model = None
            self._using_fallback = True
            print("AlphaPose: No fallback model available")
    
    @property
    def name(self) -> str:
        return "alphapose"
    
    @property
    def skeleton_name(self) -> str:
        return self._skeleton_info["name"]
    
    @property
    def using_fallback(self) -> bool:
        """Check if using YOLOv8 fallback (no feet keypoints)."""
        return self._using_fallback
    
    @property
    def has_feet_keypoints(self) -> bool:
        """Check if feet keypoints are available."""
        return not self._using_fallback
    
    def predict(self, image: np.ndarray) -> PoseResult:
        """
        Run pose estimation on an image.
        
        Args:
            image: BGR image as numpy array (H, W, 3)
            
        Returns:
            PoseResult with detected poses
        """
        if self._model is None:
            return PoseResult.empty(self.skeleton_name)
        
        # Use fallback if needed
        if self._model == "yolo_fallback":
            return self._predict_yolo_fallback(image)
        
        import torch
        
        # Detect persons
        person_bboxes = self._detect_persons(image)
        
        if len(person_bboxes) == 0:
            return PoseResult.empty(self.skeleton_name)
        
        all_keypoints = []
        all_scores = []
        
        for bbox in person_bboxes:
            # Crop and process each person
            kpts, scores = self._process_person(image, bbox)
            if kpts is not None:
                all_keypoints.append(kpts)
                all_scores.append(scores)
        
        if not all_keypoints:
            return PoseResult.empty(self.skeleton_name)
        
        return PoseResult(
            keypoints=np.array(all_keypoints),
            scores=np.array(all_scores),
            bboxes=np.array(person_bboxes[:len(all_keypoints)]),
            skeleton_name=self.skeleton_name
        )
    
    def _predict_yolo_fallback(self, image: np.ndarray) -> PoseResult:
        """Use YOLO pose as fallback.
        
        NOTE: This fallback uses YOLOv8-pose with COCO-17 keypoints.
        It does NOT provide feet keypoints (heel/toe).
        The skeleton_name is set to "coco17" to reflect this.
        """
        results = self._fallback_pose(image, verbose=False, device=self.device)
        
        if len(results) == 0 or results[0].keypoints is None:
            return PoseResult.empty("coco17")
        
        keypoints_data = results[0].keypoints
        
        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return PoseResult.empty("coco17")
        
        kpts_xy = keypoints_data.xy.cpu().numpy()  # (N, 17, 2)
        kpts_conf = (keypoints_data.conf.cpu().numpy()
                    if keypoints_data.conf is not None
                    else np.ones(kpts_xy.shape[:2]))  # (N, 17)
        
        # Return COCO-17 directly - don't try to fake HALPE format
        # This ensures downstream code knows feet are NOT available
        num_people = len(kpts_xy)
        
        # Compute bounding boxes
        bboxes = np.array([
            self._bbox_from_keypoints(kpts_xy[i], kpts_conf[i])
            for i in range(num_people)
        ])
        
        return PoseResult(
            keypoints=kpts_xy,
            scores=kpts_conf,
            bboxes=bboxes,
            skeleton_name="coco17",  # Explicitly COCO-17, not HALPE
            raw_output=results
        )
    
    def _detect_persons(self, image: np.ndarray) -> np.ndarray:
        """Detect persons in image and return bounding boxes."""
        if self._detector is None:
            h, w = image.shape[:2]
            return np.array([[0, 0, w, h]])
        
        if self._detector_type == "yolov8":
            results = self._detector(image, verbose=False, classes=[0])
            if len(results) > 0 and results[0].boxes is not None:
                return results[0].boxes.xyxy.cpu().numpy()
        
        return np.array([])
    
    def _process_person(
        self,
        image: np.ndarray,
        bbox: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process a single person crop."""
        import torch
        import cv2
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Expand bbox slightly for better pose estimation
        h, w = image.shape[:2]
        pad_w = int((x2 - x1) * 0.1)
        pad_h = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        # Crop person
        person_crop = image[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None, None
        
        # Preprocess for model
        input_size = (192, 256)  # Standard AlphaPose input
        resized = cv2.resize(person_crop, input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # To tensor
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        tensor = tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self._model(tensor)
        
        # Parse output (typically heatmaps)
        if isinstance(output, torch.Tensor):
            heatmaps = output[0].cpu().numpy()
        else:
            heatmaps = output.heatmap[0].cpu().numpy()
        
        # Decode heatmaps
        keypoints, scores = self._decode_heatmaps(heatmaps, (x2-x1, y2-y1))
        
        # Scale back to original coordinates
        keypoints[:, 0] += x1
        keypoints[:, 1] += y1
        
        return keypoints, scores
    
    def _decode_heatmaps(
        self,
        heatmaps: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode heatmaps to keypoint coordinates."""
        num_keypoints, hm_h, hm_w = heatmaps.shape
        w, h = original_size
        
        keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
        scores = np.zeros(num_keypoints, dtype=np.float32)
        
        for k in range(num_keypoints):
            heatmap = heatmaps[k]
            max_val = heatmap.max()
            
            if max_val > self.conf_threshold:
                max_idx = np.argmax(heatmap)
                hm_y, hm_x = np.unravel_index(max_idx, heatmap.shape)
                
                keypoints[k, 0] = hm_x * w / hm_w
                keypoints[k, 1] = hm_y * h / hm_h
                scores[k] = max_val
        
        return keypoints, scores


def is_alphapose_available() -> bool:
    """Check if AlphaPose is available."""
    try:
        from alphapose.models import builder
        return True
    except ImportError:
        return False
