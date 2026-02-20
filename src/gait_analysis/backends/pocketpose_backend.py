"""
PocketPose Backend Implementation.

PocketPose is a lightweight whole-body pose estimation model that can run
efficiently using ONNX Runtime. It provides whole-body keypoints including
feet for gait analysis.

References:
- PocketPose: https://github.com/pocketpose/pocketpose
"""

import os
from pathlib import Path
from typing import Optional, List
import numpy as np

from ..pose_backend import BasePoseBackend, PoseResult, check_dependency


# Default model URL and cache location
POCKETPOSE_MODELS = {
    "pocketpose-wholebody": {
        "url": "https://github.com/pocketpose/pocketpose/releases/download/v1.0/pocketpose_wholebody.onnx",
        "filename": "pocketpose_wholebody.onnx",
        "skeleton": "wholebody133",
        "num_keypoints": 133
    },
    "pocketpose-body": {
        "url": "https://github.com/pocketpose/pocketpose/releases/download/v1.0/pocketpose_body.onnx",
        "filename": "pocketpose_body.onnx",
        "skeleton": "coco17",
        "num_keypoints": 17
    }
}


def get_cache_dir() -> Path:
    """Get the model cache directory."""
    cache_dir = Path(os.environ.get(
        "GAIT_ANALYSIS_CACHE",
        Path.home() / ".cache" / "gait_analysis" / "models"
    ))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class PocketPoseBackend(BasePoseBackend):
    """
    PocketPose backend for lightweight whole-body pose estimation.
    
    Uses ONNX Runtime for efficient inference on CPU or GPU.
    Supports whole-body keypoints including feet for gait analysis.
    """
    
    def __init__(
        self,
        model_name: str = "pocketpose-wholebody",
        model_path: Optional[str] = None,
        device: str = "auto",
        conf_threshold: float = 0.3
    ):
        """
        Initialize PocketPose backend.
        
        Args:
            model_name: Model variant ("pocketpose-wholebody" or "pocketpose-body")
            model_path: Custom path to ONNX model file (overrides model_name)
            device: Inference device ('auto', 'cpu', 'cuda')
            conf_threshold: Confidence threshold for keypoints
        """
        super().__init__(device=device, conf_threshold=conf_threshold)
        
        # Check dependency
        check_dependency("onnxruntime", "pocketpose", "pocketpose")
        
        self.model_name = model_name
        self.model_path = model_path
        
        # Get model info
        if model_name in POCKETPOSE_MODELS:
            self._model_info = POCKETPOSE_MODELS[model_name]
        else:
            # Custom model - assume wholebody
            self._model_info = {
                "skeleton": "wholebody133",
                "num_keypoints": 133
            }
        
        self._session = None
        self._input_name = None
        self._input_shape = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the ONNX model."""
        import onnxruntime as ort
        
        # Determine model path
        if self.model_path:
            model_file = Path(self.model_path)
        else:
            model_file = get_cache_dir() / self._model_info["filename"]
            
            if not model_file.exists():
                self._download_model(model_file)
        
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_file}. "
                f"Please download manually or specify model_path."
            )
        
        # Setup ONNX Runtime session
        providers = self._get_providers()
        self._session = ort.InferenceSession(str(model_file), providers=providers)
        
        # Get input info
        input_info = self._session.get_inputs()[0]
        self._input_name = input_info.name
        self._input_shape = input_info.shape  # Typically [1, 3, H, W]
        
        self._is_initialized = True
    
    def _get_providers(self) -> List[str]:
        """Get ONNX Runtime execution providers based on device."""
        import onnxruntime as ort
        
        available = ort.get_available_providers()
        
        if self.device == "cpu":
            return ["CPUExecutionProvider"]
        elif self.device in ("cuda", "gpu"):
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]
        else:  # auto
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]
    
    def _download_model(self, target_path: Path):
        """Download model from URL."""
        import urllib.request
        
        if "url" not in self._model_info:
            raise ValueError(
                f"No download URL for model '{self.model_name}'. "
                f"Please download manually and specify model_path."
            )
        
        url = self._model_info["url"]
        print(f"Downloading PocketPose model from {url}...")
        
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, target_path)
            print(f"Model saved to {target_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
    
    @property
    def name(self) -> str:
        return "pocketpose"
    
    @property
    def skeleton_name(self) -> str:
        return self._model_info["skeleton"]
    
    def predict(self, image: np.ndarray) -> PoseResult:
        """
        Run pose estimation on an image.
        
        Args:
            image: BGR image as numpy array (H, W, 3)
            
        Returns:
            PoseResult with detected poses
        """
        if self._session is None:
            return PoseResult.empty(self.skeleton_name)
        
        # Preprocess image
        input_tensor = self._preprocess(image)
        
        # Run inference
        outputs = self._session.run(None, {self._input_name: input_tensor})
        
        # Parse outputs
        return self._postprocess(outputs, image.shape[:2])
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        import cv2
        
        # Get expected input size
        if self._input_shape and len(self._input_shape) == 4:
            _, _, h, w = self._input_shape
            if h > 0 and w > 0:
                target_size = (w, h)
            else:
                target_size = (256, 192)  # Default
        else:
            target_size = (256, 192)
        
        # Resize
        resized = cv2.resize(image, target_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Transpose to NCHW format
        transposed = normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def _postprocess(
        self,
        outputs: List[np.ndarray],
        original_size: tuple
    ) -> PoseResult:
        """Parse model outputs to PoseResult."""
        # Output format varies by model - adapt as needed
        # Typical: heatmaps (1, K, H, W) or keypoints (1, K, 3)
        
        if len(outputs) == 0:
            return PoseResult.empty(self.skeleton_name)
        
        output = outputs[0]
        
        # Check output format
        if output.ndim == 4:
            # Heatmap format: (B, K, H, W)
            keypoints, scores = self._decode_heatmaps(output, original_size)
        elif output.ndim == 3:
            # Direct keypoint format: (B, K, 2 or 3)
            if output.shape[-1] == 3:
                keypoints = output[:, :, :2]
                scores = output[:, :, 2]
            else:
                keypoints = output
                scores = np.ones(output.shape[:2], dtype=np.float32)
        else:
            return PoseResult.empty(self.skeleton_name)
        
        # Scale keypoints to original image size
        h, w = original_size
        if keypoints.max() <= 1.0:
            # Normalized coordinates
            keypoints[:, :, 0] *= w
            keypoints[:, :, 1] *= h
        
        # Compute bounding boxes
        bboxes = np.array([
            self._bbox_from_keypoints(keypoints[i], scores[i])
            for i in range(len(keypoints))
        ])
        
        return PoseResult(
            keypoints=keypoints,
            scores=scores,
            bboxes=bboxes,
            skeleton_name=self.skeleton_name
        )
    
    def _decode_heatmaps(
        self,
        heatmaps: np.ndarray,
        original_size: tuple
    ) -> tuple:
        """Decode heatmaps to keypoint coordinates."""
        batch_size, num_keypoints, h, w = heatmaps.shape
        
        keypoints = np.zeros((batch_size, num_keypoints, 2), dtype=np.float32)
        scores = np.zeros((batch_size, num_keypoints), dtype=np.float32)
        
        for b in range(batch_size):
            for k in range(num_keypoints):
                heatmap = heatmaps[b, k]
                
                # Find max location
                max_val = heatmap.max()
                if max_val > self.conf_threshold:
                    max_idx = np.argmax(heatmap)
                    y, x = np.unravel_index(max_idx, heatmap.shape)
                    
                    # Scale to original image size
                    orig_h, orig_w = original_size
                    keypoints[b, k, 0] = x * orig_w / w
                    keypoints[b, k, 1] = y * orig_h / h
                    scores[b, k] = max_val
        
        return keypoints, scores
