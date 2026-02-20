"""
OpenPose Backend Implementation.

Provides pose estimation using CMU OpenPose with Body_25 format,
which includes detailed foot keypoints (heel, big toe, small toe).

Supports two modes:
1. Python bindings (pyopenpose) - if built from source
2. Subprocess mode - using pre-built OpenPoseDemo executable

References:
- OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose
- Keypoints: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np

from ..pose_backend import BasePoseBackend, PoseResult, DependencyError


class OpenPoseBackend(BasePoseBackend):
    """
    OpenPose backend with Body_25 keypoints.
    
    Provides 25 body keypoints including:
    - Standard body joints (shoulders, elbows, wrists, hips, knees, ankles)
    - Detailed foot keypoints: heels, big toes, small toes
    
    This is excellent for gait analysis due to accurate foot positioning.
    """
    
    def __init__(
        self,
        openpose_path: Optional[str] = None,
        model_pose: str = "BODY_25",
        net_resolution: str = "-1x368",
        output_resolution: str = "-1x-1",
        device: str = "auto",
        conf_threshold: float = 0.3,
        use_subprocess: bool = True
    ):
        """
        Initialize OpenPose backend.
        
        Args:
            openpose_path: Path to OpenPose installation directory.
                          If None, uses OPENPOSE_PATH environment variable.
            model_pose: Pose model ("BODY_25" recommended for feet)
            net_resolution: Network input resolution
            output_resolution: Output resolution
            device: Inference device (for Python bindings)
            conf_threshold: Confidence threshold
            use_subprocess: If True, use subprocess mode with OpenPoseDemo.exe
                           If False, try Python bindings (requires build)
        """
        super().__init__(device=device, conf_threshold=conf_threshold)
        
        # Resolve OpenPose path
        if openpose_path is None:
            openpose_path = os.environ.get("OPENPOSE_PATH")
        
        if openpose_path is None:
            raise DependencyError(
                backend_name="openpose",
                package="OpenPose",
                install_cmd="Set OPENPOSE_PATH environment variable to OpenPose installation directory"
            )
        
        self.openpose_path = Path(openpose_path)
        self.model_pose = model_pose
        self.net_resolution = net_resolution
        self.output_resolution = output_resolution
        self.use_subprocess = use_subprocess
        
        self._op_wrapper = None
        self._op_module = None
        
        # Initialize based on mode
        if use_subprocess:
            self._init_subprocess_mode()
        else:
            self._init_python_bindings()
    
    def _init_subprocess_mode(self):
        """Initialize subprocess mode with OpenPoseDemo executable."""
        # Find OpenPoseDemo executable
        possible_paths = [
            self.openpose_path / "bin" / "OpenPoseDemo.exe",
            self.openpose_path / "build" / "x64" / "Release" / "OpenPoseDemo.exe",
            self.openpose_path / "OpenPoseDemo.exe",
            self.openpose_path / "bin" / "openpose.bin",  # Linux
        ]
        
        self.demo_path = None
        for path in possible_paths:
            if path.exists():
                self.demo_path = path
                break
        
        if self.demo_path is None:
            raise FileNotFoundError(
                f"OpenPoseDemo executable not found in {self.openpose_path}. "
                f"Searched: {[str(p) for p in possible_paths]}"
            )
        
        # Find models folder
        models_paths = [
            self.openpose_path / "models",
            self.openpose_path.parent / "models",
        ]
        
        self.models_path = None
        for path in models_paths:
            if path.exists():
                self.models_path = path
                break
        
        if self.models_path is None:
            raise FileNotFoundError(
                f"OpenPose models folder not found. "
                f"Searched: {[str(p) for p in models_paths]}"
            )
        
        self._is_initialized = True
    
    def _init_python_bindings(self):
        """Initialize Python bindings mode."""
        try:
            # Add OpenPose Python path
            python_path = self.openpose_path / "build" / "python" / "openpose"
            release_path = python_path / "Release"
            
            if str(python_path) not in sys.path:
                sys.path.append(str(python_path))
            if release_path.exists() and str(release_path) not in sys.path:
                sys.path.append(str(release_path))
            
            # Add DLL path on Windows
            if sys.platform == "win32":
                bin_paths = [
                    self.openpose_path / "build" / "x64" / "Release",
                    self.openpose_path / "build" / "bin",
                    self.openpose_path / "bin",
                ]
                for bin_path in bin_paths:
                    if bin_path.exists():
                        os.add_dll_directory(str(bin_path))
            
            import pyopenpose as op
            self._op_module = op
            
            # Configure OpenPose
            params = {
                "model_folder": str(self.openpose_path / "models"),
                "net_resolution": self.net_resolution,
                "output_resolution": self.output_resolution,
                "model_pose": self.model_pose,
            }
            
            self._op_wrapper = op.WrapperPython()
            self._op_wrapper.configure(params)
            self._op_wrapper.start()
            self._is_initialized = True
            
        except ImportError as e:
            raise DependencyError(
                backend_name="openpose",
                package="pyopenpose",
                install_cmd="Build OpenPose with BUILD_PYTHON=ON or use subprocess mode"
            )
    
    @property
    def name(self) -> str:
        return "openpose"
    
    @property
    def skeleton_name(self) -> str:
        return "body25"
    
    def predict(self, image: np.ndarray) -> PoseResult:
        """Run pose estimation on an image."""
        if self.use_subprocess:
            return self._predict_subprocess(image)
        else:
            return self._predict_python(image)
    
    def _predict_subprocess(self, image: np.ndarray) -> PoseResult:
        """Run prediction using OpenPoseDemo subprocess."""
        import cv2
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Save input image
            input_path = tmpdir / "input.jpg"
            cv2.imwrite(str(input_path), image)
            
            # Output paths
            output_json_dir = tmpdir / "output"
            output_json_dir.mkdir()
            
            # Build command
            cmd = [
                str(self.demo_path),
                "--image_dir", str(tmpdir),
                "--write_json", str(output_json_dir),
                "--display", "0",
                "--render_pose", "0",
                "--model_folder", str(self.models_path),
                "--model_pose", self.model_pose,
                "--net_resolution", self.net_resolution,
            ]
            
            # Run OpenPose
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(self.openpose_path)
                )
            except subprocess.TimeoutExpired:
                return PoseResult.empty(self.skeleton_name)
            except Exception as e:
                print(f"OpenPose subprocess error: {e}")
                return PoseResult.empty(self.skeleton_name)
            
            # Parse output JSON
            json_files = list(output_json_dir.glob("*.json"))
            if not json_files:
                return PoseResult.empty(self.skeleton_name)
            
            with open(json_files[0]) as f:
                data = json.load(f)
            
            return self._parse_openpose_json(data)
    
    def _predict_python(self, image: np.ndarray) -> PoseResult:
        """Run prediction using Python bindings."""
        if self._op_wrapper is None:
            return PoseResult.empty(self.skeleton_name)
        
        datum = self._op_module.Datum()
        datum.cvInputData = image
        self._op_wrapper.emplaceAndPop(self._op_module.VectorDatum([datum]))
        
        pose_keypoints = datum.poseKeypoints
        
        if pose_keypoints is None or len(pose_keypoints) == 0:
            return PoseResult.empty(self.skeleton_name)
        
        # OpenPose format: (N, 25, 3) where 3 is (x, y, confidence)
        keypoints_xy = pose_keypoints[:, :, :2]  # (N, 25, 2)
        scores = pose_keypoints[:, :, 2]          # (N, 25)
        
        # Compute bounding boxes
        bboxes = np.array([
            self._bbox_from_keypoints(keypoints_xy[i], scores[i])
            for i in range(len(keypoints_xy))
        ])
        
        return PoseResult(
            keypoints=keypoints_xy,
            scores=scores,
            bboxes=bboxes,
            skeleton_name=self.skeleton_name,
            raw_output=datum
        )
    
    def _parse_openpose_json(self, data: dict) -> PoseResult:
        """Parse OpenPose JSON output format."""
        people = data.get("people", [])
        
        if not people:
            return PoseResult.empty(self.skeleton_name)
        
        num_people = len(people)
        num_keypoints = 25  # Body_25
        
        keypoints = np.zeros((num_people, num_keypoints, 2), dtype=np.float32)
        scores = np.zeros((num_people, num_keypoints), dtype=np.float32)
        bboxes = np.zeros((num_people, 4), dtype=np.float32)
        
        for i, person in enumerate(people):
            # OpenPose JSON format: flat array [x1, y1, c1, x2, y2, c2, ...]
            kpts_flat = person.get("pose_keypoints_2d", [])
            
            if len(kpts_flat) >= num_keypoints * 3:
                kpts_array = np.array(kpts_flat).reshape(-1, 3)
                keypoints[i] = kpts_array[:num_keypoints, :2]
                scores[i] = kpts_array[:num_keypoints, 2]
            
            bboxes[i] = self._bbox_from_keypoints(keypoints[i], scores[i])
        
        return PoseResult(
            keypoints=keypoints,
            scores=scores,
            bboxes=bboxes,
            skeleton_name=self.skeleton_name
        )
    
    def get_rendered_frame(self, image: np.ndarray) -> np.ndarray:
        """Get frame with OpenPose visualization rendered (Python mode only)."""
        if not self.use_subprocess and self._op_wrapper is not None:
            datum = self._op_module.Datum()
            datum.cvInputData = image
            self._op_wrapper.emplaceAndPop(self._op_module.VectorDatum([datum]))
            if datum.cvOutputData is not None:
                return datum.cvOutputData
        return image
    
    def __del__(self):
        """Clean up OpenPose wrapper."""
        if self._op_wrapper is not None:
            try:
                self._op_wrapper.stop()
            except:
                pass


def is_openpose_available(openpose_path: Optional[str] = None) -> bool:
    """
    Check if OpenPose is available.
    
    Args:
        openpose_path: Path to OpenPose installation
        
    Returns:
        True if OpenPose can be used
    """
    if openpose_path is None:
        openpose_path = os.environ.get("OPENPOSE_PATH")
    
    if openpose_path is None:
        return False
    
    openpose_path = Path(openpose_path)
    
    # Check for demo executable (subprocess mode)
    demo_paths = [
        openpose_path / "bin" / "OpenPoseDemo.exe",
        openpose_path / "build" / "x64" / "Release" / "OpenPoseDemo.exe",
    ]
    
    for path in demo_paths:
        if path.exists():
            return True
    
    # Check for Python bindings
    try:
        python_path = openpose_path / "build" / "python" / "openpose"
        if str(python_path) not in sys.path:
            sys.path.append(str(python_path))
        import pyopenpose
        return True
    except ImportError:
        pass
    
    return False
