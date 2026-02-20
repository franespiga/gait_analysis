"""
OpenPose detector wrapper for pose estimation.

OpenPose provides detailed foot keypoints including heel and toes,
which greatly improves gait analysis accuracy.

References:
- OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose
- Keypoints: https://chingswy.github.io/easymocap-public-doc/database/2_keypoints.html

Installation:
OpenPose must be installed separately. See: 
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md

After installation, set the OPENPOSE_PATH environment variable to the OpenPose installation directory.
"""

import os
import sys
from typing import Optional
import numpy as np

from .base_detector import BaseDetector, GaitKeypoints
from .keypoint_config import (
    DetectorBackend,
    KeypointIndices,
    OPENPOSE_KEYPOINTS,
    get_keypoint_config
)


class OpenPoseDetector(BaseDetector):
    """
    OpenPose-based pose detector with foot keypoints.
    
    Provides 25 body keypoints (Body_25 format) including:
    - Hips, knees, ankles
    - Heels
    - Big toes and small toes
    
    This is the most detailed model for foot position analysis.
    """
    
    def __init__(
        self,
        openpose_path: Optional[str] = None,
        model_folder: Optional[str] = None,
        net_resolution: str = "-1x368",
        output_resolution: str = "-1x-1",
        model_pose: str = "BODY_25"
    ):
        """
        Initialize OpenPose detector.
        
        Args:
            openpose_path: Path to OpenPose installation directory.
                          If None, uses OPENPOSE_PATH environment variable.
            model_folder: Path to OpenPose models folder.
                         If None, uses {openpose_path}/models/
            net_resolution: Network input resolution (e.g., "-1x368")
            output_resolution: Output resolution (e.g., "-1x-1" for same as input)
            model_pose: Pose model to use ("BODY_25" recommended for foot keypoints)
        """
        self._backend = DetectorBackend.OPENPOSE
        self._keypoint_config = get_keypoint_config(DetectorBackend.OPENPOSE)
        
        # Resolve OpenPose path
        if openpose_path is None:
            openpose_path = os.environ.get("OPENPOSE_PATH")
        
        if openpose_path is None:
            raise ValueError(
                "OpenPose path not specified. Either pass openpose_path parameter "
                "or set the OPENPOSE_PATH environment variable."
            )
        
        self.openpose_path = openpose_path
        self.model_folder = model_folder or os.path.join(openpose_path, "models")
        self.net_resolution = net_resolution
        self.output_resolution = output_resolution
        self.model_pose = model_pose
        
        # Try to import OpenPose
        self.op = None
        self.op_wrapper = None
        self._init_openpose()
    
    def _init_openpose(self):
        """Initialize OpenPose library and wrapper."""
        try:
            # Add OpenPose Python path
            python_path = os.path.join(self.openpose_path, "build", "python", "openpose")
            release_path = os.path.join(self.openpose_path, "build", "python", "openpose", "Release")
            
            if python_path not in sys.path:
                sys.path.append(python_path)
            if os.path.exists(release_path) and release_path not in sys.path:
                sys.path.append(release_path)
            
            # Also add DLL path on Windows
            if sys.platform == "win32":
                bin_path = os.path.join(self.openpose_path, "build", "x64", "Release")
                bin_path2 = os.path.join(self.openpose_path, "build", "bin")
                if os.path.exists(bin_path):
                    os.add_dll_directory(bin_path)
                if os.path.exists(bin_path2):
                    os.add_dll_directory(bin_path2)
            
            import pyopenpose as op
            self.op = op
            
            # Configure OpenPose
            params = {
                "model_folder": self.model_folder,
                "net_resolution": self.net_resolution,
                "output_resolution": self.output_resolution,
                "model_pose": self.model_pose,
            }
            
            # Create wrapper
            self.op_wrapper = op.WrapperPython()
            self.op_wrapper.configure(params)
            self.op_wrapper.start()
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import OpenPose. Make sure OpenPose is properly installed "
                f"and the Python bindings are built. Error: {e}\n"
                f"OpenPose path: {self.openpose_path}\n"
                f"See: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenPose: {e}")
    
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
        Detect keypoints using OpenPose.
        """
        if self.op_wrapper is None:
            return None
        
        # Create datum and process
        datum = self.op.Datum()
        datum.cvInputData = frame
        self.op_wrapper.emplaceAndPop(self.op.VectorDatum([datum]))
        
        # Get keypoints
        pose_keypoints = datum.poseKeypoints
        
        if pose_keypoints is None or len(pose_keypoints) == 0:
            return None
        
        # Get the first person (highest confidence/area)
        person_kpts = pose_keypoints[0]  # Shape: (25, 3) for Body_25
        
        # OpenPose format: (x, y, confidence)
        kpts_xy = person_kpts[:, :2]
        kpts_conf = person_kpts[:, 2]
        
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
            # Heels
            left_heel=self._extract_point(kpts_xy, kpts_conf, config.left_heel, conf_threshold),
            right_heel=self._extract_point(kpts_xy, kpts_conf, config.right_heel, conf_threshold),
            # Toes (big toe)
            left_toe=self._extract_point(kpts_xy, kpts_conf, config.left_toe, conf_threshold),
            right_toe=self._extract_point(kpts_xy, kpts_conf, config.right_toe, conf_threshold),
            # Raw keypoints for visualization
            raw_keypoints=person_kpts,
            # Confidence scores
            confidences=self._extract_confidences(kpts_conf, config)
        )
        
        return gait_kpts
    
    def get_visualization_results(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5
    ):
        """
        Get OpenPose results for visualization.
        
        Returns datum object with rendered output.
        """
        if self.op_wrapper is None:
            return None
        
        datum = self.op.Datum()
        datum.cvInputData = frame
        self.op_wrapper.emplaceAndPop(self.op.VectorDatum([datum]))
        
        return datum
    
    def get_rendered_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Get frame with OpenPose visualization rendered.
        """
        datum = self.get_visualization_results(frame)
        if datum is not None and datum.cvOutputData is not None:
            return datum.cvOutputData
        return frame
    
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
    
    def __del__(self):
        """Clean up OpenPose wrapper."""
        if self.op_wrapper is not None:
            try:
                self.op_wrapper.stop()
            except:
                pass


def is_openpose_available(openpose_path: Optional[str] = None) -> bool:
    """
    Check if OpenPose is available and properly configured.
    
    Args:
        openpose_path: Path to OpenPose installation
        
    Returns:
        True if OpenPose can be loaded, False otherwise
    """
    if openpose_path is None:
        openpose_path = os.environ.get("OPENPOSE_PATH")
    
    if openpose_path is None:
        return False
    
    try:
        python_path = os.path.join(openpose_path, "build", "python", "openpose")
        if python_path not in sys.path:
            sys.path.append(python_path)
        
        import pyopenpose
        return True
    except ImportError:
        return False

