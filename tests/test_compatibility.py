"""
Compatibility Test Suite.

Runs all available pose estimation backends over a sample video
and outputs comprehensive gait analysis results plus annotated videos.

Usage:
    pytest tests/test_compatibility.py -v
    pytest tests/test_compatibility.py -v -k "yolov8"  # Test specific backend
    
    # Run as standalone script
    python -m tests.test_compatibility --video data/sample_video.mp4
    
Output:
    checks/YYYYMMDD_HHMM/
        ├── <backend>/
        │   ├── gait_analysis.json
        │   └── annotated_video.mp4
        └── compatibility_report.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pytest
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gait_analysis.backend_registry import (
    list_backends,
    create_backend,
    check_backend_available,
    BackendInfo,
)
from gait_analysis.pose_backend import PoseResult
from gait_analysis.keypoint_schema import KeypointSchemaMapper, CanonicalGaitKeypoints


# ============================================================================
# Configuration
# ============================================================================

import os

SAMPLE_VIDEO_PATHS = [
    Path("data/sample_video.mp4"),
    Path("data/sample_video.avi"),
    Path("data/sample_video.mov"),
    Path("data/sample_video.mkv"),
    Path("data/test"),  # Check subdirectory
]

# Default model paths for each backend
# Can be overridden via environment variables or CLI arguments
DEFAULT_MODEL_PATHS = {
    "yolov8": "yolov8n-pose.pt",
    "yolo_coco": "yolov8n-pose.pt",
    "yolo_lower": "models/yolo_lower/best.pt",  # Fine-tuned lower body model
}

# Environment variable names for model paths
MODEL_PATH_ENV_VARS = {
    "yolov8": "GAIT_YOLOV8_MODEL",
    "yolo_coco": "GAIT_YOLOV8_MODEL",
    "yolo_lower": "GAIT_YOLO_LOWER_MODEL",
}

# Global model path overrides (set via CLI or programmatically)
_model_path_overrides: Dict[str, str] = {}


def get_model_path(backend_name: str) -> Optional[str]:
    """Get model path for a backend from overrides, env vars, or defaults."""
    # Check CLI/programmatic overrides first
    if backend_name in _model_path_overrides:
        return _model_path_overrides[backend_name]
    
    # Check environment variable
    env_var = MODEL_PATH_ENV_VARS.get(backend_name)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    
    # Return default
    return DEFAULT_MODEL_PATHS.get(backend_name)


def set_model_path(backend_name: str, model_path: str) -> None:
    """Set model path override for a backend."""
    _model_path_overrides[backend_name] = model_path


# Skeleton-based keypoint focus (used when backend returns actual skeleton)
SKELETON_KEYPOINT_FOCUS = {
    "coco17": {
        "keypoints": ["left_hip", "right_hip", "left_knee", "right_knee", 
                      "left_ankle", "right_ankle"],
        "has_feet": False,
        "description": "COCO-17: Ankles only (no heel/toe)"
    },
    "coco_lower10": {
        "keypoints": ["left_hip", "right_hip", "left_knee", "right_knee",
                      "left_ankle", "right_ankle", "left_heel", "right_heel",
                      "left_toe", "right_toe"],
        "has_feet": True,
        "description": "Lower-body 10: Full feet with heel/toe"
    },
}


def get_keypoint_focus_for_skeleton(skeleton_name: str, backend_name: str) -> Dict[str, Any]:
    """Get keypoint focus based on actual skeleton, with fallback to backend config."""
    # First try skeleton-based lookup
    if skeleton_name in SKELETON_KEYPOINT_FOCUS:
        focus = SKELETON_KEYPOINT_FOCUS[skeleton_name].copy()
        # Add note if this differs from expected
        if backend_name in BACKEND_KEYPOINT_FOCUS:
            expected = BACKEND_KEYPOINT_FOCUS[backend_name]
            if expected["has_feet"] and not focus["has_feet"]:
                focus["description"] += " (FALLBACK - no feet available)"
        return focus
    
    # Fall back to backend-based lookup
    if backend_name in BACKEND_KEYPOINT_FOCUS:
        return BACKEND_KEYPOINT_FOCUS[backend_name]
    
    # Default
    return SKELETON_KEYPOINT_FOCUS["coco17"]

BACKEND_KEYPOINT_FOCUS = {
    "yolov8": {
        "keypoints": ["left_hip", "right_hip", "left_knee", "right_knee", 
                      "left_ankle", "right_ankle"],
        "has_feet": False,
        "description": "COCO-17: Ankles only (no heel/toe)"
    },
    "yolo_coco": {
        "keypoints": ["left_hip", "right_hip", "left_knee", "right_knee", 
                      "left_ankle", "right_ankle"],
        "has_feet": False,
        "description": "COCO-17: Ankles only (no heel/toe)"
    },
    "yolo_lower": {
        "keypoints": ["left_hip", "right_hip", "left_knee", "right_knee",
                      "left_ankle", "right_ankle", "left_heel", "right_heel",
                      "left_toe", "right_toe"],
        "has_feet": True,
        "description": "Lower-body 10: Full feet with heel/toe"
    },
}

SKELETON_CONNECTIONS = {
    "default": [
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_hip", "right_hip"),
    ],
    "with_feet": [
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("left_ankle", "left_heel"),
        ("left_ankle", "left_toe"),
        ("left_heel", "left_toe"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("right_ankle", "right_heel"),
        ("right_ankle", "right_toe"),
        ("right_heel", "right_toe"),
        ("left_hip", "right_hip"),
    ],
}

KEYPOINT_COLORS = {
    "hip": (255, 128, 0),
    "knee": (0, 255, 128),
    "ankle": (128, 0, 255),
    "heel": (255, 0, 128),
    "toe": (0, 128, 255),
    "small_toe": (128, 128, 255),
}


# ============================================================================
# Utility Functions
# ============================================================================

def get_output_directory() -> Path:
    """Create and return timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path("checks") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def find_sample_video() -> Optional[Path]:
    """Find sample video in data folder (including subdirectories)."""
    for path in SAMPLE_VIDEO_PATHS:
        if path.exists():
            if path.is_file():
                return path
            elif path.is_dir():
                for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                    videos = list(path.glob(f"*{ext}"))
                    if videos:
                        return videos[0]
    
    data_dir = Path("data")
    if data_dir.exists():
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            videos = list(data_dir.glob(f"**/*{ext}"))
            if videos:
                return videos[0]
    return None


def get_keypoint_color(keypoint_name: str) -> tuple:
    """Get color for a keypoint based on its name."""
    for key, color in KEYPOINT_COLORS.items():
        if key in keypoint_name:
            return color
    return (200, 200, 200)


def draw_skeleton_on_frame(
    frame: np.ndarray,
    canonical: CanonicalGaitKeypoints,
    backend_name: str
) -> np.ndarray:
    """Draw waist-to-feet skeleton on frame."""
    annotated = frame.copy()
    
    focus = BACKEND_KEYPOINT_FOCUS.get(backend_name, BACKEND_KEYPOINT_FOCUS["yolov8"])
    has_feet = focus["has_feet"]
    connections = SKELETON_CONNECTIONS["with_feet" if has_feet else "default"]
    
    for start_name, end_name in connections:
        start_pt = getattr(canonical, start_name, None)
        end_pt = getattr(canonical, end_name, None)
        
        if start_pt is not None and end_pt is not None:
            pt1 = (int(start_pt[0]), int(start_pt[1]))
            pt2 = (int(end_pt[0]), int(end_pt[1]))
            color = get_keypoint_color(start_name)
            cv2.line(annotated, pt1, pt2, color, 2)
    
    keypoint_names = focus["keypoints"]
    for kpt_name in keypoint_names:
        pt = getattr(canonical, kpt_name, None)
        if pt is not None:
            center = (int(pt[0]), int(pt[1]))
            color = get_keypoint_color(kpt_name)
            cv2.circle(annotated, center, 5, color, -1)
            cv2.circle(annotated, center, 6, (255, 255, 255), 1)
            
            label = kpt_name.replace("left_", "L_").replace("right_", "R_")
            cv2.putText(
                annotated, label, 
                (center[0] + 8, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
    
    return annotated


def compute_gait_metrics(
    detections: List[Dict],
    focus: Dict[str, Any],
    fps: float
) -> Dict[str, Any]:
    """Compute gait analysis metrics from detections."""
    
    metrics = {
        "backend_keypoint_focus": focus["keypoints"],
        "has_foot_keypoints": focus["has_feet"],
        "total_frames_with_detection": len(detections),
        "frames_with_full_lower_body": 0,
        "frames_with_feet": 0,
        "keypoint_detection_rates": {},
        "fps": fps,
    }
    
    keypoint_counts = {}
    for det in detections:
        for kpt in det.get("keypoints", []):
            keypoint_counts[kpt] = keypoint_counts.get(kpt, 0) + 1
        
        if det.get("has_feet", False):
            metrics["frames_with_feet"] += 1
        
        required_lower = ["left_hip", "right_hip", "left_knee", "right_knee", 
                          "left_ankle", "right_ankle"]
        if all(k in det.get("keypoints", []) for k in required_lower):
            metrics["frames_with_full_lower_body"] += 1
    
    total = len(detections) if detections else 1
    for kpt, count in keypoint_counts.items():
        metrics["keypoint_detection_rates"][kpt] = round(count / total, 3)
    
    return metrics


def analyze_video_with_backend(
    video_path: Path,
    backend_name: str,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run full gait analysis on video with specified backend.
    
    Returns:
        Dictionary with analysis results and metadata
    """
    result = {
        "backend": backend_name,
        "success": False,
        "error": None,
        "video_path": str(video_path),
        "frames_processed": 0,
        "frames_with_detection": 0,
        "gait_metrics": {},
        "output_video": None,
        "output_json": None,
    }
    
    is_available, missing = check_backend_available(backend_name)
    if not is_available:
        result["error"] = f"Missing dependencies: {', '.join(missing)}"
        return result
    
    try:
        backend_output = output_dir / backend_name
        backend_output.mkdir(parents=True, exist_ok=True)
        
        # Get model path for this backend
        model_path = get_model_path(backend_name)
        
        # Check if model file exists for backends that require it
        if model_path and backend_name in ("yolo_lower",):
            model_file = Path(model_path)
            if not model_file.exists():
                result["error"] = (
                    f"Model file not found: {model_path}. "
                    f"Set GAIT_YOLO_LOWER_MODEL env var or use --model-path"
                )
                return result
        
        # Create backend with model path
        backend_kwargs = {}
        if model_path:
            backend_kwargs["model_path"] = model_path
        
        backend = create_backend(backend_name, **backend_kwargs)
        actual_skeleton = backend.skeleton_name
        schema_mapper = KeypointSchemaMapper(actual_skeleton)
        
        # Determine keypoint focus from actual skeleton, not hardcoded config
        # This handles cases like AlphaPose falling back to YOLO (coco17)
        actual_focus = get_keypoint_focus_for_skeleton(actual_skeleton, backend_name)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            result["error"] = f"Could not open video: {video_path}"
            return result
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_video_path = backend_output / "annotated_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path), fourcc, fps, (width, height)
        )
        
        frame_number = 0
        all_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp_ms = (frame_number / fps) * 1000 if fps > 0 else 0
            pose_result = backend.predict(frame)
            
            annotated = frame.copy()
            
            if pose_result.num_persons > 0:
                result["frames_with_detection"] += 1
                
                canonical = schema_mapper.map_pose_result(
                    pose_result, 
                    person_idx=0,
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms
                )
                
                if canonical:
                    available_kpts = canonical.get_available_keypoints()
                    all_detections.append({
                        "frame": frame_number,
                        "keypoints": available_kpts,
                        "has_feet": canonical.has_feet()
                    })
                    annotated = draw_skeleton_on_frame(frame, canonical, backend_name)
            
            cv2.putText(
                annotated, 
                f"Backend: {backend_name} | {actual_focus['description']}", 
                (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.putText(
                annotated, 
                f"Frame: {frame_number}/{total_frames}", 
                (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )
            
            writer.write(annotated)
            frame_number += 1
            
            if frame_number % 100 == 0:
                print(f"  Processing frame {frame_number}/{total_frames}...", end="\r")
        
        cap.release()
        writer.release()
        
        result["frames_processed"] = frame_number
        result["output_video"] = str(output_video_path)
        
        gait_metrics = compute_gait_metrics(all_detections, actual_focus, fps)
        result["gait_metrics"] = gait_metrics
        
        output_json_path = backend_output / "gait_analysis.json"
        analysis_output = {
            "backend": backend_name,
            "video": str(video_path),
            "frames_processed": frame_number,
            "frames_with_detection": result["frames_with_detection"],
            "detection_rate": result["frames_with_detection"] / frame_number if frame_number > 0 else 0,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "keypoint_focus": actual_focus,
            "actual_skeleton": actual_skeleton,
            "gait_metrics": gait_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(output_json_path, "w") as f:
            json.dump(analysis_output, f, indent=2, default=str)
        
        result["output_json"] = str(output_json_path)
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def output_dir():
    """Fixture providing shared output directory for all tests."""
    return get_output_directory()


@pytest.fixture(scope="module")
def sample_video():
    """Fixture providing path to sample video."""
    video_path = find_sample_video()
    if video_path is None:
        pytest.skip(
            "No sample video found. Please place a video file at: "
            "data/sample_video.mp4"
        )
    return video_path


@pytest.fixture(scope="module")
def compatibility_results(output_dir):
    """Fixture to collect results from all backend tests."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "output_directory": str(output_dir),
        "backends": {}
    }
    yield results
    
    report_path = output_dir / "compatibility_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nCompatibility report saved to: {report_path}")


# ============================================================================
# Test Class
# ============================================================================

class TestCompatibility:
    """Compatibility tests for all pose estimation backends."""
    
    @pytest.fixture(autouse=True)
    def setup(self, output_dir, sample_video, compatibility_results):
        """Setup for each test."""
        self.output_dir = output_dir
        self.sample_video = sample_video
        self.results = compatibility_results
    
    def _run_backend_test(self, backend_name: str) -> Dict[str, Any]:
        """Run test for a specific backend."""
        print(f"\n{'='*60}")
        print(f"Testing backend: {backend_name}")
        print(f"{'='*60}")
        
        result = analyze_video_with_backend(
            self.sample_video,
            backend_name,
            self.output_dir
        )
        
        self.results["backends"][backend_name] = result
        
        if result["success"]:
            print(f"  SUCCESS: {result['frames_processed']} frames processed")
            print(f"  Detection rate: {result['frames_with_detection']}/{result['frames_processed']}")
            print(f"  Output video: {result['output_video']}")
            print(f"  Output JSON: {result['output_json']}")
        else:
            print(f"  FAILED: {result['error']}")
        
        return result
    
    def test_yolov8_compatibility(self):
        """Test YOLOv8 COCO-17 backend."""
        result = self._run_backend_test("yolov8")
        
        if result["error"] and "Missing dependencies" in result["error"]:
            pytest.skip(result["error"])
        
        assert result["success"], f"YOLOv8 failed: {result['error']}"
        assert result["frames_processed"] > 0
        
        metrics = result["gait_metrics"]
        assert not metrics["has_foot_keypoints"], "COCO-17 should not have foot keypoints"
    
    def test_yolo_lower_compatibility(self):
        """Test YOLO Lower Body backend."""
        result = self._run_backend_test("yolo_lower")
        
        if result["error"] and "Missing dependencies" in result["error"]:
            pytest.skip(result["error"])
        
        assert result["success"], f"YOLO Lower failed: {result['error']}"
        
        metrics = result["gait_metrics"]
        assert metrics["has_foot_keypoints"], "Lower body model should have foot keypoints"


# ============================================================================
# CLI Entry Point
# ============================================================================

def run_compatibility_tests(
    video_path: Optional[Path] = None,
    backend_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Run compatibility tests programmatically."""
    if video_path is None:
        video_path = find_sample_video()
    
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(
            "No sample video found. Please provide a video with --video "
            "or place one at data/sample_video.mp4"
        )
    
    if output_dir is None:
        output_dir = get_output_directory()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sample video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    all_backends = list_backends(include_unavailable=True)
    
    if backend_names:
        backends_to_test = [b.name for b in all_backends if b.name in backend_names]
    else:
        backends_to_test = [b.name for b in all_backends]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "video": str(video_path),
        "output_directory": str(output_dir),
        "backends": {}
    }
    
    for backend_name in backends_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {backend_name}")
        print(f"{'='*60}")
        
        result = analyze_video_with_backend(video_path, backend_name, output_dir)
        results["backends"][backend_name] = result
        
        if result["success"]:
            print(f"  SUCCESS")
            print(f"  Video: {result['output_video']}")
            print(f"  JSON: {result['output_json']}")
        else:
            print(f"  FAILED: {result['error']}")
    
    report_path = output_dir / "compatibility_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("COMPATIBILITY TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Report: {report_path}")
    
    passed = sum(1 for r in results["backends"].values() if r["success"])
    total = len(results["backends"])
    print(f"Passed: {passed}/{total}")
    
    return results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run compatibility tests for all pose backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tests.test_compatibility
  python -m tests.test_compatibility --video data/sample_video.mp4
  python -m tests.test_compatibility --backends yolov8 yolo_lower
  python -m tests.test_compatibility --output-dir checks/custom
  
  # Specify model paths for specific backends
  python -m tests.test_compatibility --model-path yolo_lower=models/yolo_lower/best.pt

Environment Variables:
  GAIT_YOLO_LOWER_MODEL   Path to YOLO lower body model (default: models/yolo_lower/best.pt)
  GAIT_YOLOV8_MODEL       Path to YOLOv8 pose model
        """
    )
    
    parser.add_argument(
        "--video", "-v",
        type=Path,
        help="Path to sample video (default: data/sample_video.mp4)"
    )
    
    parser.add_argument(
        "--backends", "-b",
        nargs="+",
        help="Specific backends to test"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Override output directory"
    )
    
    parser.add_argument(
        "--model-path", "-m",
        action="append",
        metavar="BACKEND=PATH",
        help="Set model path for a backend (e.g., yolo_lower=models/yolo_lower/best.pt)"
    )
    
    args = parser.parse_args()
    
    # Process model path arguments
    if args.model_path:
        for model_spec in args.model_path:
            if "=" in model_spec:
                backend, path = model_spec.split("=", 1)
                set_model_path(backend.strip(), path.strip())
            else:
                print(f"Warning: Invalid model-path format: {model_spec}", file=sys.stderr)
                print("  Expected format: BACKEND=PATH (e.g., yolo_lower=models/yolo_lower/best.pt)", file=sys.stderr)
    
    try:
        results = run_compatibility_tests(
            video_path=args.video,
            backend_names=args.backends,
            output_dir=args.output_dir
        )
        
        failed = [k for k, v in results["backends"].items() if not v["success"]]
        if failed:
            print(f"\nFailed backends: {', '.join(failed)}")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
