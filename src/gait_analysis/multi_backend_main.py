"""
Multi-backend gait analysis CLI supporting different pose estimation models.

Supports:
1. YOLO COCO (default) - Standard YOLOv8-pose, ankle only
2. YOLO Lower Body - Fine-tuned model with heel/toe keypoints
3. OpenPose - CMU OpenPose with detailed foot keypoints

References:
- YOLO Lower Body: https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints
- OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2

from .keypoint_config import DetectorBackend
from .base_detector import BaseDetector, GaitKeypoints
from .yolo_detector import YOLOCocoDetector, YOLOLowerBodyDetector, create_yolo_detector
from .angle_analyzer import EnhancedGaitAnalyzer
from .heel_toe_analyzer import HeelToeGaitAnalyzer
from .advanced_visualizer import (
    AdvancedGaitVisualizer,
    create_visualization_window,
    show_frame,
    destroy_windows
)


def get_timestamped_output_dir(base_dir: str, video_name: str) -> Path:
    """Create timestamped output directory: yyyymmdd_hhmm_videoname"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    clean_name = Path(video_name).stem
    clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in clean_name)
    
    folder_name = f"{timestamp}_{clean_name}"
    output_dir = Path(base_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def create_detector(
    backend: str,
    model_path: str,
    device: str = "auto",
    openpose_path: Optional[str] = None
) -> BaseDetector:
    """
    Create detector based on backend selection.
    
    Args:
        backend: Backend name ('yolov8', 'yolo_coco', 'yolo_lower', 'openpose', 
                              'pocketpose', 'sdpose', 'alphapose')
        model_path: Path to model weights
        device: Inference device
        openpose_path: Path to OpenPose installation (for openpose backend)
        
    Returns:
        Configured detector instance
    """
    # Legacy/YOLO backends using existing detectors
    if backend in ("yolo_coco", "yolov8"):
        return YOLOCocoDetector(model_name=model_path, device=device)
    
    elif backend == "yolo_lower":
        return YOLOLowerBodyDetector(model_path=model_path, device=device)
    
    elif backend == "openpose":
        from .openpose_detector import OpenPoseDetector
        return OpenPoseDetector(openpose_path=openpose_path)
    
    # New backends using unified PoseBackend interface
    elif backend in ("pocketpose", "sdpose", "alphapose"):
        from .backend_registry import create_backend
        return UnifiedBackendAdapter(
            create_backend(backend, model_path=model_path, device=device)
        )
    
    else:
        available = ["yolov8", "yolo_coco", "yolo_lower", "openpose", "pocketpose", "sdpose", "alphapose"]
        raise ValueError(f"Unknown backend: {backend}. Available: {available}")


class UnifiedBackendAdapter(BaseDetector):
    """
    Adapter to use new PoseBackend interface with existing BaseDetector API.
    
    This allows the new backends (PocketPose, SDPose, AlphaPose) to be used
    with the existing gait analysis pipeline.
    """
    
    def __init__(self, pose_backend):
        """
        Initialize adapter with a PoseBackend instance.
        
        Args:
            pose_backend: A PoseBackend instance (PocketPose, SDPose, etc.)
        """
        from .keypoint_config import DetectorBackend, get_keypoint_config
        from .keypoint_schema import KeypointSchemaMapper
        
        self._pose_backend = pose_backend
        self._schema_mapper = KeypointSchemaMapper(
            pose_backend.skeleton_name,
            conf_threshold=pose_backend.conf_threshold
        )
        
        # Map skeleton name to DetectorBackend
        backend_map = {
            "coco17": DetectorBackend.YOLO_COCO,
            "coco_lower10": DetectorBackend.YOLO_LOWER_BODY,
            "body25": DetectorBackend.OPENPOSE,
            "wholebody133": DetectorBackend.POCKETPOSE,
            "halpe26": DetectorBackend.ALPHAPOSE_BODY,
            "halpe136": DetectorBackend.ALPHAPOSE,
        }
        
        skeleton = pose_backend.skeleton_name
        self._backend_enum = backend_map.get(skeleton, DetectorBackend.YOLO_COCO)
        self._keypoint_config = get_keypoint_config(self._backend_enum)
    
    @property
    def backend(self):
        return self._backend_enum
    
    @property
    def keypoint_config(self):
        return self._keypoint_config
    
    def detect(
        self,
        frame,
        frame_number: int,
        timestamp_ms: float,
        conf_threshold: float = 0.5
    ):
        """Detect keypoints using the unified backend."""
        from .base_detector import GaitKeypoints
        
        # Run pose estimation
        result = self._pose_backend.predict(frame)
        
        if result.num_persons == 0:
            return None
        
        # Map to canonical gait keypoints
        canonical = self._schema_mapper.map_pose_result(
            result,
            person_idx=0,
            frame_number=frame_number,
            timestamp_ms=timestamp_ms
        )
        
        if canonical is None:
            return None
        
        # Convert to GaitKeypoints (existing format)
        return GaitKeypoints(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            left_hip=canonical.left_hip,
            right_hip=canonical.right_hip,
            left_knee=canonical.left_knee,
            right_knee=canonical.right_knee,
            left_ankle=canonical.left_ankle,
            right_ankle=canonical.right_ankle,
            left_heel=canonical.left_heel,
            right_heel=canonical.right_heel,
            left_toe=canonical.left_toe,
            right_toe=canonical.right_toe,
            raw_keypoints=result.keypoints[0] if result.num_persons > 0 else None,
            confidences=canonical.confidences
        )
    
    def get_visualization_results(self, frame, conf_threshold: float = 0.5):
        """Get results for visualization."""
        return self._pose_backend.predict(frame)


def create_analyzer(detector: BaseDetector):
    """
    Create appropriate analyzer based on detector capabilities.
    
    If detector has heel/toe keypoints, uses HeelToeGaitAnalyzer.
    Otherwise, uses EnhancedGaitAnalyzer with ankle-based estimation.
    """
    if detector.has_heel_toe:
        return HeelToeGaitAnalyzer()
    else:
        return EnhancedGaitAnalyzer()


def analyze_video_multi_backend(
    video_path: str,
    output_dir: str = "results",
    show_visualization: bool = False,
    save_video: bool = True,
    backend: str = "yolo_coco",
    model_path: str = "yolov8n-pose.pt",
    device: str = "auto",
    openpose_path: Optional[str] = None
) -> dict:
    """
    Analyze gait with configurable pose estimation backend.
    
    Args:
        video_path: Path to input video file
        output_dir: Base directory for results
        show_visualization: Whether to display visualization window
        save_video: Whether to save annotated video
        backend: Pose estimation backend ('yolo_coco', 'yolo_lower', 'openpose')
        model_path: Path to model weights
        device: Inference device
        openpose_path: Path to OpenPose installation
        
    Returns:
        Analysis results as dictionary
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory
    output_dir_path = get_timestamped_output_dir(output_dir, video_path.name)
    print(f"Output directory: {output_dir_path}")
    
    # Create detector and analyzer
    print(f"Loading {backend} detector with model: {model_path}")
    detector = create_detector(backend, model_path, device, openpose_path)
    analyzer = create_analyzer(detector)
    visualizer = AdvancedGaitVisualizer(show_angles=True)
    
    print(f"Detector backend: {detector.backend.value}")
    print(f"Has heel/toe keypoints: {detector.has_heel_toe}")
    print(f"Model path: {model_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")
    
    # Setup output video
    output_video_path = None
    if save_video:
        output_video_path = output_dir_path / f"{video_path.stem}_analyzed.mp4"
        if not visualizer.init_video_writer(str(output_video_path), fps, width, height):
            print(f"Warning: Could not initialize video writer")
            save_video = False
    
    # Visualization window
    window_name = None
    if show_visualization:
        window_name = create_visualization_window(f"Gait Analysis ({backend})")
    
    # Process frames
    frame_number = 0
    analysis_state = None
    
    print("Processing video...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp_ms = (frame_number / fps) * 1000 if fps > 0 else 0
            
            # Detect keypoints (use lower confidence for lower body model)
            conf_thresh = 0.25 if backend == "yolo_lower" else 0.5
            keypoints = detector.detect(frame, frame_number, timestamp_ms, conf_threshold=conf_thresh)
            
            # Debug: print raw detection info for first few frames
            if frame_number < 3:
                # Check raw model output before our filtering
                raw_results = detector.model(frame, verbose=False, device=detector.device, conf=0.1)
                print(f"\n--- Frame {frame_number} Debug ---")
                if len(raw_results) > 0 and raw_results[0].keypoints is not None:
                    raw_kpts = raw_results[0].keypoints
                    print(f"Raw keypoints shape: xy={raw_kpts.xy.shape if raw_kpts.xy is not None else None}")
                    if raw_kpts.xy is not None and len(raw_kpts.xy) > 0:
                        print(f"Number of people detected: {len(raw_kpts.xy)}")
                        for person_idx, person_kpts in enumerate(raw_kpts.xy):
                            print(f"  Person {person_idx}: {len(person_kpts)} keypoints")
                            if raw_kpts.conf is not None:
                                confs = raw_kpts.conf[person_idx].cpu().numpy()
                                print(f"    Confidences: {[f'{c:.2f}' for c in confs]}")
                else:
                    print(f"No detections in raw results")
                    print(f"Results object: {raw_results}")
                    if len(raw_results) > 0:
                        print(f"  boxes: {raw_results[0].boxes}")
                        print(f"  keypoints: {raw_results[0].keypoints}")
            
            # Analyze gait
            if keypoints is not None:
                analysis_state = analyzer.process_frame(keypoints)
            
            # Get visualization
            viz_results = detector.get_visualization_results(frame)
            
            # For OpenPose, get rendered frame directly
            if backend == "openpose" and viz_results is not None:
                annotated_frame = viz_results.cvOutputData if hasattr(viz_results, 'cvOutputData') else frame
            else:
                # Use our custom visualizer for YOLO
                from .detector import FrameKeypoints
                # Convert GaitKeypoints to FrameKeypoints for visualizer compatibility
                frame_kpts = None
                if keypoints is not None and keypoints.raw_keypoints is not None:
                    frame_kpts = FrameKeypoints(
                        frame_number=keypoints.frame_number,
                        timestamp_ms=keypoints.timestamp_ms,
                        keypoints={},  # Not used by visualizer
                        raw_keypoints=keypoints.raw_keypoints
                    )
                
                annotated_frame = visualizer.draw_frame(
                    frame, frame_kpts, analysis_state, viz_results,
                    frame_number, total_frames, backend=backend
                )
            
            # Write to output
            if save_video:
                visualizer.write_frame(annotated_frame)
            
            # Show if enabled
            if show_visualization:
                if not show_frame(window_name, annotated_frame, wait_ms=1):
                    print("\nVisualization stopped by user")
                    break
            
            # Progress
            if frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"\rProgress: {progress:.1f}%", end="", flush=True)
            
            frame_number += 1
    
    finally:
        cap.release()
        visualizer.release_video_writer()
        if show_visualization:
            destroy_windows()
    
    print(f"\nProcessed {frame_number} frames")
    
    # Get results
    results = analyzer.get_results(
        str(video_path),
        frame_number,
        fps,
        str(output_video_path) if output_video_path else None
    )
    results_dict = results.to_dict()
    
    # Add backend info to results
    results_dict["detector_info"] = {
        "backend": backend,
        "model_path": model_path,
        "has_heel_toe": detector.has_heel_toe
    }
    
    # Save JSON
    json_output_path = output_dir_path / f"{video_path.stem}_results.json"
    with open(json_output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved to: {json_output_path}")
    if save_video and output_video_path:
        print(f"Video saved to: {output_video_path}")
    
    # Print summary
    _print_summary(results_dict, backend, detector.has_heel_toe)
    
    return results_dict


def _print_summary(results_dict: dict, backend: str, has_heel_toe: bool):
    """Print analysis summary."""
    print("\n" + "=" * 60)
    print(f"GAIT ANALYSIS SUMMARY (Backend: {backend})")
    if has_heel_toe:
        print("  Using actual heel/toe keypoints for classification")
    else:
        print("  Using ankle-based trajectory estimation")
    print("=" * 60)
    
    summary = results_dict["summary"]
    metrics = results_dict["metrics"]
    
    print(f"\nTotal Steps: {summary['total_steps']}")
    print(f"  Correct (Heel Strike): {summary['correct_steps']} ({summary['correct_percentage']:.1f}%)")
    print(f"  Incorrect (Toe Walking): {summary['incorrect_steps']} ({summary['incorrect_percentage']:.1f}%)")
    print(f"  Suboptimal (Flat Foot): {summary['flat_foot_steps']} ({summary['flat_foot_percentage']:.1f}%)")
    
    def print_stats(label: str, stats: dict):
        if stats["count"] > 0:
            print(f"      {label}: mean={stats['mean']:.1f}, std={stats['std']:.1f}, "
                  f"min={stats['min']:.1f}, max={stats['max']:.1f}")
    
    print("\n--- PER-FOOT STATISTICS ---")
    
    for foot in ["left_foot", "right_foot"]:
        foot_label = foot.replace("_", " ").title()
        print(f"\n{foot_label}:")
        
        for step_type in ["correct_steps", "incorrect_steps", "flat_foot_steps"]:
            step_data = metrics[foot][step_type]
            angles = step_data["angles"]
            
            if angles["count"] > 0:
                step_label = step_type.replace("_", " ").title()
                print(f"  {step_label} ({angles['count']}):")
                print_stats("Contact Angle (°)", angles)
                print_stats("Stance Time (ms)", step_data["stance_time_ms"])
                if "knee_flexion" in step_data:
                    print_stats("Knee at Contact (°)", step_data["knee_flexion"]["at_contact"])
    
    print("\n" + "=" * 60)


def main():
    """CLI entry point for multi-backend gait analysis."""
    parser = argparse.ArgumentParser(
        description="Gait analysis with multiple pose estimation backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backends:
  yolov8      Standard YOLOv8-pose with COCO keypoints (17 points, ankle only)
  yolo_coco   Alias for yolov8
  yolo_lower  Fine-tuned YOLOv8 for lower body (10 points with heel/toe)
  openpose    CMU OpenPose Body_25 (25 points with detailed foot keypoints)
  pocketpose  PocketPose ONNX whole-body (133 points with feet)
  sdpose      SDPose from HuggingFace (133 whole-body keypoints)
  alphapose   AlphaPose with HALPE-136 whole-body keypoints

Examples:
  # Standard YOLO (default)
  gait-analyze-multi video.mp4
  
  # YOLO Lower Body with custom model
  gait-analyze-multi video.mp4 --backend yolo_lower --model best.pt
  
  # OpenPose (requires OPENPOSE_PATH env var or --openpose-path)
  gait-analyze-multi video.mp4 --backend openpose --openpose-path C:/openpose
  
  # PocketPose (lightweight ONNX model)
  gait-analyze-multi video.mp4 --backend pocketpose
  
  # AlphaPose whole-body
  gait-analyze-multi video.mp4 --backend alphapose

Backends with foot keypoints (heel/toe) provide more accurate gait analysis:
  yolo_lower, openpose, pocketpose, sdpose, alphapose

Install backend dependencies with:
  pip install gait-analysis[yolo]      # YOLOv8 (default)
  pip install gait-analysis[pocketpose] # PocketPose  
  pip install gait-analysis[sdpose]    # SDPose
  pip install gait-analysis[alphapose] # AlphaPose
  pip install gait-analysis[all]       # All backends

List available backends:
  gait-backends
        """
    )
    
    parser.add_argument("video", help="Path to input video file")
    
    parser.add_argument(
        "-o", "--output-dir",
        default="results",
        help="Base directory for results (default: results)"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualization window during analysis"
    )
    
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Don't save annotated output video"
    )
    
    parser.add_argument(
        "--backend",
        choices=["yolov8", "yolo_coco", "yolo_lower", "openpose", "pocketpose", "sdpose", "alphapose"],
        default="yolov8",
        help="Pose estimation backend (default: yolov8)"
    )
    
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="Model path/name (default: yolov8n-pose.pt for YOLO)"
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda (default: auto)"
    )
    
    parser.add_argument(
        "--openpose-path",
        help="Path to OpenPose installation (required for openpose backend)"
    )
    
    args = parser.parse_args()
    
    # Validate OpenPose path
    if args.backend == "openpose" and not args.openpose_path:
        import os
        if not os.environ.get("OPENPOSE_PATH"):
            print("Error: OpenPose backend requires --openpose-path or OPENPOSE_PATH env variable",
                  file=sys.stderr)
            sys.exit(1)
    
    try:
        analyze_video_multi_backend(
            video_path=args.video,
            output_dir=args.output_dir,
            show_visualization=args.show,
            save_video=not args.no_video,
            backend=args.backend,
            model_path=args.model,
            device=args.device,
            openpose_path=args.openpose_path
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

