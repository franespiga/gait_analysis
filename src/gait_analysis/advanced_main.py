"""
Advanced gait analysis CLI with angle calculations and video output.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

from .analysis_output import get_analysis_run_dir
from .detector import KeypointDetector
from .angle_analyzer import EnhancedGaitAnalyzer
from .advanced_visualizer import (
    AdvancedGaitVisualizer,
    create_visualization_window,
    show_frame,
    destroy_windows
)


def get_timestamped_output_dir(base_dir: str, video_name: str) -> Path:
    """Create timestamped output directory: yyyymmdd_hhmm_videoname"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # Clean video name (remove extension and special characters)
    clean_name = Path(video_name).stem
    clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in clean_name)
    
    folder_name = f"{timestamp}_{clean_name}"
    output_dir = Path(base_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def analyze_video_advanced(
    video_path: str,
    output_dir: str = "results",
    show_visualization: bool = False,
    save_video: bool = True,
    model_name: str = "yolov8n-pose.pt",
    device: str = "auto"
) -> dict:
    """
    Advanced gait analysis with angle measurements and video output.
    
    Args:
        video_path: Path to input video file
        output_dir: Base directory for results
        show_visualization: Whether to display visualization window
        save_video: Whether to save annotated video
        model_name: YOLO model to use
        device: Inference device
        
    Returns:
        Analysis results as dictionary
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # output_dir is the run directory (all outputs go here)
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir_path}")
    
    # Initialize components
    print(f"Loading model: {model_name}")
    detector = KeypointDetector(model_name=model_name, device=device)
    analyzer = EnhancedGaitAnalyzer()
    visualizer = AdvancedGaitVisualizer(show_angles=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
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
            print(f"Warning: Could not initialize video writer for {output_video_path}")
            save_video = False
    
    # Create visualization window if needed
    window_name = None
    if show_visualization:
        window_name = create_visualization_window("Advanced Gait Analysis")
    
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
            
            # Detect keypoints
            keypoints = detector.detect_frame(frame, frame_number, timestamp_ms)
            
            # Analyze gait
            if keypoints is not None:
                analysis_state = analyzer.process_frame(keypoints)
            
            # Get YOLO results for visualization
            yolo_results = detector.get_results_with_visualization(frame)
            
            # Draw visualization
            annotated_frame = visualizer.draw_frame(
                frame, keypoints, analysis_state, yolo_results,
                frame_number, total_frames
            )
            
            # Write to output video
            if save_video:
                visualizer.write_frame(annotated_frame)
            
            # Show if enabled
            if show_visualization:
                if not show_frame(window_name, annotated_frame, wait_ms=1):
                    print("\nVisualization stopped by user")
                    break
            
            # Progress indicator
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
    
    # Save JSON results
    json_output_path = output_dir_path / f"{video_path.stem}_results.json"
    with open(json_output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved to: {json_output_path}")
    if save_video and output_video_path:
        print(f"Video saved to: {output_video_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ADVANCED GAIT ANALYSIS SUMMARY")
    print("=" * 60)
    
    summary = results_dict["summary"]
    metrics = results_dict["metrics"]
    
    print(f"\nTotal Steps: {summary['total_steps']}")
    print(f"  Correct (Heel Strike): {summary['correct_steps']} ({summary['correct_percentage']:.1f}%)")
    print(f"  Incorrect (Toe Walking): {summary['incorrect_steps']} ({summary['incorrect_percentage']:.1f}%)")
    print(f"  Suboptimal (Flat Foot): {summary['flat_foot_steps']} ({summary['flat_foot_percentage']:.1f}%)")
    
    def print_stats(label: str, stats: dict):
        """Helper to print statistics with std deviation."""
        if stats["count"] > 0:
            print(f"      {label}: mean={stats['mean']:.1f}, std={stats['std']:.1f}, "
                  f"min={stats['min']:.1f}, max={stats['max']:.1f}, median={stats['median']:.1f}")
    
    print("\n--- PER-FOOT STATISTICS ---")
    
    for foot in ["left_foot", "right_foot"]:
        foot_label = foot.replace("_", " ").title()
        print(f"\n{foot_label}:")
        
        for step_type in ["correct_steps", "incorrect_steps", "flat_foot_steps"]:
            step_label = step_type.replace("_", " ").title()
            step_data = metrics[foot][step_type]
            angles = step_data["angles"]
            stance = step_data["stance_time_ms"]
            knee = step_data["knee_flexion"]
            
            if angles["count"] > 0:
                print(f"  {step_label} ({angles['count']}):")
                print_stats("Contact Angle (°)", angles)
                print_stats("Stance Time (ms)", stance)
                print_stats("Knee at Contact (°)", knee["at_contact"])
                print_stats("Min Knee Flexion (°)", knee["min_during_stance"])
    
    print("\n--- OVERALL STATISTICS ---")
    overall = metrics["overall"]
    
    for step_type in ["correct_steps", "incorrect_steps", "flat_foot_steps"]:
        step_label = step_type.replace("_", " ").title()
        step_data = overall[step_type]
        angles = step_data["angles"]
        stance = step_data["stance_time_ms"]
        knee = step_data["knee_flexion"]
        
        if angles["count"] > 0:
            print(f"\n{step_label} ({angles['count']}):")
            print_stats("Contact Angle (°)", angles)
            print_stats("Stance Time (ms)", stance)
            print_stats("Knee at Contact (°)", knee["at_contact"])
            print_stats("Min Knee Flexion (°)", knee["min_during_stance"])
    
    print("\n" + "=" * 60)
    
    return results_dict


def main():
    """CLI entry point for advanced gait analysis."""
    parser = argparse.ArgumentParser(
        description="Advanced gait analysis with angle measurements and video output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gait-analyze-advanced video.mp4
  gait-analyze-advanced video.mp4 --show
  gait-analyze-advanced video.mp4 --output-dir my_results --show
  gait-analyze-advanced video.mp4 --no-video --show
  gait-analyze-advanced video.mp4 --model yolov8m-pose.pt --device cuda

Output:
  Results are saved to: <output-dir>/<timestamp>_<videoname>/
  - <videoname>_results.json : Analysis results with angle metrics
  - <videoname>_analyzed.mp4 : Annotated video with keypoints
        """
    )
    
    parser.add_argument(
        "video",
        help="Path to input video file"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="results",
        help="Run directory for results (default: analyses/CLI/YYYYMMDD_HHMM)"
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
        "--model",
        default="yolov8n-pose.pt",
        help="YOLO pose model to use (default: yolov8n-pose.pt)"
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference: auto, cpu, cuda, cuda:0, etc. (default: auto)"
    )
    
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = (
        str(get_analysis_run_dir(project_root, "CLI"))
        if args.output_dir == "results"
        else str(get_timestamped_output_dir(args.output_dir, Path(args.video).name))
    )

    try:
        analyze_video_advanced(
            video_path=args.video,
            output_dir=output_dir,
            show_visualization=args.show,
            save_video=not args.no_video,
            model_name=args.model,
            device=args.device
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

