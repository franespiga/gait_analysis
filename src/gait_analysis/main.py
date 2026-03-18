"""
Main entry point for gait analysis CLI.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from .analysis_output import get_analysis_run_dir
from .detector import KeypointDetector
from .analyzer import GaitAnalyzer
from .visualizer import (
    GaitVisualizer, 
    create_visualization_window, 
    show_frame, 
    destroy_windows
)


def analyze_video(
    video_path: str,
    output_path: str = None,
    show_visualization: bool = False,
    model_name: str = "yolov8n-pose.pt",
    device: str = "auto"
) -> dict:
    """
    Analyze gait patterns in a video.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save JSON results (optional)
        show_visualization: Whether to display visualization window
        model_name: YOLO model to use
        device: Inference device
        
    Returns:
        Analysis results as dictionary
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Initialize components
    print(f"Loading model: {model_name}")
    detector = KeypointDetector(model_name=model_name, device=device)
    analyzer = GaitAnalyzer()
    visualizer = GaitVisualizer() if show_visualization else None
    
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
    
    # Create visualization window if needed
    window_name = None
    if show_visualization:
        window_name = create_visualization_window("Gait Analysis")
    
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
            
            # Visualize if enabled
            if show_visualization and visualizer is not None:
                # Get YOLO results for visualization
                yolo_results = detector.get_results_with_visualization(frame)
                
                annotated_frame = visualizer.draw_frame(
                    frame, keypoints, analysis_state, yolo_results
                )
                
                if not show_frame(window_name, annotated_frame, wait_ms=1):
                    print("Visualization stopped by user")
                    break
            
            # Progress indicator
            if frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"\rProgress: {progress:.1f}%", end="", flush=True)
            
            frame_number += 1
    
    finally:
        cap.release()
        if show_visualization:
            destroy_windows()
    
    print(f"\nProcessed {frame_number} frames")
    
    # Get results
    results = analyzer.get_results(str(video_path), frame_number, fps)
    results_dict = results.to_dict()
    
    # Save to JSON: default is analyses/CLI/YYYYMMDD_HHMM/<stem>.json
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        _project_root = Path(__file__).resolve().parent.parent.parent
        run_dir = get_analysis_run_dir(_project_root, "CLI")
        output_path = run_dir / f"{video_path.stem}.json"
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("GAIT ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total Steps: {results.total_steps}")
    print(f"Correct Steps (Heel Strike): {results.correct_steps}")
    print(f"Incorrect Steps (Toe Walking): {results.incorrect_steps}")
    print(f"Correct Percentage: {results.correct_percentage:.1f}%")
    print(f"Average Step Time: {results.avg_step_time_ms:.1f}ms")
    print(f"  - Correct Steps: {results.avg_correct_step_time_ms:.1f}ms")
    print(f"  - Incorrect Steps: {results.avg_incorrect_step_time_ms:.1f}ms")
    print("="*50)
    
    return results_dict


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze gait patterns in video using YOLO pose estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gait-analyze video.mp4
  gait-analyze video.mp4 --show
  gait-analyze video.mp4 --output results.json --show
  gait-analyze video.mp4 --model yolov8m-pose.pt --device cuda
        """
    )
    
    parser.add_argument(
        "video",
        help="Path to input video file"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path to output JSON file (default: same as video with .json extension)"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualization window during analysis"
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
    
    try:
        analyze_video(
            video_path=args.video,
            output_path=args.output,
            show_visualization=args.show,
            model_name=args.model,
            device=args.device
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

