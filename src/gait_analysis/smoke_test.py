"""
Smoke Test Script for Pose Backends.

Runs a quick test of each available backend to verify:
1. Backend can be instantiated
2. Model can process an image
3. Output has expected structure

Usage:
    gait-smoke-test
    gait-smoke-test --image test.jpg
    gait-smoke-test --backends yolov8 pocketpose
"""

import argparse
import sys
from typing import List, Optional
import numpy as np

from .backend_registry import (
    list_backends,
    create_backend,
    check_backend_available,
    BackendInfo
)
from .pose_backend import PoseResult
from .keypoint_schema import KeypointSchemaMapper, skeleton_has_feet


def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a simple test image (black with white rectangle)."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # Add a white rectangle (simulates a person silhouette)
    cv2 = None
    try:
        import cv2
        cv2.rectangle(image, (width//3, height//4), (2*width//3, 3*height//4), (255, 255, 255), -1)
    except ImportError:
        # Simple rectangle without OpenCV
        image[height//4:3*height//4, width//3:2*width//3] = 255
    return image


def load_test_image(image_path: str) -> np.ndarray:
    """Load an image from file."""
    try:
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    except ImportError:
        raise ImportError("OpenCV required to load images. Install with: pip install opencv-python")


def test_backend(
    backend_info: BackendInfo,
    test_image: np.ndarray,
    verbose: bool = True
) -> dict:
    """
    Test a single backend.
    
    Returns dict with:
        - success: bool
        - error: Optional[str]
        - result: PoseResult info if successful
    """
    result = {
        "backend": backend_info.name,
        "success": False,
        "error": None,
        "num_persons": 0,
        "num_keypoints": 0,
        "has_feet": False,
        "canonical_keypoints": []
    }
    
    try:
        # Check availability
        is_available, missing = check_backend_available(backend_info.name)
        if not is_available:
            result["error"] = f"Missing dependencies: {', '.join(missing)}"
            return result
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing: {backend_info.name}")
            print(f"Description: {backend_info.description}")
            print(f"Skeleton: {backend_info.skeleton_name}")
            print(f"{'='*60}")
        
        # Create backend
        if verbose:
            print("  Creating backend...", end=" ")
        
        backend = create_backend(backend_info.name)
        
        if verbose:
            print("OK")
        
        # Run prediction
        if verbose:
            print("  Running prediction...", end=" ")
        
        pose_result = backend.predict(test_image)
        
        if verbose:
            print("OK")
        
        # Check result
        result["num_persons"] = pose_result.num_persons
        result["num_keypoints"] = pose_result.num_keypoints
        
        if verbose:
            print(f"  Detected persons: {pose_result.num_persons}")
            print(f"  Keypoints per person: {pose_result.num_keypoints}")
            print(f"  Skeleton: {pose_result.skeleton_name}")
        
        # Test schema mapping
        if pose_result.num_persons > 0:
            if verbose:
                print("  Testing schema mapping...", end=" ")
            
            mapper = KeypointSchemaMapper(pose_result.skeleton_name)
            canonical = mapper.map_pose_result(pose_result)
            
            if canonical:
                result["has_feet"] = canonical.has_feet()
                result["canonical_keypoints"] = canonical.get_available_keypoints()
                
                if verbose:
                    print("OK")
                    print(f"  Has feet keypoints: {canonical.has_feet()}")
                    print(f"  Available canonical keypoints: {len(result['canonical_keypoints'])}")
        
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"FAILED: {e}")
    
    return result


def run_smoke_tests(
    image_path: Optional[str] = None,
    backend_names: Optional[List[str]] = None,
    verbose: bool = True
) -> List[dict]:
    """
    Run smoke tests on all available backends.
    
    Args:
        image_path: Optional path to test image
        backend_names: Optional list of specific backends to test
        verbose: Whether to print detailed output
        
    Returns:
        List of test results
    """
    # Get test image
    if image_path:
        test_image = load_test_image(image_path)
    else:
        test_image = create_test_image()
    
    # Get backends to test
    all_backends = list_backends(include_unavailable=True)
    
    if backend_names:
        backends_to_test = [b for b in all_backends if b.name in backend_names]
    else:
        backends_to_test = all_backends
    
    if verbose:
        print("\n" + "="*60)
        print("POSE BACKEND SMOKE TEST")
        print("="*60)
        print(f"Test image size: {test_image.shape[1]}x{test_image.shape[0]}")
        print(f"Backends to test: {len(backends_to_test)}")
    
    results = []
    for backend_info in backends_to_test:
        result = test_backend(backend_info, test_image, verbose)
        results.append(result)
    
    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        passed = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print(f"\nPassed: {len(passed)}/{len(results)}")
        
        if passed:
            print("\n✓ Working backends:")
            for r in passed:
                feet = "with feet" if r["has_feet"] else "no feet"
                print(f"  - {r['backend']}: {r['num_keypoints']} keypoints ({feet})")
        
        if failed:
            print("\n✗ Failed/unavailable backends:")
            for r in failed:
                print(f"  - {r['backend']}: {r['error']}")
        
        print("\n" + "="*60)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Smoke test for pose estimation backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gait-smoke-test                       # Test all available backends
  gait-smoke-test --image test.jpg      # Test with specific image
  gait-smoke-test --backends yolov8 pocketpose  # Test specific backends
  gait-smoke-test --quiet               # Minimal output
        """
    )
    
    parser.add_argument(
        "--image",
        help="Path to test image (default: synthetic test image)"
    )
    
    parser.add_argument(
        "--backends",
        nargs="+",
        help="Specific backends to test"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    results = run_smoke_tests(
        image_path=args.image,
        backend_names=args.backends,
        verbose=not args.quiet
    )
    
    # Exit with error if any tests failed
    if any(not r["success"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
