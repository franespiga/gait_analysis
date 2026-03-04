"""
Backend Registry and Factory.

Provides a unified way to instantiate pose estimation backends by name.
Handles dependency checking and provides helpful error messages when
optional dependencies are missing.

Usage:
    from gait_analysis.backend_registry import create_backend, list_backends
    
    # List available backends
    backends = list_backends()
    
    # Create a backend
    backend = create_backend("yolov8", model_path="yolov8n-pose.pt")
    result = backend.predict(image)
"""

from typing import Dict, List, Optional, Type, Callable, Any
from dataclasses import dataclass
import warnings

from .pose_backend import BasePoseBackend, PoseBackend, DependencyError


@dataclass
class BackendInfo:
    """Information about a registered backend."""
    name: str
    description: str
    skeleton_name: str
    has_feet: bool
    requires: List[str]  # Required Python packages
    extras_name: str     # pip extras name for installation
    factory: Optional[Callable[..., PoseBackend]] = None
    is_available: Optional[bool] = None


# Global registry
_BACKEND_REGISTRY: Dict[str, BackendInfo] = {}


def register_backend(
    name: str,
    description: str,
    skeleton_name: str,
    has_feet: bool,
    requires: List[str],
    extras_name: str,
    factory: Callable[..., PoseBackend]
) -> None:
    """
    Register a new pose backend.
    
    Args:
        name: Unique backend identifier
        description: Human-readable description
        skeleton_name: Keypoint schema name
        has_feet: Whether this backend provides foot keypoints
        requires: List of required Python packages
        extras_name: Name for pip extras installation
        factory: Function to create backend instance
    """
    _BACKEND_REGISTRY[name.lower()] = BackendInfo(
        name=name,
        description=description,
        skeleton_name=skeleton_name,
        has_feet=has_feet,
        requires=requires,
        extras_name=extras_name,
        factory=factory
    )


def check_backend_available(name: str) -> tuple:
    """
    Check if a backend is available (dependencies installed).
    
    Returns:
        (is_available: bool, missing_packages: List[str])
    """
    name = name.lower()
    if name not in _BACKEND_REGISTRY:
        return False, [f"Backend '{name}' not registered"]
    
    info = _BACKEND_REGISTRY[name]
    missing = []
    
    for package in info.requires:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, missing


def list_backends(include_unavailable: bool = False) -> List[BackendInfo]:
    """
    List all registered backends.
    
    Args:
        include_unavailable: If True, include backends with missing dependencies
        
    Returns:
        List of BackendInfo objects
    """
    result = []
    for name, info in _BACKEND_REGISTRY.items():
        is_available, _ = check_backend_available(name)
        info.is_available = is_available
        if include_unavailable or is_available:
            result.append(info)
    return result


def get_backend_info(name: str) -> Optional[BackendInfo]:
    """Get information about a specific backend."""
    name = name.lower()
    if name not in _BACKEND_REGISTRY:
        return None
    info = _BACKEND_REGISTRY[name]
    info.is_available, _ = check_backend_available(name)
    return info


def create_backend(name: str, **kwargs) -> PoseBackend:
    """
    Create a pose backend instance by name.
    
    Args:
        name: Backend name (e.g., 'yolov8', 'yolo_lower')
        **kwargs: Backend-specific configuration
        
    Returns:
        Configured PoseBackend instance
        
    Raises:
        ValueError: If backend not registered
        DependencyError: If required dependencies are missing
    """
    name = name.lower()
    
    if name not in _BACKEND_REGISTRY:
        available = list(_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Unknown backend: '{name}'. "
            f"Available backends: {available}"
        )
    
    info = _BACKEND_REGISTRY[name]
    
    # Check dependencies
    is_available, missing = check_backend_available(name)
    if not is_available:
        raise DependencyError(
            backend_name=name,
            package=", ".join(missing),
            install_cmd=f"pip install gait-analysis[{info.extras_name}]"
        )
    
    if info.factory is None:
        raise ValueError(f"Backend '{name}' has no factory function registered")
    
    return info.factory(**kwargs)


def get_available_backend_names() -> List[str]:
    """Get names of backends with all dependencies satisfied."""
    return [
        name for name in _BACKEND_REGISTRY.keys()
        if check_backend_available(name)[0]
    ]


# ============================================================================
# Backend Factory Functions
# ============================================================================

def _create_yolov8_backend(
    model_path: str = "yolov8n-pose.pt",
    device: str = "auto",
    conf_threshold: float = 0.3,
    **kwargs
) -> PoseBackend:
    """Create YOLOv8 pose backend."""
    from .backends.yolov8_backend import YOLOv8PoseBackend
    return YOLOv8PoseBackend(
        model_path=model_path,
        device=device,
        conf_threshold=conf_threshold
    )


def _create_yolo_lower_backend(
    model_path: str = "models/yolo_lower/best.pt",
    device: str = "auto",
    conf_threshold: float = 0.25,
    **kwargs
) -> PoseBackend:
    """Create YOLO Lower Body pose backend."""
    from .backends.yolov8_backend import YOLOv8LowerBodyBackend
    return YOLOv8LowerBodyBackend(
        model_path=model_path,
        device=device,
        conf_threshold=conf_threshold
    )


# ============================================================================
# Register All Backends (YOLO only)
# ============================================================================

def _register_all_backends():
    """Register all supported backends (YOLO-based only)."""
    
    # YOLOv8 (standard COCO-17)
    register_backend(
        name="yolov8",
        description="YOLOv8 Pose with COCO-17 keypoints (no feet)",
        skeleton_name="coco17",
        has_feet=False,
        requires=["ultralytics"],
        extras_name="yolo",
        factory=_create_yolov8_backend
    )
    
    # Alias for backward compatibility
    register_backend(
        name="yolo_coco",
        description="YOLOv8 Pose with COCO-17 keypoints (alias)",
        skeleton_name="coco17",
        has_feet=False,
        requires=["ultralytics"],
        extras_name="yolo",
        factory=_create_yolov8_backend
    )
    
    # YOLO Lower Body
    register_backend(
        name="yolo_lower",
        description="Fine-tuned YOLOv8 for lower body with heel/toe (10 keypoints)",
        skeleton_name="coco_lower10",
        has_feet=True,
        requires=["ultralytics"],
        extras_name="yolo",
        factory=_create_yolo_lower_backend
    )


# Register backends on module import
_register_all_backends()


# ============================================================================
# CLI Helper
# ============================================================================

def print_backends_table():
    """Print a formatted table of all backends."""
    all_backends = list_backends(include_unavailable=True)
    
    print("\n" + "=" * 80)
    print("AVAILABLE POSE BACKENDS")
    print("=" * 80)
    
    for info in all_backends:
        status = "✓" if info.is_available else "✗"
        feet = "Yes" if info.has_feet else "No"
        print(f"\n{status} {info.name}")
        print(f"  Description: {info.description}")
        print(f"  Skeleton: {info.skeleton_name} | Feet: {feet}")
        print(f"  Requires: {', '.join(info.requires)}")
        if not info.is_available:
            print(f"  Install: pip install gait-analysis[{info.extras_name}]")
    
    print("\n" + "=" * 80)
