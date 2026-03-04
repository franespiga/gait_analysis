"""
Pose estimation backend implementations.

Each backend provides a unified PoseBackend interface for pose estimation
using YOLO-based models.

Available backends:
- yolov8_backend: YOLOv8 pose (COCO-17 and Lower Body variants)
"""

from typing import TYPE_CHECKING

# Lazy imports to avoid loading all dependencies
__all__ = [
    "YOLOv8PoseBackend",
    "YOLOv8LowerBodyBackend",
]


def __getattr__(name: str):
    """Lazy import of backend classes."""
    if name in ("YOLOv8PoseBackend", "YOLOv8LowerBodyBackend"):
        from .yolov8_backend import YOLOv8PoseBackend, YOLOv8LowerBodyBackend
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
