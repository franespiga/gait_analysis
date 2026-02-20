"""
Pose estimation backend implementations.

Each backend provides a unified PoseBackend interface for pose estimation
using different underlying models (YOLO, OpenPose, AlphaPose, etc.).

Available backends:
- yolov8_backend: YOLOv8 pose (COCO-17 and Lower Body variants)
- openpose_backend: CMU OpenPose (Body_25)
- pocketpose_backend: PocketPose lightweight models
- sdpose_backend: SDPose from HuggingFace
- alphapose_backend: AlphaPose with HALPE keypoints
"""

from typing import TYPE_CHECKING

# Lazy imports to avoid loading all dependencies
__all__ = [
    "YOLOv8PoseBackend",
    "YOLOv8LowerBodyBackend",
    "OpenPoseBackend",
    "PocketPoseBackend",
    "SDPoseBackend",
    "AlphaPoseBackend",
]


def __getattr__(name: str):
    """Lazy import of backend classes."""
    if name in ("YOLOv8PoseBackend", "YOLOv8LowerBodyBackend"):
        from .yolov8_backend import YOLOv8PoseBackend, YOLOv8LowerBodyBackend
        return locals()[name]
    elif name == "OpenPoseBackend":
        from .openpose_backend import OpenPoseBackend
        return OpenPoseBackend
    elif name == "PocketPoseBackend":
        from .pocketpose_backend import PocketPoseBackend
        return PocketPoseBackend
    elif name == "SDPoseBackend":
        from .sdpose_backend import SDPoseBackend
        return SDPoseBackend
    elif name == "AlphaPoseBackend":
        from .alphapose_backend import AlphaPoseBackend
        return AlphaPoseBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
