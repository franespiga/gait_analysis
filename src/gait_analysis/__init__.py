"""
Gait Analysis - Computer Vision based gait analysis using pose estimation.

Supports YOLO-based pose estimation backends:
- YOLOv8: Standard YOLOv8-pose with 17 COCO keypoints
- YOLO Lower Body: Fine-tuned model with heel/toe keypoints (10 points)

References:
- YOLO Lower Body: https://github.com/yankaizhao322/Fine-Tuned-YOLOv8-Pose-Lower-body-Keypoints
"""

__version__ = "0.2.0"

# Legacy detector (YOLO COCO only)
from .detector import KeypointDetector

# Core analysis
from .analyzer import GaitAnalyzer, StepType
from .visualizer import GaitVisualizer
from .angle_analyzer import (
    EnhancedGaitAnalyzer, 
    EnhancedGaitAnalysisResult,
    FootIdentifier,
    calculate_knee_angle,
)
from .advanced_visualizer import AdvancedGaitVisualizer

# Multi-backend support (legacy API)
from .keypoint_config import DetectorBackend, KeypointIndices, get_keypoint_config
from .base_detector import BaseDetector, GaitKeypoints
from .yolo_detector import YOLODetector, YOLOCocoDetector, YOLOLowerBodyDetector
from .heel_toe_analyzer import HeelToeGaitAnalyzer, HeelToeStep

# Unified pose backend interface (new API)
from .pose_backend import PoseResult, PersonPose, PoseBackend, BasePoseBackend, DependencyError
from .keypoint_schema import (
    CanonicalGaitKeypoints,
    KeypointSchemaMapper,
    create_schema_mapper,
    get_supported_skeletons,
    skeleton_has_feet,
)
from .backend_registry import (
    create_backend,
    list_backends,
    get_backend_info,
    check_backend_available,
    get_available_backend_names,
    print_backends_table,
)

# Production gait pipeline (IC/TO, metrics, inference)
from .gait_events import GaitEvent, ContactType, detect_ic_to, classify_contact, events_to_contact_types
from .gait_metrics import StepRecord, GaitSummary, build_step_records, compute_summary
from .smoothing import smooth_keypoints, fps_aware_window_length
from .inference_pipeline import (
    PipelineConfig,
    run_pipeline,
    select_main_person,
    raw_to_six_keypoints,
    events_to_dataframe,
    step_records_to_dataframe,
)

__all__ = [
    # Legacy
    "KeypointDetector", 
    "GaitAnalyzer",
    "StepType",
    "GaitVisualizer",
    # Enhanced analysis
    "EnhancedGaitAnalyzer",
    "EnhancedGaitAnalysisResult",
    "FootIdentifier",
    "calculate_knee_angle",
    "AdvancedGaitVisualizer",
    # Multi-backend (legacy API)
    "DetectorBackend",
    "KeypointIndices",
    "get_keypoint_config",
    "BaseDetector",
    "GaitKeypoints",
    "YOLODetector",
    "YOLOCocoDetector",
    "YOLOLowerBodyDetector",
    "HeelToeGaitAnalyzer",
    "HeelToeStep",
    # Unified pose backend (new API)
    "PoseResult",
    "PersonPose",
    "PoseBackend",
    "BasePoseBackend",
    "DependencyError",
    "CanonicalGaitKeypoints",
    "KeypointSchemaMapper",
    "create_schema_mapper",
    "get_supported_skeletons",
    "skeleton_has_feet",
    "create_backend",
    "list_backends",
    "get_backend_info",
    "check_backend_available",
    "get_available_backend_names",
    "print_backends_table",
    # Production pipeline
    "GaitEvent",
    "ContactType",
    "detect_ic_to",
    "classify_contact",
    "events_to_contact_types",
    "StepRecord",
    "GaitSummary",
    "build_step_records",
    "compute_summary",
    "smooth_keypoints",
    "fps_aware_window_length",
    "PipelineConfig",
    "run_pipeline",
    "select_main_person",
    "raw_to_six_keypoints",
    "events_to_dataframe",
    "step_records_to_dataframe",
]

