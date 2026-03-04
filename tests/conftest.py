"""
Pytest configuration and shared fixtures for gait analysis tests.

Model paths can be configured via:
1. Environment variables:
   - GAIT_YOLO_LOWER_MODEL=models/yolo_lower/best.pt
   - GAIT_YOLOV8_MODEL=yolov8n-pose.pt
   
2. Pytest command line:
   pytest tests/test_compatibility.py --yolo-lower-model=models/yolo_lower/best.pt
"""

import os
import pytest
from pathlib import Path


def pytest_addoption(parser):
    """Add custom command line options for model paths."""
    parser.addoption(
        "--yolo-lower-model",
        action="store",
        default=None,
        help="Path to YOLO lower body model (e.g., models/yolo_lower/best.pt)"
    )
    parser.addoption(
        "--yolov8-model",
        action="store",
        default=None,
        help="Path to YOLOv8 pose model"
    )


def pytest_configure(config):
    """Configure pytest with custom markers and model paths."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "compatibility: marks compatibility tests that require sample video"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests that require external dependencies"
    )
    
    # Set model paths from command line options to environment variables
    # This allows test_compatibility.py to pick them up
    if config.getoption("--yolo-lower-model"):
        os.environ["GAIT_YOLO_LOWER_MODEL"] = config.getoption("--yolo-lower-model")
    
    if config.getoption("--yolov8-model"):
        os.environ["GAIT_YOLOV8_MODEL"] = config.getoption("--yolov8-model")


def pytest_collection_modifyitems(config, items):
    """Auto-mark compatibility tests."""
    for item in items:
        if "test_compatibility" in item.nodeid:
            item.add_marker(pytest.mark.compatibility)
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return data directory path."""
    return project_root / "data"


@pytest.fixture(scope="session")
def checks_dir(project_root):
    """Return checks output directory path."""
    checks = project_root / "checks"
    checks.mkdir(exist_ok=True)
    return checks


@pytest.fixture(scope="session")
def yolo_lower_model(request):
    """Get YOLO lower body model path."""
    return (
        request.config.getoption("--yolo-lower-model") or 
        os.environ.get("GAIT_YOLO_LOWER_MODEL") or
        "models/yolo_lower/best.pt"
    )
