"""
Tests for backend registry and factory.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from gait_analysis.backend_registry import (
    register_backend,
    list_backends,
    get_backend_info,
    check_backend_available,
    create_backend,
    get_available_backend_names,
    _BACKEND_REGISTRY,
    BackendInfo
)
from gait_analysis.pose_backend import (
    PoseResult,
    PersonPose,
    BasePoseBackend,
    DependencyError
)


class TestBackendInfo:
    """Tests for BackendInfo dataclass."""
    
    def test_creation(self):
        """Test creating BackendInfo."""
        info = BackendInfo(
            name="test",
            description="Test backend",
            skeleton_name="test_skeleton",
            has_feet=True,
            requires=["numpy"],
            extras_name="test"
        )
        
        assert info.name == "test"
        assert info.has_feet == True
        assert "numpy" in info.requires


class TestBackendRegistry:
    """Tests for backend registry functions."""
    
    def test_registry_has_default_backends(self):
        """Test that default backends are registered."""
        # YOLO backends should always be registered
        assert "yolov8" in _BACKEND_REGISTRY
        assert "yolo_coco" in _BACKEND_REGISTRY
        assert "yolo_lower" in _BACKEND_REGISTRY
    
    def test_list_backends_all(self):
        """Test listing all backends including unavailable."""
        backends = list_backends(include_unavailable=True)
        
        assert len(backends) > 0
        assert any(b.name == "yolov8" for b in backends)
    
    def test_get_backend_info(self):
        """Test getting info for a specific backend."""
        info = get_backend_info("yolov8")
        
        assert info is not None
        assert info.name == "yolov8"
        assert info.skeleton_name == "coco17"
        assert info.has_feet == False
    
    def test_get_backend_info_unknown(self):
        """Test getting info for unknown backend."""
        info = get_backend_info("nonexistent_backend")
        assert info is None
    
    def test_check_backend_available_numpy(self):
        """Test checking availability for backend requiring numpy."""
        # numpy should always be available in test environment
        is_available, missing = check_backend_available("yolov8")
        
        # Note: yolov8 requires ultralytics, which may or may not be installed
        # The test just checks the function works
        assert isinstance(is_available, bool)
        assert isinstance(missing, list)
    
    def test_check_backend_unknown(self):
        """Test checking unknown backend."""
        is_available, missing = check_backend_available("unknown_backend")
        assert is_available == False
        assert len(missing) > 0
    
    def test_get_available_backend_names(self):
        """Test getting list of available backends."""
        names = get_available_backend_names()
        
        assert isinstance(names, list)
        # At minimum, some backends should be available if their deps are installed


class TestCreateBackend:
    """Tests for backend creation."""
    
    def test_create_unknown_backend(self):
        """Test that creating unknown backend raises error."""
        with pytest.raises(ValueError) as exc_info:
            create_backend("nonexistent_backend")
        
        assert "Unknown backend" in str(exc_info.value)
    
    @patch("gait_analysis.backend_registry.check_backend_available")
    def test_create_backend_missing_deps(self, mock_check):
        """Test that missing dependencies raise DependencyError."""
        mock_check.return_value = (False, ["missing_package"])
        
        with pytest.raises(DependencyError) as exc_info:
            create_backend("yolov8")
        
        assert "missing_package" in str(exc_info.value)


class TestPoseResult:
    """Tests for PoseResult dataclass."""
    
    def test_empty_result(self):
        """Test creating empty result."""
        result = PoseResult.empty("coco17")
        
        assert result.num_persons == 0
        assert result.num_keypoints == 0
        assert result.skeleton_name == "coco17"
    
    def test_result_with_data(self):
        """Test creating result with pose data."""
        keypoints = np.random.rand(2, 17, 2)  # 2 people, 17 keypoints
        scores = np.random.rand(2, 17)
        bboxes = np.array([[0, 0, 100, 200], [50, 50, 150, 250]])
        
        result = PoseResult(
            keypoints=keypoints,
            scores=scores,
            bboxes=bboxes,
            skeleton_name="coco17"
        )
        
        assert result.num_persons == 2
        assert result.num_keypoints == 17
    
    def test_get_person(self):
        """Test extracting single person from result."""
        keypoints = np.random.rand(2, 17, 2)
        scores = np.random.rand(2, 17)
        bboxes = np.array([[0, 0, 100, 200], [50, 50, 150, 250]])
        
        result = PoseResult(
            keypoints=keypoints,
            scores=scores,
            bboxes=bboxes,
            skeleton_name="coco17"
        )
        
        person = result.get_person(0)
        assert person is not None
        assert person.keypoints.shape == (17, 2)
        
        # Out of bounds returns None
        person_none = result.get_person(5)
        assert person_none is None
    
    def test_filter_by_confidence(self):
        """Test filtering low-confidence keypoints."""
        keypoints = np.array([[[100, 100], [200, 200]]])
        scores = np.array([[0.9, 0.1]])  # Second keypoint low confidence
        bboxes = np.array([[0, 0, 300, 300]])
        
        result = PoseResult(
            keypoints=keypoints,
            scores=scores,
            bboxes=bboxes,
            skeleton_name="coco17"
        )
        
        filtered = result.filter_by_confidence(min_score=0.5)
        
        # Low-confidence keypoint should be zeroed
        assert filtered.keypoints[0, 1, 0] == 0
        assert filtered.keypoints[0, 1, 1] == 0
        # High-confidence keypoint should remain
        assert filtered.keypoints[0, 0, 0] == 100


class TestPersonPose:
    """Tests for PersonPose dataclass."""
    
    def test_get_keypoint(self):
        """Test getting single keypoint."""
        keypoints = np.array([[100, 100], [200, 200]])
        scores = np.array([0.9, 0.3])
        
        person = PersonPose(keypoints=keypoints, scores=scores)
        
        # High confidence keypoint
        kpt = person.get_keypoint(0, min_score=0.5)
        assert kpt == (100.0, 100.0)
        
        # Low confidence keypoint filtered
        kpt_low = person.get_keypoint(1, min_score=0.5)
        assert kpt_low is None
    
    def test_get_keypoint_with_conf(self):
        """Test getting keypoint with confidence score."""
        keypoints = np.array([[100, 100]])
        scores = np.array([0.85])
        
        person = PersonPose(keypoints=keypoints, scores=scores)
        
        x, y, conf = person.get_keypoint_with_conf(0)
        
        assert x == 100.0
        assert y == 100.0
        assert conf == 0.85


class TestBasePoseBackend:
    """Tests for BasePoseBackend abstract class."""
    
    def test_resolve_device_auto_no_cuda(self):
        """Test device resolution when CUDA not available."""
        
        class TestBackend(BasePoseBackend):
            @property
            def name(self):
                return "test"
            
            @property
            def skeleton_name(self):
                return "test"
            
            def predict(self, image):
                return PoseResult.empty("test")
        
        # Mock torch to not have CUDA
        with patch.dict('sys.modules', {'torch': MagicMock(cuda=MagicMock(is_available=lambda: False))}):
            backend = TestBackend(device="auto")
            # Should fall back to cpu
            assert backend.device in ("cpu", "auto")
    
    def test_bbox_from_keypoints(self):
        """Test bounding box computation from keypoints."""
        
        class TestBackend(BasePoseBackend):
            @property
            def name(self):
                return "test"
            
            @property
            def skeleton_name(self):
                return "test"
            
            def predict(self, image):
                return PoseResult.empty("test")
        
        backend = TestBackend()
        
        keypoints = np.array([[100, 100], [200, 200], [0, 0]])  # Third is invalid
        scores = np.array([0.9, 0.9, 0.0])
        
        bbox = backend._bbox_from_keypoints(keypoints, scores, min_score=0.5)
        
        # Should compute box from valid keypoints only
        assert bbox[0] < 100  # x1 with padding
        assert bbox[2] > 200  # x2 with padding


class TestDependencyError:
    """Tests for DependencyError."""
    
    def test_error_message(self):
        """Test error message formatting."""
        error = DependencyError(
            backend_name="test_backend",
            package="test_package",
            install_cmd="pip install test"
        )
        
        assert "test_backend" in str(error)
        assert "test_package" in str(error)
        assert "pip install test" in str(error)
