"""
Tests for keypoint schema mapping.
"""

import pytest
import numpy as np

from gait_analysis.keypoint_schema import (
    SkeletonType,
    CanonicalGaitKeypoints,
    KeypointSchemaMapper,
    create_schema_mapper,
    get_supported_skeletons,
    skeleton_has_feet,
    SKELETON_MAPPINGS_BY_NAME,
    COCO_17_INDICES,
    BODY_25_INDICES,
    HALPE_26_INDICES,
    WHOLEBODY_133_INDICES,
)
from gait_analysis.pose_backend import PoseResult, PersonPose


class TestSkeletonType:
    """Tests for SkeletonType enum."""
    
    def test_enum_values(self):
        """Test enum has expected values."""
        assert SkeletonType.COCO_17.value == "coco17"
        assert SkeletonType.BODY_25.value == "body25"
        assert SkeletonType.HALPE_26.value == "halpe26"
        assert SkeletonType.HALPE_136.value == "halpe136"
        assert SkeletonType.WHOLEBODY_133.value == "wholebody133"


class TestCanonicalGaitKeypoints:
    """Tests for CanonicalGaitKeypoints dataclass."""
    
    def test_basic_creation(self):
        """Test creating canonical keypoints."""
        kpts = CanonicalGaitKeypoints(
            left_hip=(100, 50),
            right_hip=(200, 50),
            left_knee=(100, 100),
            right_knee=(200, 100),
            left_ankle=(100, 150),
            right_ankle=(200, 150)
        )
        
        assert kpts.left_hip == (100, 50)
        assert kpts.left_ankle == (100, 150)
    
    def test_has_feet_false(self):
        """Test has_feet when no foot keypoints."""
        kpts = CanonicalGaitKeypoints(
            left_ankle=(100, 150),
            right_ankle=(200, 150)
        )
        
        assert not kpts.has_feet()
    
    def test_has_feet_true(self):
        """Test has_feet when foot keypoints present."""
        kpts = CanonicalGaitKeypoints(
            left_ankle=(100, 150),
            right_ankle=(200, 150),
            left_heel=(100, 160),
            right_heel=(200, 160),
            left_toe=(110, 165),
            right_toe=(210, 165)
        )
        
        assert kpts.has_feet()
    
    def test_has_detailed_feet(self):
        """Test has_detailed_feet with small toe keypoints."""
        kpts = CanonicalGaitKeypoints(
            left_ankle=(100, 150),
            left_heel=(100, 160),
            left_toe=(110, 165),
            left_small_toe=(95, 165)
        )
        
        assert kpts.has_detailed_feet()
    
    def test_get_available_keypoints(self):
        """Test getting list of available keypoints."""
        kpts = CanonicalGaitKeypoints(
            left_hip=(100, 50),
            left_knee=(100, 100),
            left_ankle=(100, 150)
        )
        
        available = kpts.get_available_keypoints()
        
        assert "left_hip" in available
        assert "left_knee" in available
        assert "left_ankle" in available
        assert "right_hip" not in available  # Not set
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        kpts = CanonicalGaitKeypoints(
            left_hip=(100, 50),
            left_ankle=(100, 150),
            source_skeleton="coco17"
        )
        
        result = kpts.to_dict()
        
        assert result["left_hip"] == (100, 50)
        assert result["source_skeleton"] == "coco17"
        assert "available_keypoints" in result


class TestSkeletonMappings:
    """Tests for skeleton index mappings."""
    
    def test_coco17_no_feet(self):
        """Test COCO-17 has no foot keypoints."""
        indices = COCO_17_INDICES
        
        assert "left_hip" in indices
        assert "left_ankle" in indices
        assert "left_heel" not in indices
        assert "left_toe" not in indices
    
    def test_body25_has_feet(self):
        """Test Body_25 has foot keypoints."""
        indices = BODY_25_INDICES
        
        assert "left_heel" in indices
        assert "right_heel" in indices
        assert "left_toe" in indices
        assert "left_small_toe" in indices
    
    def test_halpe26_has_feet(self):
        """Test HALPE-26 has foot keypoints."""
        indices = HALPE_26_INDICES
        
        assert "left_heel" in indices
        assert "left_toe" in indices
        assert indices["left_heel"] == 19
    
    def test_wholebody133_has_feet(self):
        """Test WholeBody-133 has foot keypoints."""
        indices = WHOLEBODY_133_INDICES
        
        assert "left_heel" in indices
        assert "left_toe" in indices
        assert "left_small_toe" in indices
        # Check foot keypoint indices (17-22 in COCO-WholeBody)
        assert indices["left_toe"] == 17  # L_BigToe
        assert indices["left_heel"] == 19


class TestKeypointSchemaMapper:
    """Tests for KeypointSchemaMapper class."""
    
    def test_init_valid_skeleton(self):
        """Test initialization with valid skeleton."""
        mapper = KeypointSchemaMapper("coco17")
        
        assert mapper.source_skeleton == "coco17"
        assert "left_hip" in mapper.indices
    
    def test_init_invalid_skeleton(self):
        """Test initialization with invalid skeleton raises error."""
        with pytest.raises(ValueError) as exc_info:
            KeypointSchemaMapper("invalid_skeleton")
        
        assert "Unknown skeleton type" in str(exc_info.value)
    
    def test_has_feet_mapping(self):
        """Test checking if skeleton has feet mapping."""
        mapper_coco = KeypointSchemaMapper("coco17")
        assert not mapper_coco.has_feet_mapping
        
        mapper_body25 = KeypointSchemaMapper("body25")
        assert mapper_body25.has_feet_mapping
    
    def test_map_pose_result_coco17(self):
        """Test mapping COCO-17 PoseResult to canonical."""
        # Create mock pose result with COCO-17 keypoints
        keypoints = np.zeros((1, 17, 2), dtype=np.float32)
        scores = np.ones((1, 17), dtype=np.float32) * 0.9
        
        # Set some keypoints
        keypoints[0, 11] = [100, 50]   # left_hip
        keypoints[0, 12] = [200, 50]   # right_hip
        keypoints[0, 15] = [100, 150]  # left_ankle
        keypoints[0, 16] = [200, 150]  # right_ankle
        
        result = PoseResult(
            keypoints=keypoints,
            scores=scores,
            bboxes=np.array([[0, 0, 300, 200]]),
            skeleton_name="coco17"
        )
        
        mapper = KeypointSchemaMapper("coco17")
        canonical = mapper.map_pose_result(result)
        
        assert canonical is not None
        assert canonical.left_hip == (100.0, 50.0)
        assert canonical.left_ankle == (100.0, 150.0)
        assert canonical.left_heel is None  # COCO-17 has no heel
    
    def test_map_pose_result_body25(self):
        """Test mapping Body_25 PoseResult to canonical."""
        keypoints = np.zeros((1, 25, 2), dtype=np.float32)
        scores = np.ones((1, 25), dtype=np.float32) * 0.9
        
        # Set keypoints according to Body_25 indices
        keypoints[0, 12] = [100, 50]   # left_hip
        keypoints[0, 14] = [100, 150]  # left_ankle
        keypoints[0, 21] = [100, 160]  # left_heel
        keypoints[0, 19] = [110, 165]  # left_toe
        
        result = PoseResult(
            keypoints=keypoints,
            scores=scores,
            bboxes=np.array([[0, 0, 300, 200]]),
            skeleton_name="body25"
        )
        
        mapper = KeypointSchemaMapper("body25")
        canonical = mapper.map_pose_result(result)
        
        assert canonical is not None
        assert canonical.left_hip == (100.0, 50.0)
        assert canonical.left_heel == (100.0, 160.0)
        assert canonical.left_toe == (110.0, 165.0)
    
    def test_map_pose_result_empty(self):
        """Test mapping empty PoseResult."""
        result = PoseResult.empty("coco17")
        
        mapper = KeypointSchemaMapper("coco17")
        canonical = mapper.map_pose_result(result)
        
        assert canonical is None
    
    def test_map_low_confidence_filtered(self):
        """Test that low-confidence keypoints are filtered."""
        keypoints = np.zeros((1, 17, 2), dtype=np.float32)
        scores = np.zeros((1, 17), dtype=np.float32)
        
        # High confidence hip
        keypoints[0, 11] = [100, 50]
        scores[0, 11] = 0.9
        
        # Low confidence ankle
        keypoints[0, 15] = [100, 150]
        scores[0, 15] = 0.1
        
        result = PoseResult(
            keypoints=keypoints,
            scores=scores,
            bboxes=np.array([[0, 0, 300, 200]]),
            skeleton_name="coco17"
        )
        
        mapper = KeypointSchemaMapper("coco17", conf_threshold=0.5)
        canonical = mapper.map_pose_result(result)
        
        assert canonical.left_hip == (100.0, 50.0)  # High confidence
        assert canonical.left_ankle is None  # Low confidence filtered
    
    def test_map_raw_keypoints(self):
        """Test mapping raw keypoint arrays."""
        keypoints = np.zeros((17, 2), dtype=np.float32)
        scores = np.ones(17, dtype=np.float32) * 0.9
        
        keypoints[11] = [100, 50]  # left_hip
        keypoints[15] = [100, 150] # left_ankle
        
        mapper = KeypointSchemaMapper("coco17")
        canonical = mapper.map_raw_keypoints(
            keypoints=keypoints,
            scores=scores,
            frame_number=10,
            timestamp_ms=333.3
        )
        
        assert canonical.left_hip == (100.0, 50.0)
        assert canonical.frame_number == 10
        assert canonical.timestamp_ms == 333.3
    
    def test_get_expected_keypoints(self):
        """Test getting list of expected keypoint names."""
        mapper = KeypointSchemaMapper("body25")
        expected = mapper.get_expected_keypoints()
        
        assert "left_hip" in expected
        assert "left_heel" in expected
        assert "left_small_toe" in expected


class TestHelperFunctions:
    """Tests for module-level helper functions."""
    
    def test_create_schema_mapper(self):
        """Test factory function."""
        mapper = create_schema_mapper("coco17", conf_threshold=0.4)
        
        assert mapper.source_skeleton == "coco17"
        assert mapper.conf_threshold == 0.4
    
    def test_get_supported_skeletons(self):
        """Test getting list of supported skeletons."""
        skeletons = get_supported_skeletons()
        
        assert "coco17" in skeletons
        assert "body25" in skeletons
        assert "halpe26" in skeletons
        assert "wholebody133" in skeletons
    
    def test_skeleton_has_feet(self):
        """Test checking if skeleton has feet."""
        assert not skeleton_has_feet("coco17")
        assert skeleton_has_feet("body25")
        assert skeleton_has_feet("halpe26")
        assert skeleton_has_feet("wholebody133")
    
    def test_skeleton_has_feet_unknown(self):
        """Test checking unknown skeleton."""
        assert not skeleton_has_feet("unknown_skeleton")


class TestSchemaConsistency:
    """Tests for consistency across schema mappings."""
    
    def test_all_mappings_have_required_keypoints(self):
        """Test that all mappings have hip, knee, ankle."""
        required = ["left_hip", "right_hip", "left_knee", "right_knee",
                   "left_ankle", "right_ankle"]
        
        for skeleton_name, indices in SKELETON_MAPPINGS_BY_NAME.items():
            for kpt in required:
                assert kpt in indices, f"{skeleton_name} missing {kpt}"
    
    def test_feet_skeletons_have_heel_toe(self):
        """Test that skeletons claiming feet have heel and toe."""
        feet_skeletons = ["body25", "halpe26", "halpe136", "wholebody133", "coco_lower10"]
        
        for skeleton_name in feet_skeletons:
            if skeleton_name in SKELETON_MAPPINGS_BY_NAME:
                indices = SKELETON_MAPPINGS_BY_NAME[skeleton_name]
                assert "left_heel" in indices, f"{skeleton_name} missing left_heel"
                assert "left_toe" in indices, f"{skeleton_name} missing left_toe"
