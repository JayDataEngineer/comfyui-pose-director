"""Unit tests for node class metadata — INPUT_TYPES, RETURN_TYPES, CATEGORY, FUNCTION."""

import json

import pytest


# Import all node classes directly
from pose_director_root.nodes.pose_manipulator import PoseManipulatorNode
from pose_director_root.nodes.motion_director import MotionDirectorNode
from pose_director_root.nodes.pose_renderer import PoseRendererNode
from pose_director_root.nodes.pose_extractor import PoseExtractorNode
from pose_director_root.nodes.keyframe_reviewer import KeyframeReviewerNode

ALL_NODES = [
    PoseManipulatorNode,
    MotionDirectorNode,
    PoseRendererNode,
    PoseExtractorNode,
    KeyframeReviewerNode,
]


class TestNodeMetadata:
    """All nodes must have correct ComfyUI node metadata."""

    @pytest.mark.parametrize("node_cls", ALL_NODES, ids=lambda c: c.__name__)
    def test_has_category(self, node_cls):
        assert node_cls.CATEGORY == "pose_director"

    @pytest.mark.parametrize("node_cls", ALL_NODES, ids=lambda c: c.__name__)
    def test_has_function(self, node_cls):
        assert hasattr(node_cls, "FUNCTION")
        assert isinstance(node_cls.FUNCTION, str)
        assert hasattr(node_cls, node_cls.FUNCTION)

    @pytest.mark.parametrize("node_cls", ALL_NODES, ids=lambda c: c.__name__)
    def test_has_return_types(self, node_cls):
        assert isinstance(node_cls.RETURN_TYPES, tuple)
        assert len(node_cls.RETURN_TYPES) > 0

    @pytest.mark.parametrize("node_cls", ALL_NODES, ids=lambda c: c.__name__)
    def test_has_output_names_if_defined(self, node_cls):
        if hasattr(node_cls, "OUTPUT_NAMES"):
            assert isinstance(node_cls.OUTPUT_NAMES, tuple)
            assert len(node_cls.OUTPUT_NAMES) == len(node_cls.RETURN_TYPES)

    @pytest.mark.parametrize("node_cls", ALL_NODES, ids=lambda c: c.__name__)
    def test_input_types_is_callable(self, node_cls):
        result = node_cls.INPUT_TYPES()
        assert isinstance(result, dict)
        assert "required" in result
        assert isinstance(result["required"], dict)

    @pytest.mark.parametrize("node_cls", ALL_NODES, ids=lambda c: c.__name__)
    def test_has_description(self, node_cls):
        assert hasattr(node_cls, "DESCRIPTION")
        assert isinstance(node_cls.DESCRIPTION, str)
        assert len(node_cls.DESCRIPTION) > 10


class TestSpecificNodeMetadata:
    """Node-specific metadata checks."""

    def test_pose_manipulator_inputs(self):
        inputs = PoseManipulatorNode.INPUT_TYPES()
        required = inputs["required"]
        assert "keypoints" in required
        assert "instruction" in required
        assert "base_url" in required
        assert "api_key" in required
        assert "model" in required
        assert "width" in required
        assert "height" in required
        assert "pose_keypoint" in inputs.get("optional", {})

    def test_motion_director_inputs(self):
        inputs = MotionDirectorNode.INPUT_TYPES()
        required = inputs["required"]
        assert "keypoints" in required
        assert "motion_description" in required
        assert "num_keyframes" in required
        assert "total_frames" in required
        assert "pose_keypoint" in inputs.get("optional", {})

    def test_pose_renderer_inputs(self):
        inputs = PoseRendererNode.INPUT_TYPES()
        required = inputs["required"]
        assert "keypoints" in required
        assert "width" in required
        assert "height" in required
        assert "line_width" in required
        assert "point_radius" in required

    def test_pose_extractor_inputs(self):
        inputs = PoseExtractorNode.INPUT_TYPES()
        required = inputs["required"]
        assert "pose_keypoint" in required
        assert required["pose_keypoint"][0] == "POSE_KEYPOINT"
        assert "person_index" in inputs.get("optional", {})

    def test_keyframe_reviewer_inputs(self):
        inputs = KeyframeReviewerNode.INPUT_TYPES()
        required = inputs["required"]
        assert "images" in required
        assert "pass_threshold" in required
        assert required["images"][0] == "IMAGE"

    def test_pose_manipulator_returns_image(self):
        assert PoseManipulatorNode.RETURN_TYPES[0] == "IMAGE"

    def test_motion_director_returns_image_batch(self):
        assert MotionDirectorNode.RETURN_TYPES[0] == "IMAGE"
        assert MotionDirectorNode.RETURN_TYPES[1] == "STRING"
        assert MotionDirectorNode.RETURN_TYPES[2] == "POSE_KEYPOINT"

    def test_keyframe_reviewer_returns_bool(self):
        assert KeyframeReviewerNode.RETURN_TYPES[1] == "BOOLEAN"
        assert KeyframeReviewerNode.RETURN_TYPES[2] == "INT"
