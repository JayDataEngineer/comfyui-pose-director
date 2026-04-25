"""Unit tests for PoseManipulator node with mocked LLM calls."""

import json
from unittest.mock import patch, MagicMock

import pytest

from pose_director_root.nodes.pose_manipulator import PoseManipulatorNode


def _make_coco18_keypoints():
    """Generate a valid COCO-18 keypoint array (54 floats)."""
    kpts = []
    for i in range(18):
        kpts.extend([float(100 + i * 10), float(200 + i * 5), 0.9])
    return kpts


class TestPoseManipulatorManipulate:
    """Tests for PoseManipulatorNode.manipulate() with mocked LLM."""

    @patch("pose_director_root.nodes.pose_manipulator.call_llm")
    def test_manipulate_basic(self, mock_call_llm):
        """Happy path: modify pose from string keypoints."""
        kpts = _make_coco18_keypoints()
        mock_call_llm.return_value = json.dumps(kpts)

        node = PoseManipulatorNode()
        img_tensor, kf_json, pose_kpt = node.manipulate(
            keypoints=json.dumps(kpts),
            instruction="Raise right arm",
            base_url="http://test/v1",
            model="m",
            width=512,
            height=512,
        )

        assert img_tensor.shape == (1, 512, 512, 3)  # (batch, H, W, C)
        assert isinstance(kf_json, str)
        assert "pose_keypoints_2d" in pose_kpt["people"][0]

    @patch("pose_director_root.nodes.pose_manipulator.call_llm")
    def test_manipulate_with_pose_keypoint(self, mock_call_llm):
        """Extract keypoints from POSE_KEYPOINT input."""
        kpts = _make_coco18_keypoints()
        mock_call_llm.return_value = json.dumps(kpts)

        pose_keypoint = {
            "people": [{"pose_keypoints_2d": kpts}],
            "canvas_height": 1024,
            "canvas_width": 768,
        }

        node = PoseManipulatorNode()
        img_tensor, _, pose_kpt = node.manipulate(
            keypoints="[]",
            instruction="Cross arms",
            base_url="http://test/v1",
            model="m",
            width=512,
            height=512,
            pose_keypoint=pose_keypoint,
        )

        # Should have used canvas dimensions from pose_keypoint
        assert img_tensor.shape == (1, 1024, 768, 3)
        assert pose_kpt["canvas_height"] == 1024

    @patch("pose_director_root.nodes.pose_manipulator.call_llm")
    def test_manipulate_base_pose_not_found(self, mock_call_llm):
        """Non-existent base_pose should be skipped with a warning."""
        kpts = _make_coco18_keypoints()
        mock_call_llm.return_value = json.dumps(kpts)

        node = PoseManipulatorNode()
        img_tensor, _, _ = node.manipulate(
            keypoints=json.dumps(kpts),
            instruction="Sit down",
            base_url="http://test/v1",
            model="m",
            width=256,
            height=256,
            base_pose="nonexistent_pose",
        )

        assert img_tensor.shape == (1, 256, 256, 3)
    @patch("pose_director_root.nodes.pose_manipulator.call_llm")
    @patch("pose_director_root.nodes.pose_manipulator.validate_keypoints")
    def test_manipulate_validation_warnings(self, mock_validate, mock_call_llm):
        """Validation issues should be logged but not block execution."""
        kpts = _make_coco18_keypoints()
        mock_call_llm.return_value = json.dumps(kpts)

        mock_issue = MagicMock()
        mock_issue.severity = "warning"
        mock_issue.message = "test warning"
        mock_validate.return_value = (False, [mock_issue])

        node = PoseManipulatorNode()
        img_tensor, _, _ = node.manipulate(
            keypoints=json.dumps(kpts),
            instruction="Jump",
            base_url="http://test/v1",
            model="m",
            width=128,
            height=128,
        )

        assert img_tensor.shape == (1, 128, 128, 3)
        mock_validate.assert_called_once()

    def test_manipulate_keypoints_as_list(self):
        """keypoints can be passed as a list, not just a JSON string."""
        kpts = _make_coco18_keypoints()
        with patch("pose_director_root.nodes.pose_manipulator.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps(kpts)
            node = PoseManipulatorNode()
            img_tensor, kf_json, _ = node.manipulate(
                keypoints=kpts,  # list directly
                instruction="Wave",
                base_url="http://test/v1",
                model="m",
                width=64,
                height=64,
            )
            assert img_tensor.shape == (1, 64, 64, 3)
