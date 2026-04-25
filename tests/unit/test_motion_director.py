"""Unit tests for MotionDirector node with mocked LLM calls."""

import json
from unittest.mock import patch, MagicMock

import pytest
import torch

from pose_director_root.nodes.motion_director import MotionDirectorNode


def _make_coco18_keypoints():
    """Generate a valid COCO-18 keypoint array (54 floats)."""
    kpts = []
    for i in range(18):
        kpts.extend([float(100 + i * 10), float(200 + i * 5), 0.9])
    return kpts


class TestMotionDirectorGenerate:
    """Tests for MotionDirectorNode.generate() with mocked LLM."""

    @patch("pose_director_root.nodes.motion_director.call_llm")
    def test_generate_basic(self, mock_call_llm):
        """Happy path: generate keyframes from string keypoints."""
        kpts = _make_coco18_keypoints()
        mock_call_llm.return_value = json.dumps({
            "keyframes": [
                {"frame": 0, "keypoints": kpts},
                {"frame": 16, "keypoints": kpts},
            ]
        })

        node = MotionDirectorNode()
        batch, kf_json, pose_kpts, rotations_json = node.generate(
            keypoints=json.dumps(kpts),
            motion_description="wave",
            num_keyframes=2,
            total_frames=32,
            base_url="http://test/v1",
            model="m",
            width=512,
            height=512,
            output_mode="keypoints",
        )

        assert batch.shape[0] == 2  # 2 keyframes in batch
        assert batch.shape[1] == 512  # height
        assert batch.shape[2] == 512  # width
        assert batch.shape[3] == 3   # channels
        assert isinstance(kf_json, str)
        assert len(pose_kpts) == 2
        assert pose_kpts[0]["frame"] == 0

    @patch("pose_director_root.nodes.motion_director.call_llm")
    def test_generate_with_pose_keypoint(self, mock_call_llm):
        """Extract keypoints from POSE_KEYPOINT input instead of STRING."""
        kpts = _make_coco18_keypoints()
        mock_call_llm.return_value = json.dumps({
            "keyframes": [
                {"frame": 0, "keypoints": kpts},
            ]
        })

        pose_keypoint = {
            "people": [{"pose_keypoints_2d": kpts}],
            "canvas_height": 1024,
            "canvas_width": 768,
        }

        node = MotionDirectorNode()
        batch, kf_json, pose_kpts, rotations_json = node.generate(
            keypoints="[]",
            motion_description="wave",
            num_keyframes=1,
            total_frames=16,
            base_url="http://test/v1",
            model="m",
            width=512,
            height=512,
            pose_keypoint=pose_keypoint,
            output_mode="keypoints",
        )

        # Should have extracted canvas dimensions from pose_keypoint
        assert batch.shape[1] == 1024  # height from pose_keypoint
        assert batch.shape[2] == 768   # width from pose_keypoint
        assert len(pose_kpts) == 1
        assert pose_kpts[0]["canvas_height"] == 1024

    @patch("pose_director_root.nodes.motion_director.call_llm")
    def test_generate_list_wrap_unwrap(self, mock_call_llm):
        """LLM may return a list wrapping the dict — unwrap it."""
        kpts = _make_coco18_keypoints()
        mock_call_llm.return_value = json.dumps([{
            "keyframes": [
                {"frame": 0, "keypoints": kpts},
            ]
        }])

        node = MotionDirectorNode()
        batch, _, _, _ = node.generate(
            keypoints=json.dumps(kpts),
            motion_description="wave",
            num_keyframes=1,
            total_frames=16,
            base_url="http://test/v1",
            model="m",
            width=512,
            height=512,
            output_mode="keypoints",
        )

        assert batch.shape[0] == 1

    @patch("pose_director_root.nodes.motion_director.call_llm")
    def test_generate_base_pose_not_found(self, mock_call_llm):
        """Non-existent base_pose should be skipped with a warning."""
        kpts = _make_coco18_keypoints()
        mock_call_llm.return_value = json.dumps({
            "keyframes": [{"frame": 0, "keypoints": kpts}]
        })

        node = MotionDirectorNode()
        batch, _, _, _ = node.generate(
            keypoints=json.dumps(kpts),
            motion_description="wave",
            num_keyframes=1,
            total_frames=16,
            base_url="http://test/v1",
            model="m",
            width=512,
            height=512,
            base_pose="nonexistent_pose",
            output_mode="keypoints",
        )

        assert batch.shape[0] == 1

    @patch("pose_director_root.nodes.motion_director.call_llm")
    @patch("pose_director_root.nodes.motion_director.validate_keypoints")
    def test_generate_validation_warnings(self, mock_validate, mock_call_llm):
        """Validation issues should be logged but not block execution."""
        kpts = _make_coco18_keypoints()
        mock_call_llm.return_value = json.dumps({
            "keyframes": [
                {"frame": 0, "keypoints": kpts},
                {"frame": 8, "keypoints": kpts},
            ]
        })

        # Simulate validation finding issues
        mock_issue = MagicMock()
        mock_issue.severity = "warning"
        mock_issue.message = "test issue"
        mock_validate.return_value = (False, [mock_issue])

        node = MotionDirectorNode()
        batch, _, _, _ = node.generate(
            keypoints=json.dumps(kpts),
            motion_description="wave",
            num_keyframes=2,
            total_frames=16,
            base_url="http://test/v1",
            model="m",
            width=512,
            height=512,
            output_mode="keypoints",
        )

        assert batch.shape[0] == 2
        assert mock_validate.call_count == 2

    @patch("pose_director_root.nodes.motion_director.call_llm")
    @patch("pose_director_root.nodes.motion_director.validate_keyframe_consistency")
    def test_generate_consistency_warnings(self, mock_consistency, mock_call_llm):
        """Keyframe consistency issues should be logged but not block."""
        kpts = _make_coco18_keypoints()
        mock_call_llm.return_value = json.dumps({
            "keyframes": [
                {"frame": 0, "keypoints": kpts},
                {"frame": 8, "keypoints": kpts},
            ]
        })

        mock_consistency.return_value = {
            "is_consistent": False,
            "flagged_bones": [("bone1", 0.5, 0.8)],
        }

        node = MotionDirectorNode()
        batch, _, _, _ = node.generate(
            keypoints=json.dumps(kpts),
            motion_description="wave",
            num_keyframes=2,
            total_frames=16,
            base_url="http://test/v1",
            model="m",
            width=512,
            height=512,
            output_mode="keypoints",
        )

        assert batch.shape[0] == 2
        mock_consistency.assert_called_once()

    def test_generate_keypoints_as_list(self):
        """keypoints can be passed as a list, not just a JSON string."""
        kpts = _make_coco18_keypoints()
        with patch("pose_director_root.nodes.motion_director.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps({
                "keyframes": [{"frame": 0, "keypoints": kpts}]
            })
            node = MotionDirectorNode()
            batch, _, _, _ = node.generate(
                keypoints=kpts,  # pass list directly
                motion_description="wave",
                num_keyframes=1,
                total_frames=16,
                base_url="http://test/v1",
                model="m",
                width=256,
                height=256,
                output_mode="keypoints",
            )
            assert batch.shape[0] == 1

    def test_generate_total_frames_one_keyframe(self):
        """Edge case: only 1 keyframe should still work."""
        kpts = _make_coco18_keypoints()
        with patch("pose_director_root.nodes.motion_director.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps({
                "keyframes": [{"frame": 0, "keypoints": kpts}]
            })
            node = MotionDirectorNode()
            batch, _, _, _ = node.generate(
                keypoints=json.dumps(kpts),
                motion_description="wave",
                num_keyframes=1,
                total_frames=16,
                base_url="http://test/v1",
                model="m",
                width=128,
                height=128,
                output_mode="keypoints",
            )
            assert batch.shape[0] == 1
