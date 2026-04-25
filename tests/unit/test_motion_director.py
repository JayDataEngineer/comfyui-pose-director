"""Unit tests for MotionDirector node (rotation-only mode) with mocked LLM calls."""

import json
from unittest.mock import patch, MagicMock

import pytest
import torch

from pose_director_root.nodes.motion_director import MotionDirectorNode


# Valid LLM response with all 15 joints
VALID_ROTATIONS = {
    "spine": [0, 0, 0], "neck": [0, 0, 0], "head": [0, 0, 0],
    "right_shoulder": [0, 0, 90], "right_elbow": [0, 0, -45],
    "right_wrist": [0, 0, 0], "left_shoulder": [0, 0, -90],
    "left_elbow": [0, 0, 45], "left_wrist": [0, 0, 0],
    "right_hip": [0, 0, 0], "right_knee": [0, 0, 0],
    "right_ankle": [0, 0, 0], "left_hip": [0, 0, 0],
    "left_knee": [0, 0, 0], "left_ankle": [0, 0, 0],
}


def _make_valid_response(num_keyframes=2, total_frames=32):
    """Create a valid LLM response with all joints."""
    frames = [0]
    if num_keyframes > 1:
        interval = total_frames // (num_keyframes - 1)
        frames = [i * interval for i in range(num_keyframes)]
        frames[-1] = total_frames - 1

    keyframes = [
        {"frame": f, "rotations": dict(VALID_ROTATIONS)}
        for f in frames
    ]
    return json.dumps({"keyframes": keyframes})


class TestMotionDirectorGenerate:
    """Tests for MotionDirectorNode.generate() with mocked LLM."""

    @patch("pose_director_root.nodes.motion_director.call_llm")
    def test_generate_basic_rotation(self, mock_call_llm):
        """Happy path: generate rotation keyframes."""
        mock_call_llm.return_value = _make_valid_response(2, 32)

        node = MotionDirectorNode()
        batch, kf_json, rotations_json = node.generate(
            motion_description="wave",
            num_keyframes=2,
            total_frames=32,
            base_url="http://test/v1",
            model="m",
            width=256,
            height=256,
        )

        assert batch.shape[0] == 2  # 2 keyframes
        assert batch.shape[3] == 3  # RGB
        assert isinstance(kf_json, str)
        assert isinstance(rotations_json, str)
        kf_data = json.loads(kf_json)
        assert len(kf_data["keyframes"]) == 2

    @patch("pose_director_root.nodes.motion_director.call_llm")
    def test_generate_partial_rotations_filled(self, mock_call_llm):
        """LLM outputs only changed joints — missing ones filled from prev frame."""
        response = json.dumps({
            "keyframes": [
                {"frame": 0, "rotations": dict(VALID_ROTATIONS)},
                {"frame": 16, "rotations": {"right_shoulder": [0, 0, 180]}},
            ]
        })
        mock_call_llm.return_value = response

        node = MotionDirectorNode()
        batch, kf_json, _ = node.generate(
            motion_description="raise arm",
            num_keyframes=2,
            total_frames=32,
            base_url="http://test/v1",
            model="m",
            width=256,
            height=256,
        )

        assert batch.shape[0] == 2
        kf_data = json.loads(kf_json)
        # Frame 1 should have all 15 joints filled from frame 0
        frame1_rots = kf_data["keyframes"][1]["rotations"]
        assert "left_shoulder" in frame1_rots
        assert frame1_rots["right_shoulder"] == [0, 0, 180]

    @patch("pose_director_root.nodes.motion_director.call_llm")
    def test_generate_list_wrap_unwrap(self, mock_call_llm):
        """LLM may return a list wrapping the dict — unwrap it."""
        mock_call_llm.return_value = json.dumps([json.loads(_make_valid_response(1, 16))])

        node = MotionDirectorNode()
        batch, _, _ = node.generate(
            motion_description="wave",
            num_keyframes=1,
            total_frames=16,
            base_url="http://test/v1",
            model="m",
            width=128,
            height=128,
        )
        assert batch.shape[0] == 1

    @patch("pose_director_root.nodes.motion_director.call_llm")
    def test_generate_missing_joints_triggers_retry(self, mock_call_llm):
        """Missing joints on first attempt → retry with error feedback."""
        bad_response = json.dumps({
            "keyframes": [
                {"frame": 0, "rotations": {"spine": [0, 0, 0]}},
            ]
        })
        good_response = _make_valid_response(1, 16)
        mock_call_llm.side_effect = [bad_response, good_response]

        node = MotionDirectorNode()
        batch, _, _ = node.generate(
            motion_description="wave",
            num_keyframes=1,
            total_frames=16,
            base_url="http://test/v1",
            model="m",
            width=128,
            height=128,
            max_retries=1,
        )
        assert batch.shape[0] == 1
        assert mock_call_llm.call_count == 2

    @patch("pose_director_root.nodes.motion_director.call_llm")
    def test_generate_parse_error_triggers_retry(self, mock_call_llm):
        """Unparseable response → retry."""
        bad_response = "This is not JSON at all"
        good_response = _make_valid_response(1, 16)
        mock_call_llm.side_effect = [bad_response, good_response]

        node = MotionDirectorNode()
        batch, _, _ = node.generate(
            motion_description="wave",
            num_keyframes=1,
            total_frames=16,
            base_url="http://test/v1",
            model="m",
            width=128,
            height=128,
            max_retries=1,
        )
        assert batch.shape[0] == 1
        assert mock_call_llm.call_count == 2

    @patch("pose_director_root.nodes.motion_director.call_llm")
    def test_generate_all_retries_exhausted(self, mock_call_llm):
        """All retries fail → uses best-effort last parsed result."""
        bad_response = json.dumps({
            "keyframes": [
                {"frame": 0, "rotations": {"spine": [0, 0, 0], "neck": [0, 0, 0]}},
            ]
        })
        mock_call_llm.return_value = bad_response

        node = MotionDirectorNode()
        batch, _, _ = node.generate(
            motion_description="wave",
            num_keyframes=1,
            total_frames=16,
            base_url="http://test/v1",
            model="m",
            width=128,
            height=128,
            max_retries=0,
        )
        # Should still produce output (best effort after sanitization)
        assert batch.shape[0] == 1

    @patch("pose_director_root.nodes.motion_director.call_llm")
    def test_generate_sanitizes_extreme_angles(self, mock_call_llm):
        """Extreme angles get clamped to anatomical limits."""
        extreme_rotations = dict(VALID_ROTATIONS)
        extreme_rotations["right_knee"] = [-50, 0, 0]  # backward bend
        extreme_rotations["left_elbow"] = [0, 0, 200]   # over-bend

        response = json.dumps({
            "keyframes": [
                {"frame": 0, "rotations": extreme_rotations},
            ]
        })
        mock_call_llm.return_value = response

        node = MotionDirectorNode()
        _, _, rotations_json = node.generate(
            motion_description="extreme pose",
            num_keyframes=1,
            total_frames=16,
            base_url="http://test/v1",
            model="m",
            width=128,
            height=128,
        )

        rots = json.loads(rotations_json)
        # Knee should be clamped to min -5
        assert rots[0]["right_knee"][0] >= -5
        # Elbow should be clamped to max 145
        assert rots[0]["left_elbow"][2] <= 145

    def test_generate_no_pose_keypoint_input(self):
        """MotionDirector should not accept pose_keypoint input."""
        inputs = MotionDirectorNode.INPUT_TYPES()
        assert "pose_keypoint" not in inputs.get("required", {})
        assert "pose_keypoint" not in inputs.get("optional", {})

    def test_generate_no_output_mode_input(self):
        """MotionDirector should not have output_mode input."""
        inputs = MotionDirectorNode.INPUT_TYPES()
        assert "output_mode" not in inputs.get("required", {})
        assert "output_mode" not in inputs.get("optional", {})

    def test_output_types(self):
        """Should have 3 outputs: IMAGE, STRING, STRING."""
        assert MotionDirectorNode.RETURN_TYPES == ("IMAGE", "STRING", "STRING")
        assert MotionDirectorNode.OUTPUT_NAMES == ("pose_batch", "keyframes_json", "rotations_json")
