"""Tests for AnimationExporterNode (ComfyUI node)."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from pose_director_root.nodes.animation_exporter import AnimationExporterNode
from utils.interpolation import JOINT_NAMES


def _tpose():
    return {j: [0.0, 0.0, 0.0] for j in JOINT_NAMES}


def _keyframes_json(frame_numbers=None, n=4, total=48):
    """Build a keyframes_json string."""
    if frame_numbers is None:
        interval = total // (n - 1) if n > 1 else total
        frame_numbers = [i * interval for i in range(n)]
        if frame_numbers:
            frame_numbers[-1] = total - 1
    kfs = [{"frame": f, "rotations": _tpose()} for f in frame_numbers]
    return json.dumps({"keyframes": kfs})


def _rotations_json(n=4):
    """Build a rotations_json string (list of rotation dicts)."""
    return json.dumps([_tpose() for _ in range(n)])


class TestAnimationExporterNodeAPI:
    def test_input_types_structure(self):
        """INPUT_TYPES should have required and optional keys."""
        inputs = AnimationExporterNode.INPUT_TYPES()
        assert "required" in inputs
        assert "optional" in inputs

    def test_return_types(self):
        """RETURN_TYPES should be (IMAGE, STRING, STRING)."""
        assert AnimationExporterNode.RETURN_TYPES == ("IMAGE", "STRING", "STRING")

    def test_output_names(self):
        """OUTPUT_NAMES should be (preview_batch, bvh_path, video_path)."""
        assert AnimationExporterNode.OUTPUT_NAMES == ("preview_batch", "bvh_path", "video_path")

    def test_function(self):
        """FUNCTION should be 'export'."""
        assert AnimationExporterNode.FUNCTION == "export"

    def test_category(self):
        """CATEGORY should be 'pose_director'."""
        assert AnimationExporterNode.CATEGORY == "pose_director"


class TestExtractFrameNumbers:
    def test_from_keyframes_data(self):
        """Should extract frame numbers from keyframes_json."""
        node = AnimationExporterNode()
        kf_data = json.loads(_keyframes_json(frame_numbers=[0, 12, 24, 47]))
        result = node._extract_frame_numbers(kf_data, 4, 48)
        assert result == [0, 12, 24, 47]

    def test_even_distribution(self):
        """Should distribute evenly when no frame numbers in data."""
        node = AnimationExporterNode()
        result = node._extract_frame_numbers({}, 4, 48)
        assert len(result) == 4
        assert result[0] == 0
        assert result[-1] == 47

    def test_single_keyframe(self):
        """Single keyframe should return [0]."""
        node = AnimationExporterNode()
        result = node._extract_frame_numbers({}, 1, 48)
        assert result == [0]


class TestNodeExport:
    @patch("pose_director_root.nodes.animation_exporter.AnimationExporterNode._build_preview_batch")
    @patch("pose_director_root.nodes.animation_exporter.get_output_dir")
    def test_bvh_only(self, mock_dir, mock_preview):
        """With export_video=False, should create BVH but no video."""
        tmpdir = tempfile.mkdtemp()
        mock_dir.return_value = tmpdir
        mock_preview.return_value = MagicMock()

        node = AnimationExporterNode()
        result = node.export(
            rotations_json=_rotations_json(3),
            keyframes_json=_keyframes_json(n=3),
            total_frames=30,
            fps=30,
            export_bvh=True,
            export_video=False,
            width=256,
            height=256,
        )

        _, bvh_path, video_path = result
        assert bvh_path.endswith(".bvh")
        assert os.path.exists(bvh_path)
        assert video_path == ""

    @patch("pose_director_root.nodes.animation_exporter.AnimationExporterNode._build_preview_batch")
    @patch("pose_director_root.nodes.animation_exporter.get_output_dir")
    def test_video_only(self, mock_dir, mock_preview):
        """With export_bvh=False, should create video but no BVH."""
        tmpdir = tempfile.mkdtemp()
        mock_dir.return_value = tmpdir
        mock_preview.return_value = MagicMock()

        node = AnimationExporterNode()
        result = node.export(
            rotations_json=_rotations_json(2),
            keyframes_json=_keyframes_json(n=2, total=10),
            total_frames=10,
            fps=24,
            export_bvh=False,
            export_video=True,
            width=64,
            height=64,
        )

        _, bvh_path, video_path = result
        assert bvh_path == ""
        assert video_path.endswith(".mp4")
        assert os.path.exists(video_path)

    @patch("pose_director_root.nodes.animation_exporter.AnimationExporterNode._build_preview_batch")
    @patch("pose_director_root.nodes.animation_exporter.get_output_dir")
    def test_both_exports(self, mock_dir, mock_preview):
        """Should export both BVH and video."""
        tmpdir = tempfile.mkdtemp()
        mock_dir.return_value = tmpdir
        mock_preview.return_value = MagicMock()

        node = AnimationExporterNode()
        result = node.export(
            rotations_json=_rotations_json(2),
            keyframes_json=_keyframes_json(n=2, total=10),
            total_frames=10,
            fps=24,
            export_bvh=True,
            export_video=True,
            width=64,
            height=64,
        )

        _, bvh_path, video_path = result
        assert bvh_path.endswith(".bvh")
        assert os.path.exists(bvh_path)
        assert video_path.endswith(".mp4")
        assert os.path.exists(video_path)

    @patch("pose_director_root.nodes.animation_exporter.AnimationExporterNode._build_preview_batch")
    @patch("pose_director_root.nodes.animation_exporter.get_output_dir")
    def test_gif_format(self, mock_dir, mock_preview):
        """GIF format should produce a .gif file."""
        tmpdir = tempfile.mkdtemp()
        mock_dir.return_value = tmpdir
        mock_preview.return_value = MagicMock()

        node = AnimationExporterNode()
        result = node.export(
            rotations_json=_rotations_json(2),
            keyframes_json=_keyframes_json(n=2, total=8),
            total_frames=8,
            fps=10,
            export_bvh=False,
            export_video=True,
            width=32,
            height=32,
            video_format="gif",
        )

        _, _, video_path = result
        assert video_path.endswith(".gif")
        assert os.path.exists(video_path)
