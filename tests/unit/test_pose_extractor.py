"""Unit tests for PoseExtractor node."""

import json
from unittest.mock import patch

from pose_director_root.nodes.pose_extractor import PoseExtractorNode


def _make_pose_keypoint(kpts, h=1024, w=1024):
    """Build a POSE_KEYPOINT dict."""
    return {
        "people": [{"pose_keypoints_2d": kpts}],
        "canvas_height": h,
        "canvas_width": w,
    }


class TestPoseExtractorExtract:
    """Tests for PoseExtractorNode.extract()."""

    def test_extract_basic(self):
        """Extract keypoints from valid POSE_KEYPOINT."""
        kpts = [100.0, 200.0, 0.9] * 18
        pose_kpt = _make_pose_keypoint(kpts, 512, 768)
        node = PoseExtractorNode()
        result_json, h, w = node.extract(pose_kpt)
        parsed = json.loads(result_json)
        assert parsed == kpts
        assert h == 512
        assert w == 768

    def test_extract_person_index(self):
        """Extract specific person when multiple exist."""
        kpts0 = [float(i) for i in range(54)]
        kpts1 = [float(i + 100) for i in range(54)]
        pose_kpt = _make_pose_keypoint(kpts0, 1024, 1024)
        pose_kpt["people"].append({"pose_keypoints_2d": kpts1})

        node = PoseExtractorNode()
        result_json, h, w = node.extract(pose_kpt, person_index=1)
        parsed = json.loads(result_json)
        assert parsed == kpts1

    def test_extract_missing_canvas_dims(self):
        """Missing canvas dimensions should default to 0."""
        kpts = [100.0, 200.0, 0.9] * 18
        pose_kpt = {"people": [{"pose_keypoints_2d": kpts}]}
        node = PoseExtractorNode()
        result_json, h, w = node.extract(pose_kpt)
        assert h == 0
        assert w == 0

    def test_extract_returns_json_string(self):
        """Output should be a valid JSON string."""
        kpts = [1.0, 2.0, 0.5] * 18
        pose_kpt = _make_pose_keypoint(kpts)
        node = PoseExtractorNode()
        result_json, _, _ = node.extract(pose_kpt)
        assert isinstance(result_json, str)
        assert json.loads(result_json) is not None

    def test_extract_empty_keypoints(self):
        """Empty keypoints should produce empty JSON array."""
        pose_kpt = _make_pose_keypoint([])
        node = PoseExtractorNode()
        result_json, h, w = node.extract(pose_kpt)
        assert json.loads(result_json) == []

    def test_input_types_has_pose_keypoint(self):
        """INPUT_TYPES should declare pose_keypoint as required."""
        inputs = PoseExtractorNode.INPUT_TYPES()
        assert "pose_keypoint" in inputs["required"]
        assert inputs["required"]["pose_keypoint"][0] == "POSE_KEYPOINT"

    def test_return_types(self):
        """RETURN_TYPES should be STRING, INT, INT."""
        assert PoseExtractorNode.RETURN_TYPES == ("STRING", "INT", "INT")
        assert PoseExtractorNode.OUTPUT_NAMES == ("keypoints_json", "canvas_height", "canvas_width")
