"""Unit tests for PoseRenderer node."""

import json

import numpy as np
import pytest
import torch

from pose_director_root.nodes.pose_renderer import PoseRendererNode


def make_coco18_keypoints(canvas_w=512, canvas_h=512):
    """Generate a plausible COCO-18 keypoint array for a standing person."""
    cx, cy = canvas_w / 2, canvas_h / 2
    return [
        cx, cy - 180, 0.9,       # 0 Nose
        cx, cy - 160, 0.9,       # 1 Neck
        cx - 50, cy - 140, 0.9,  # 2 RShoulder
        cx - 70, cy - 80, 0.9,   # 3 RElbow
        cx - 60, cy - 20, 0.9,   # 4 RWrist
        cx + 50, cy - 140, 0.9,  # 5 LShoulder
        cx + 70, cy - 80, 0.9,   # 6 LElbow
        cx + 60, cy - 20, 0.9,   # 7 LWrist
        cx - 30, cy - 60, 0.9,   # 8 RHip
        cx - 35, cy + 30, 0.9,   # 9 RKnee
        cx - 40, cy + 120, 0.9,  # 10 RAnkle
        cx + 30, cy - 60, 0.9,   # 11 LHip
        cx + 35, cy + 30, 0.9,   # 12 LKnee
        cx + 40, cy + 120, 0.9,  # 13 LAnkle
        cx - 5, cy - 190, 0.9,   # 14 REye
        cx + 5, cy - 190, 0.9,   # 15 LEye
        cx - 15, cy - 185, 0.9,  # 16 REar
        cx + 15, cy - 185, 0.9,  # 17 LEar
    ]


class TestPoseRenderer:
    """Tests for PoseRendererNode.render()."""

    def test_render_single_person(self):
        renderer = PoseRendererNode()
        kpts = make_coco18_keypoints()
        keypoints_json = json.dumps([{"pose_keypoints_2d": kpts}])
        result = renderer.render(keypoints_json, 512, 512)

        assert isinstance(result, tuple)
        assert len(result) == 1
        img = result[0]
        assert isinstance(img, torch.Tensor)
        assert img.dim() == 4  # (1, H, W, C)
        assert img.shape == (1, 512, 512, 3)

    def test_render_output_shape(self):
        renderer = PoseRendererNode()
        kpts = make_coco18_keypoints(1024, 768)
        keypoints_json = json.dumps([{"pose_keypoints_2d": kpts}])
        result = renderer.render(keypoints_json, 1024, 768, line_width=6, point_radius=8)

        img = result[0]
        assert img.shape == (1, 768, 1024, 3)  # (batch, H, W, C)

    def test_render_non_black_output(self):
        """Skeleton should produce non-black pixels."""
        renderer = PoseRendererNode()
        kpts = make_coco18_keypoints()
        keypoints_json = json.dumps([{"pose_keypoints_2d": kpts}])
        result = renderer.render(keypoints_json, 512, 512)

        img = result[0]
        pixel_sum = img.sum().item()
        assert pixel_sum > 0, "Rendered skeleton should have non-zero pixels"

    def test_render_empty_keypoints(self):
        """Empty keypoint array should produce a blank black image."""
        renderer = PoseRendererNode()
        result = renderer.render("[]", 256, 256)

        img = result[0]
        assert img.shape == (1, 256, 256, 3)
        assert img.sum().item() == 0, "Empty keypoints should produce all-black image"

    def test_render_low_confidence_filtered(self):
        """Keypoints with confidence <= 0.1 should not be drawn."""
        renderer = PoseRendererNode()
        kpts = [256.0, 256.0, 0.05] * 18  # All low confidence
        keypoints_json = json.dumps([{"pose_keypoints_2d": kpts}])
        result = renderer.render(keypoints_json, 512, 512)

        img = result[0]
        assert img.sum().item() == 0, "Low-confidence keypoints should produce black image"

    def test_render_multiple_people(self):
        """Should render multiple people on the same canvas."""
        renderer = PoseRendererNode()
        kpts1 = make_coco18_keypoints(256, 512)
        kpts2 = make_coco18_keypoints(768, 512)
        keypoints_json = json.dumps([
            {"pose_keypoints_2d": kpts1},
            {"pose_keypoints_2d": kpts2},
        ])
        result = renderer.render(keypoints_json, 1024, 512)

        img = result[0]
        assert img.shape == (1, 512, 1024, 3)
        assert img.sum().item() > 0

    def test_render_output_in_valid_range(self):
        """Output tensor values should be 0-1 float32."""
        renderer = PoseRendererNode()
        kpts = make_coco18_keypoints()
        keypoints_json = json.dumps([{"pose_keypoints_2d": kpts}])
        result = renderer.render(keypoints_json, 512, 512)

        img = result[0]
        assert img.dtype == torch.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_render_with_raw_keypoints_array(self):
        """Should work when JSON is a flat array wrapped in a dict with 'keypoints' key."""
        renderer = PoseRendererNode()
        kpts = make_coco18_keypoints()
        keypoints_json = json.dumps([{"keypoints": kpts}])
        result = renderer.render(keypoints_json, 512, 512)

        img = result[0]
        assert img.shape == (1, 512, 512, 3)
        assert img.sum().item() > 0
