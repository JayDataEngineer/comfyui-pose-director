"""Integration test: PoseRenderer node via ComfyUI API."""

import json

import pytest

from conftest import submit_workflow, get_output_images


pytestmark = pytest.mark.integration


# Sample COCO-18 keypoints for a standing person at center of 512x512
SAMPLE_KEYPOINTS = json.dumps([
    256.0, 76.0, 0.9,   # 0 Nose
    256.0, 96.0, 0.9,   # 1 Neck
    206.0, 116.0, 0.9,  # 2 RShoulder
    186.0, 176.0, 0.9,  # 3 RElbow
    196.0, 236.0, 0.9,  # 4 RWrist
    306.0, 116.0, 0.9,  # 5 LShoulder
    326.0, 176.0, 0.9,  # 6 LElbow
    316.0, 236.0, 0.9,  # 7 LWrist
    226.0, 196.0, 0.9,  # 8 RHip
    221.0, 286.0, 0.9,  # 9 RKnee
    216.0, 376.0, 0.9,  # 10 RAnkle
    286.0, 196.0, 0.9,  # 11 LHip
    291.0, 286.0, 0.9,  # 12 LKnee
    296.0, 376.0, 0.9,  # 13 LAnkle
    251.0, 66.0, 0.9,   # 14 REye
    261.0, 66.0, 0.9,   # 15 LEye
    241.0, 71.0, 0.9,   # 16 REar
    271.0, 71.0, 0.9,   # 17 LEar
])


def test_pose_renderer_via_api(comfyui):
    """PoseRenderer should produce an image via ComfyUI API."""
    workflow = {
        "1": {"class_type": "PoseRenderer", "inputs": {
            "keypoints": f'[{{"pose_keypoints_2d": {SAMPLE_KEYPOINTS}}}]',
            "width": 512,
            "height": 512,
            "line_width": 4,
            "point_radius": 6,
        }},
        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/renderer_test", "images": ["1", 0]}},
    }
    result = submit_workflow(workflow, timeout=120)
    images = get_output_images(result["outputs"], "2")

    assert len(images) > 0, "No rendered image"
    assert len(images[0]) > 1000, "Rendered image too small"


def test_pose_renderer_empty_keypoints(comfyui):
    """PoseRenderer with empty keypoints should produce a black image."""
    workflow = {
        "1": {"class_type": "PoseRenderer", "inputs": {
            "keypoints": "[]",
            "width": 256,
            "height": 256,
            "line_width": 4,
            "point_radius": 4,
        }},
        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/renderer_empty", "images": ["1", 0]}},
    }
    result = submit_workflow(workflow, timeout=120)
    images = get_output_images(result["outputs"], "2")

    assert len(images) > 0
    # Black image is still a valid PNG, just small file size
    assert len(images[0]) > 100, "Even black image should have some content"
