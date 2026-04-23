"""Integration test: OpenposePreprocessor -> PoseExtractor pipeline."""

import json

import pytest

from conftest import submit_workflow, get_output_images


pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def extractor_result(comfyui, character_image):
    """Run OpenposePreprocessor -> PoseExtractor -> PoseRenderer and return result.

    PoseExtractor outputs STRING/INT types that ComfyUI doesn't include in /history.
    We chain into PoseRenderer to verify the extractor output works correctly,
    and capture the OpenposePreprocessor's openpose_json for direct validation.
    """
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/character.png", "upload": "image"}},
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "disable",
            "resolution": 1024,
        }},
        "3": {"class_type": "PoseExtractor", "inputs": {"pose_keypoint": ["2", 1], "person_index": 0}},
        # Render using extracted keypoints + dimensions from PoseExtractor
        "4": {"class_type": "PoseRenderer", "inputs": {
            "keypoints": ["3", 0],
            "width": 1024,
            "height": 1024,
            "line_width": 4,
            "point_radius": 6,
        }},
        "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/extractor_rendered", "images": ["4", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/extractor_dwpose", "images": ["2", 0]}},
    }
    return submit_workflow(workflow, timeout=120)


def test_extractor_renders_skeleton(extractor_result):
    """PoseExtractor -> PoseRenderer should produce a skeleton image."""
    images = get_output_images(extractor_result["outputs"], "5")
    assert len(images) > 0, "No rendered skeleton image"
    assert len(images[0]) > 1000, "Rendered skeleton image too small"


def test_openpose_returns_openpose_json(extractor_result):
    """OpenposePreprocessor should include openpose_json in its UI output.

    The openpose_json comes from the node's 'ui' return value, which ComfyUI
    includes in the history outputs alongside any tracked outputs.
    This is intermittent — ComfyUI may not always include it in history.
    """
    out = extractor_result["outputs"].get("2", {})
    json_texts = out.get("openpose_json", [])
    if not json_texts:
        pytest.skip(f"openpose_json not available in history (keys: {list(out.keys())})")

    # openpose_json is a list of pose data objects: [{"people": [...], "canvas_height": ..., ...}]
    pose_data = json.loads(json_texts[0])

    # Can be a list of dicts or a single dict
    if isinstance(pose_data, list):
        pose_data = pose_data[0]

    assert "people" in pose_data
    people = pose_data["people"]
    assert len(people) > 0, "No people detected"

    kpts = people[0]["pose_keypoints_2d"]
    assert len(kpts) >= 54, f"Expected 54+ floats, got {len(kpts)}"


def test_openpose_keypoints_in_range(extractor_result):
    """OpenPose keypoints should have valid coordinates and confidence."""
    out = extractor_result["outputs"].get("2", {})
    json_texts = out.get("openpose_json", [])
    if not json_texts:
        pytest.skip("No openpose_json available")

    pose_data = json.loads(json_texts[0])
    if isinstance(pose_data, list):
        pose_data = pose_data[0]

    kpts = pose_data["people"][0]["pose_keypoints_2d"]

    for i in range(0, len(kpts), 3):
        x, y, conf = kpts[i], kpts[i + 1], kpts[i + 2]
        # Normalized coords from OpenposePreprocessor (0-1 range)
        assert 0 <= x <= 1.0, f"x={x} out of normalized range at index {i}"
        assert 0 <= y <= 1.0, f"y={y} out of normalized range at index {i}"
        assert 0 <= conf <= 1.0, f"conf={conf} out of range at index {i}"


def test_openpose_image_generated(extractor_result):
    """OpenposePreprocessor should produce a skeleton visualization image."""
    images = get_output_images(extractor_result["outputs"], "6")
    assert len(images) > 0, "No OpenPose image generated"
    assert len(images[0]) > 5000, "OpenPose image too small"
