"""Integration test: MotionDirector node (requires LLM API)."""

import pytest

from conftest import submit_workflow, get_output_images


pytestmark = [pytest.mark.integration, pytest.mark.requires_api_key]


@pytest.fixture(scope="module")
def motion_result(comfyui, character_image, llm_config):
    """Run OpenposePreprocessor -> MotionDirector and return result.

    MotionDirector outputs (IMAGE, STRING, POSE_KEYPOINT).
    Only IMAGE is tracked in ComfyUI history via the downstream SaveImage node.
    STRING and POSE_KEYPOINT outputs do NOT appear in history outputs.
    """
    llm_url, llm_model, llm_key = llm_config
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/character.png", "upload": "image"}},
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "disable",
            "resolution": 1024,
        }},
        "3": {"class_type": "MotionDirector", "inputs": {
            "keypoints": "[]",
            "motion_description": "Character waves their right hand in a friendly greeting",
            "num_keyframes": 4,
            "total_frames": 32,
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
            "width": 1024,
            "height": 1024,
            "pose_keypoint": ["2", 1],
        }},
        "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/motion_keyframes", "images": ["3", 0]}},
    }
    return submit_workflow(workflow, timeout=180, retries=2)


def test_motion_director_generates_image(motion_result):
    """MotionDirector should produce a batch of skeleton images captured by SaveImage."""
    images = get_output_images(motion_result["outputs"], "4")
    assert len(images) > 0, "No keyframe images generated"


def test_motion_director_image_nontrivial(motion_result):
    """The generated keyframe images should have non-trivial content."""
    images = get_output_images(motion_result["outputs"], "4")
    assert len(images[0]) > 5000, f"Keyframe image too small: {len(images[0])} bytes"
