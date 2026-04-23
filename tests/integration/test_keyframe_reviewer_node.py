"""Integration test: KeyframeReviewer node (requires vision LLM API).

KeyframeReviewer outputs (STRING, BOOLEAN, INT) -- none of these are tracked
types in ComfyUI's /history endpoint. We verify the workflow succeeds by
checking that a downstream SaveImage node (consuming the keyframe images)
produces output. The reviewer's actual output cannot be retrieved via the API.
"""

import pytest

from conftest import submit_workflow, get_output_images


pytestmark = [pytest.mark.integration, pytest.mark.requires_api_key]


@pytest.fixture(scope="module")
def review_result(comfyui, character_image, llm_config):
    """Generate keyframes and review them. Returns the workflow result.

    We chain MotionDirector -> KeyframeReviewer + SaveImage so that:
    1. The keyframe images are saved (verifies MotionDirector worked)
    2. The KeyframeReviewer runs (verifies it doesn't crash)
    The reviewer's STRING/BOOLEAN/INT outputs are not accessible via history.
    """
    llm_url, llm_model, llm_key = llm_config
    review_workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/character.png", "upload": "image"}},
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": "enable", "detect_body": "enable", "detect_face": "disable",
            "resolution": 512,
        }},
        "3": {"class_type": "MotionDirector", "inputs": {
            "keypoints": "[]",
            "motion_description": "Character raises right arm",
            "num_keyframes": 3,
            "total_frames": 24,
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
            "width": 512,
            "height": 512,
            "pose_keypoint": ["2", 1],
        }},
        # KeyframeReviewer consumes the IMAGE batch
        "5": {"class_type": "KeyframeReviewer", "inputs": {
            "images": ["3", 0],
            "review_prompt": "Review these keyframe skeleton images for pose quality and progression.",
            "pass_threshold": 5,
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
        }},
        # Save keyframes to verify the workflow produced images
        "6": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/review_keyframes", "images": ["3", 0]}},
    }
    return submit_workflow(review_workflow, timeout=300, retries=2)


def test_reviewer_workflow_completes(review_result):
    """The reviewer workflow should complete without errors."""
    assert review_result is not None
    assert "outputs" in review_result


def test_reviewer_keyframes_generated(review_result):
    """The keyframe images should have been saved, proving MotionDirector ran."""
    images = get_output_images(review_result["outputs"], "6")
    assert len(images) > 0, "No keyframe images saved in reviewer workflow"


def test_reviewer_keyframes_nontrivial(review_result):
    """The saved keyframe images should have meaningful content."""
    images = get_output_images(review_result["outputs"], "6")
    assert len(images[0]) > 3000, f"Keyframe image too small: {len(images[0])} bytes"
