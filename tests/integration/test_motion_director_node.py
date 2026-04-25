"""Integration test: MotionDirector node (requires LLM API)."""

import pytest

from conftest import submit_workflow, get_output_images


pytestmark = [pytest.mark.integration, pytest.mark.requires_api_key]


@pytest.fixture(scope="module")
def motion_result(comfyui, llm_config):
    """Run MotionDirector (rotation-only) and return result.

    MotionDirector outputs (IMAGE, STRING, STRING).
    Only IMAGE is tracked in ComfyUI history via the downstream SaveImage node.
    """
    llm_url, llm_model, llm_key = llm_config
    workflow = {
        "1": {"class_type": "MotionDirector", "inputs": {
            "motion_description": "Character waves their right hand in a friendly greeting",
            "num_keyframes": 4,
            "total_frames": 32,
            "base_url": llm_url,
            "model": llm_model,
            "width": 1024,
            "height": 1024,
        }},
        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/motion_keyframes", "images": ["1", 0]}},
    }
    return submit_workflow(workflow, timeout=180, retries=2)


def test_motion_director_generates_image(motion_result):
    """MotionDirector should produce a batch of body mesh images captured by SaveImage."""
    images = get_output_images(motion_result["outputs"], "2")
    assert len(images) > 0, "No keyframe images generated"


def test_motion_director_image_nontrivial(motion_result):
    """The generated keyframe images should have non-trivial content."""
    images = get_output_images(motion_result["outputs"], "2")
    assert len(images[0]) > 5000, f"Keyframe image too small: {len(images[0])} bytes"
