"""Integration test: PoseManipulator node (requires LLM API)."""

import pytest

from conftest import submit_workflow, get_output_images


pytestmark = [pytest.mark.integration, pytest.mark.requires_api_key]


@pytest.fixture(scope="module")
def manipulator_result(comfyui, character_image, llm_config):
    """Run OpenposePreprocessor -> PoseManipulator and return result.

    PoseManipulator outputs (IMAGE, STRING, POSE_KEYPOINT).
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
        "3": {"class_type": "PoseManipulator", "inputs": {
            "keypoints": "[]",
            "instruction": "Raise both arms above head",
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
            "width": 1024,
            "height": 1024,
            "pose_keypoint": ["2", 1],
        }},
        "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/manipulated_pose", "images": ["3", 0]}},
    }
    return submit_workflow(workflow, timeout=180, retries=2)


def test_manipulator_generates_image(manipulator_result):
    """PoseManipulator should produce a skeleton image."""
    images = get_output_images(manipulator_result["outputs"], "4")
    assert len(images) > 0, "No manipulated pose image generated"


def test_manipulator_image_nontrivial(manipulator_result):
    """The manipulated skeleton image should have non-trivial content."""
    images = get_output_images(manipulator_result["outputs"], "4")
    assert len(images[0]) > 3000, "Manipulated pose image too small"
