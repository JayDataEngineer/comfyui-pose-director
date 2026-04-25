"""Integration test: Motion keyframe pipeline (character -> keyframes -> edited images).

Pipeline: LoadImage -> OpenposePreprocessor -> MotionDirector -> batch split ->
          per-keyframe VNCCS_QWEN_Encoder -> KSampler -> SaveImage
"""

import pytest

from conftest import (
    submit_workflow, get_output_images,
    load_ui_workflow, patch_ui_workflow, convert_ui_to_api,
    WORKFLOWS_DIR,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_api_key]


@pytest.fixture(scope="module")
def motion_keyframe_result(comfyui, character_image, llm_config):
    """Run the motion keyframe pipeline and return result."""
    llm_url, llm_model, llm_key = llm_config
    workflow = load_ui_workflow(WORKFLOWS_DIR / "03_motion_keyframes_qwen_edit_mesh.json")
    patch_ui_workflow(workflow, {
        "LoadImage": {"image": "PoseDirector/character.png"},
        "MotionDirector": {
            "motion_description": "Character waves their right hand",
            "num_keyframes": 2,
            "total_frames": 16,
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
        },
        "KSampler": {"seed": 42, "steps": 40},
        "SaveImage": {"filename_prefix": "pytest/mkf_keyframe_0"},
        "PreviewImage": {"filename_prefix": "pytest/mkf_keyframes"},
    })
    api = convert_ui_to_api(workflow)
    return submit_workflow(api, timeout=600, retries=2)


def test_motion_keyframe_produces_edited_image(motion_keyframe_result):
    """Motion keyframe pipeline should produce at least one edited keyframe."""
    images = get_output_images(motion_keyframe_result["outputs"], "12")
    assert len(images) > 0, "No edited keyframe image produced"


def test_motion_keyframe_image_reasonable_size(motion_keyframe_result):
    """The edited keyframe image should be reasonably sized."""
    images = get_output_images(motion_keyframe_result["outputs"], "12")
    assert len(images[0]) > 10000, f"Edited keyframe too small: {len(images[0])} bytes"


def test_motion_keyframe_preview_generated(motion_keyframe_result):
    """The keyframes preview should have been generated."""
    images = get_output_images(motion_keyframe_result["outputs"], "4")
    assert len(images) > 0, "No keyframes preview generated"
