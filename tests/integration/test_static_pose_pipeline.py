"""Integration test: Full static pose pipeline (character -> pose -> edited image).

Pipeline: LoadImage -> OpenposePreprocessor -> PoseManipulator -> VNCCS_QWEN_Encoder -> KSampler -> SaveImage
"""

import pytest

from conftest import (
    submit_workflow, get_output_images,
    load_ui_workflow, patch_ui_workflow, convert_ui_to_api,
    WORKFLOWS_DIR,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_api_key]


@pytest.fixture(scope="module")
def static_pose_result(comfyui, character_image, llm_config):
    """Run the full static pose pipeline and return result."""
    llm_url, llm_model, llm_key = llm_config
    workflow = load_ui_workflow(WORKFLOWS_DIR / "02_vnccs_edit_mesh.json")
    patch_ui_workflow(workflow, {
        "LoadImage": {"image": "PoseDirector/character.png"},
        "PoseManipulator": {
            "instruction": "Cross arms across chest",
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
        },
        "KSampler": {"seed": 42, "steps": 40},
        "SaveImage": {"filename_prefix": "pytest/static_edited"},
        "PreviewImage": {"filename_prefix": "pytest/static_skeleton"},
    })
    api = convert_ui_to_api(workflow)
    return submit_workflow(api, timeout=600, retries=2)


def test_static_pipeline_produces_edited_image(static_pose_result):
    """Full static pipeline should produce an edited character image."""
    images = get_output_images(static_pose_result["outputs"], "10")
    assert len(images) > 0, "No edited image produced"


def test_static_pipeline_edited_image_reasonable_size(static_pose_result):
    """The edited image should be reasonably sized."""
    images = get_output_images(static_pose_result["outputs"], "10")
    assert len(images[0]) > 10000, f"Edited image too small: {len(images[0])} bytes"


def test_static_pipeline_skeleton_preview(static_pose_result):
    """The skeleton preview should have been generated."""
    images = get_output_images(static_pose_result["outputs"], "11")
    assert len(images) > 0, "No skeleton preview generated"
