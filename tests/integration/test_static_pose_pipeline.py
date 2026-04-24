"""Integration test: Full static pose pipeline (character -> pose -> edited image).

Pipeline: LoadImage -> OpenposePreprocessor -> PoseManipulator -> VNCCS_QWEN_Encoder -> KSampler -> SaveImage
"""

import pytest

from conftest import submit_workflow, get_output_images
from pipeline.workflows import build_vnccs_edit_nodes


pytestmark = [pytest.mark.integration, pytest.mark.requires_api_key]


@pytest.fixture(scope="module")
def static_pose_result(comfyui, character_image, llm_config):
    """Run the full static pose pipeline and return result."""
    llm_url, llm_model, llm_key = llm_config
    workflow = {
        # Load character image
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/character.png", "upload": "image"}},

        # Extract pose
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": "enable", "detect_body": "enable", "detect_face": "disable",
            "resolution": 1024,
        }},

        # Manipulate pose via LLM
        "3": {"class_type": "PoseManipulator", "inputs": {
            "keypoints": "[]",
            "instruction": "Cross arms across chest",
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
            "width": 1024,
            "height": 1024,
            "pose_keypoint": ["2", 1],
        }},

        # Preview the skeleton
        "20": {"class_type": "PreviewImage", "inputs": {"filename_prefix": "pytest/static_skeleton", "images": ["3", 0]}},
    }

    # VNCCS encoder + sampler + decode + save (nodes 4-10)
    vnccs_nodes = build_vnccs_edit_nodes(
        model_node_id="4",
        clip_node_id="5",
        vae_node_id="6",
        encoder_node_id="7",
        ksampler_node_id="8",
        decode_node_id="9",
        save_node_id="10",
        image1_ref=("1", 0),   # character image
        image2_ref=("3", 0),   # skeleton image from PoseManipulator
        seed=42,
        steps=40,
        target_size=1024,
        filename_prefix="pytest/static_edited",
        use_gguf_unet=False,
    )
    workflow.update(vnccs_nodes)
    return submit_workflow(workflow, timeout=600, retries=2)


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
    images = get_output_images(static_pose_result["outputs"], "20")
    assert len(images) > 0, "No skeleton preview generated"
