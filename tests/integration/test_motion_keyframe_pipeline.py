"""Integration test: Motion keyframe pipeline (character -> keyframes -> edited images).

Pipeline: LoadImage -> OpenposePreprocessor -> MotionDirector -> batch split ->
          per-keyframe VNCCS_QWEN_Encoder -> KSampler -> SaveImage

Note: ImpactImageBatchToImageList outputs OUTPUT_IS_LIST=(True,), which means
consuming individual items from it in API format requires index notation.
We use the simpler approach of saving the full batch and then using just the
first keyframe for the VNCCS QWEN Encoder.
"""

import pytest

from conftest import submit_workflow, get_output_images
from pipeline.workflows import build_vnccs_edit_nodes


pytestmark = [pytest.mark.integration, pytest.mark.requires_api_key]


@pytest.fixture(scope="module")
def motion_keyframe_result(comfyui, character_image, llm_config):
    """Run the motion keyframe pipeline and return result."""
    llm_url, llm_model, llm_key = llm_config
    workflow = {
        # Load character
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/character.png", "upload": "image"}},

        # Extract pose
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": "enable", "detect_body": "enable", "detect_face": "disable",
            "resolution": 512,
        }},

        # Generate keyframes via LLM
        "3": {"class_type": "MotionDirector", "inputs": {
            "keypoints": "[]",
            "motion_description": "Character waves their right hand",
            "num_keyframes": 2,
            "total_frames": 16,
            "base_url": llm_url,
            "model": llm_model,
            "width": 512,
            "height": 512,
            "pose_keypoint": ["2", 1],
        }},

        # Preview keyframes batch
        "4": {"class_type": "PreviewImage", "inputs": {"filename_prefix": "pytest/mkf_keyframes", "images": ["3", 0]}},

        # Split batch into individual images
        "5": {"class_type": "ImpactImageBatchToImageList", "inputs": {"image": ["3", 0]}},
    }

    # VNCCS encoder + sampler + decode + save (nodes 6-12)
    vnccs_nodes = build_vnccs_edit_nodes(
        model_node_id="6",
        clip_node_id="7",
        vae_node_id="8",
        encoder_node_id="9",
        ksampler_node_id="10",
        decode_node_id="11",
        save_node_id="12",
        image1_ref=("1", 0),   # character image
        image2_ref=("5", 0),   # first keyframe from batch split
        seed=42,
        steps=40,
        target_size=512,
        filename_prefix="pytest/mkf_keyframe_0",
        use_gguf_unet=False,
    )
    workflow.update(vnccs_nodes)
    return submit_workflow(workflow, timeout=600, retries=2)


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
