"""Integration test: Full static pose pipeline (character -> pose -> edited image).

Pipeline: LoadImage -> OpenposePreprocessor -> PoseManipulator -> VNCCS_QWEN_Encoder -> KSampler -> SaveImage
"""

import pytest

from conftest import submit_workflow, get_output_images


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

        # Load Qwen Image Edit models
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui.safetensors",
        }},
        "5": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image",
        }},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},

        # VNCCS QWEN Encoder -- character + skeleton -> edited image
        "7": {"class_type": "VNCCS_QWEN_Encoder", "inputs": {
            "prompt": "Draw character from image2",
            "latent_image_index": 1,
            "image3": None,
            "image1_name": "Character Reference",
            "image2_name": "Pose Guide",
            "image3_name": "Picture 3",
            "target_size": 1024,
            "upscale_method": "lanczos",
            "crop_method": "center",
            "weight1": 1.0,
            "weight2": 1.0,
            "weight3": 1.0,
            "vl_size": 384,
            "instruction": "Modify the character's pose to match the skeleton guide shown in Picture 2. Maintain the character's exact appearance, clothing, and identity.",
            "qwen_2511": True,
            "clip": ["5", 0],
            "vae": ["6", 0],
            "image1": ["1", 0],   # character image
            "image2": ["3", 0],   # skeleton image
        }},

        # Sample
        "8": {"class_type": "KSampler", "inputs": {
            "seed": 42, "steps": 15, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["7", 0], "negative": ["7", 1],
            "latent_image": ["7", 2],
        }},

        # Decode
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["6", 0]}},

        # Save
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/static_edited", "images": ["9", 0]}},
    }
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
