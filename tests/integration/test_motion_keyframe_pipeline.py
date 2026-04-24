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
            "api_key": llm_key,
            "model": llm_model,
            "width": 512,
            "height": 512,
            "pose_keypoint": ["2", 1],
        }},

        # Preview keyframes batch
        "4": {"class_type": "PreviewImage", "inputs": {"filename_prefix": "pytest/mkf_keyframes", "images": ["3", 0]}},

        # Split batch into individual images
        "5": {"class_type": "ImpactImageBatchToImageList", "inputs": {"image": ["3", 0]}},

        # Load Qwen Image Edit models
        "6": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui.safetensors",
        }},
        "7": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image",
        }},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},

        # VNCCS QWEN Encoder for keyframe 0
        "9": {"class_type": "VNCCS_QWEN_Encoder", "inputs": {
            "prompt": "Draw character from image2",
            "latent_image_index": 1,
            "image3": None,
            "image1_name": "Character Reference",
            "image2_name": "Pose Guide",
            "image3_name": "Picture 3",
            "target_size": 512,
            "upscale_method": "lanczos",
            "crop_method": "center",
            "weight1": 1.0,
            "weight2": 1.0,
            "weight3": 1.0,
            "vl_size": 384,
            "instruction": "Modify the character's pose to match the skeleton guide shown in Picture 2. Maintain the character's exact appearance, clothing, and identity.",
            "qwen_2511": True,
            "clip": ["7", 0],
            "vae": ["8", 0],
            "image1": ["1", 0],
            "image2": ["5", 0],
        }},

        # KSampler for keyframe 0
        "10": {"class_type": "KSampler", "inputs": {
            "seed": 42, "steps": 15, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple",
            "denoise": 1.0,
            "model": ["6", 0],
            "positive": ["9", 0], "negative": ["9", 1],
            "latent_image": ["9", 2],
        }},

        # Decode keyframe 0
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["8", 0]}},

        # Save keyframe 0
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/mkf_keyframe_0", "images": ["11", 0]}},
    }
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
