"""E2E test: Full pipeline from character generation to video output.

Stages (each is a separate ComfyUI workflow with restart between stages):
  A. Generate character (Z-Image Turbo)
  C. Extract pose → Generate keyframes (MotionDirector via local LLM)
  D. Edit ALL keyframes with VNCCS QWEN Encoder (matches 03_static_pose workflow)
  E. Assemble video with LTX Video 2.3 using edited character images as guides

Skipped unless RUN_E2E=1 env var is set.
"""

import io

import pytest
from PIL import Image

from conftest import (
    submit_workflow, get_output_images, get_output_video, upload_image,
    restart_comfyui,
)


pytestmark = [pytest.mark.e2e, pytest.mark.requires_api_key]

# VNCCS prompt text — matches the known-working 03_static_pose_qwen_edit workflow
VNCCS_INSTRUCTION = "Modify the character's pose to match the skeleton guide shown in Picture 2. Maintain the character's exact appearance, clothing, and identity."
VNCCS_PROMPT = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."


@pytest.fixture(scope="module")
def e2e_result(comfyui, llm_config):
    """Run the full E2E pipeline and return the final video workflow result."""
    llm_url, llm_model, llm_key = llm_config

    # ── Step A: Generate character ──────────────────────────────────────
    print("\n[E2E] Step A: Generating character with Z-Image Turbo...")
    char_workflow = {
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "6": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 1024, "batch_size": 1}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "Full body portrait, wide shot from head to toe. 90s OVA anime cel style. A cute young woman with long flowing pink hair, big sparkling blue eyes, wearing a cute sailor school uniform with pink ribbon. Standing with a cheerful smile, hands clasped together. Clean white background. Detailed linework, vibrant pastel colors.",
            "clip": ["6", 0],
        }},
        "11": {"class_type": "KSampler", "inputs": {
            "seed": 42, "steps": 8, "cfg": 1.0,
            "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0,
            "model": ["3", 0], "positive": ["10", 0], "negative": ["10", 0],
            "latent_image": ["5", 0],
        }},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["12", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": "e2e/character", "images": ["13", 0]}},
    }
    char_result = submit_workflow(char_workflow, timeout=600, retries=2)
    char_images = get_output_images(char_result["outputs"], "14")
    assert len(char_images) > 0, "E2E: No character image generated"
    assert len(char_images[0]) > 10000, f"E2E: Character image too small: {len(char_images[0])} bytes"
    upload_image(char_images[0], "e2e_character.png", "PoseDirector")
    print(f"[E2E] Step A done: character image {len(char_images[0])} bytes")

    # Restart ComfyUI to clear Z-Image model from VRAM
    restart_comfyui()

    # ── Step C: Extract pose + generate keyframes ──────────────────────
    print("[E2E] Step C: Extracting pose and generating motion keyframes...")
    kf_workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/e2e_character.png", "upload": "image"}},
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": "enable", "detect_body": "enable", "detect_face": "disable",
            "resolution": 1024,
        }},
        "3": {"class_type": "MotionDirector", "inputs": {
            "keypoints": "[]",
            "motion_description": "Character performs a cute anime dance, waving arms and shifting weight side to side with playful poses",
            "num_keyframes": 3,
            "total_frames": 25,
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
            "width": 1024,
            "height": 1024,
            "pose_keypoint": ["2", 1],
        }},
        "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "e2e/keyframes", "images": ["3", 0]}},
    }
    kf_result = submit_workflow(kf_workflow, timeout=600, retries=2)
    kf_images = get_output_images(kf_result["outputs"], "4")
    assert len(kf_images) > 0, "E2E: No keyframe images generated"
    print(f"[E2E] Step C done: {len(kf_images)} keyframe image(s)")

    # Upload each keyframe individually so Step D can load them by name
    for i, kf_bytes in enumerate(kf_images):
        upload_image(kf_bytes, f"e2e_kf_{i}.png", "PoseDirector")

    # Restart ComfyUI to clear OpenPose + MotionDirector from VRAM before loading Qwen Edit
    restart_comfyui()

    # ── Step D: Edit ALL keyframes with VNCCS QWEN Encoder ────────────
    # Matches the known-working 03_static_pose_qwen_edit.json workflow:
    #   GGUF model (same as official VN workflow), resolution 1024, 30 steps, VNCCS prompt text
    # VRAM: Qwen Edit UNET GGUF (~10GB) + CLIP (~7GB) + VAE (~1.5GB) = ~18GB sequential, fits 24GB
    edited_keyframes = []
    for kf_idx in range(len(kf_images)):
        print(f"[E2E] Step D: Editing keyframe {kf_idx} with VNCCS QWEN Encoder...")
        edit_workflow = {
            # Character image
            "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/e2e_character.png", "upload": "image"}},
            # Load this specific keyframe skeleton (uploaded from Step C)
            "21": {"class_type": "LoadImage", "inputs": {"image": f"PoseDirector/e2e_kf_{kf_idx}.png"}},

            # Qwen Image Edit models — GGUF (same as official VN workflow)
            "4": {"class_type": "UnetLoaderGGUF", "inputs": {
                "unet_name": "Qwen-Image-Edit-2511-Q5_0.gguf",
            }},
            "5": {"class_type": "CLIPLoader", "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image",
            }},
            "6": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},

            # VNCCS encoder: character + skeleton → edited character image
            # Settings match 03_static_pose_qwen_edit.json exactly
            "7": {"class_type": "VNCCS_QWEN_Encoder", "inputs": {
                "instruction": VNCCS_INSTRUCTION,
                "seed": 1,
                "image2": None, "image3": None, "reference_image_list": None,
                "image1_label": "Character Reference",
                "image2_label": "Pose Guide",
                "image3_label": "Picture 3",
                "resolution": 1024,
                "resize_method": "lanczos", "crop": "center",
                "image1_weight": 1.0, "image2_weight": 1.0, "image3_weight": 1.0,
                "start_step": 384,
                "prompt": VNCCS_PROMPT,
                "qwen_2511": True,
                "clip": ["5", 0], "vae": ["6", 0],
                "image1": ["1", 0],       # character
                "image2": ["21", 0],      # specific keyframe skeleton
            }},

            # KSampler — 30 steps (matches working workflow)
            "8": {"class_type": "KSampler", "inputs": {
                "seed": 42, "steps": 30, "cfg": 1.0,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1],
                "latent_image": ["7", 2],
            }},

            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["6", 0]}},
            "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"e2e/edited_kf{kf_idx}", "images": ["9", 0]}},
        }
        edit_result = submit_workflow(edit_workflow, timeout=900, retries=2)
        edited_imgs = get_output_images(edit_result["outputs"], "10")
        assert len(edited_imgs) > 0, f"E2E: No edited keyframe {kf_idx} generated"
        assert len(edited_imgs[0]) > 10000, f"E2E: Edited keyframe {kf_idx} too small: {len(edited_imgs[0])} bytes"
        edited_keyframes.append(edited_imgs[0])
        print(f"[E2E] Step D: keyframe {kf_idx} edited — {len(edited_imgs[0])} bytes")

    print(f"[E2E] Step D done: {len(edited_keyframes)} keyframes edited")

    # Restart ComfyUI to clear Qwen Edit models from VRAM before LTX Video loads
    restart_comfyui()

    # ── Step E: Assemble video with LTX Video 2.3 ─────────────────────
    print("[E2E] Step E: Assembling video with LTX Video 2.3...")

    # Upload edited character images as LTX video guides (NOT raw skeletons)
    for i, kf_bytes in enumerate(edited_keyframes):
        img = Image.open(io.BytesIO(kf_bytes)).convert("RGB")
        img = img.resize((256, 256), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        upload_image(buf.getvalue(), f"edited_kf_{i}.png", "PoseDirector")
        print(f"[E2E] Step E: uploaded edited keyframe {i} ({len(kf_bytes)} bytes)")

    video_workflow = {
        # Load LTX models — IQ4_XS Gemma (6.5GB) + Q4_K_M UNET (16.4GB) = fits 24GB with sequential loading
        "5": {"class_type": "UnetLoaderGGUF", "inputs": {
            "unet_name": "ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf",
        }},
        "6": {"class_type": "DualCLIPLoaderGGUF", "inputs": {
            "clip_name1": "gemma-3-12b-it-IQ4_XS.gguf",
            "clip_name2": "ltxv_text_projection_dual.safetensors",
            "type": "ltxv",
        }},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": "LTX23_video_vae_bf16.safetensors"}},

        # Text prompt
        "8": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "A cute anime girl performing a playful dance, waving arms and swaying. Smooth natural motion.",
            "clip": ["6", 0],
        }},

        # Latent video (9 frames at 256x256)
        "9": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": 256, "height": 256, "length": 9, "batch_size": 1,
        }},

        # Load edited character images (NOT raw skeleton keyframes)
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/edited_kf_0.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/edited_kf_1.png"}},

        # Guide keyframe at frame 0
        "10": {"class_type": "LTXVAddGuideAdvanced", "inputs": {
            "frame_idx": 0, "strength": 1.0,
            "crf": 29, "blur_radius": 0,
            "interpolation": "lanczos", "crop": "disabled",
            "positive": ["8", 0], "negative": ["8", 0],
            "vae": ["7", 0], "latent": ["9", 0], "image": ["1", 0],
        }},

        # Guide keyframe at frame 8
        "11": {"class_type": "LTXVAddGuideAdvanced", "inputs": {
            "frame_idx": 8, "strength": 1.0,
            "crf": 29, "blur_radius": 0,
            "interpolation": "lanczos", "crop": "disabled",
            "positive": ["10", 0], "negative": ["10", 1],
            "vae": ["7", 0], "latent": ["10", 2], "image": ["2", 0],
        }},

        # Sample
        "14": {"class_type": "KSampler", "inputs": {
            "seed": 42, "steps": 20, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            "model": ["5", 0], "positive": ["11", 0], "negative": ["11", 1],
            "latent_image": ["11", 2],
        }},

        # Decode
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["7", 0]}},

        # Combine to video
        "16": {"class_type": "VHS_VideoCombine", "inputs": {
            "filename_prefix": "e2e/output",
            "frame_rate": 24, "loop_count": 0,
            "format": "video/h264-mp4", "pingpong": False, "save_output": True,
            "images": ["15", 0],
        }},
    }
    video_result = submit_workflow(video_workflow, timeout=900, retries=2)
    print("[E2E] Step E done")
    return video_result


def test_e2e_produces_video(e2e_result):
    """E2E pipeline should produce a video file."""
    video = get_output_video(e2e_result["outputs"], "16")
    assert video is not None, "No video output from E2E pipeline"
    assert len(video) > 10000, f"Video too small: {len(video)} bytes"


def test_e2e_video_is_mp4(e2e_result):
    """The output video should be a valid MP4 file."""
    video = get_output_video(e2e_result["outputs"], "16")
    if video is None:
        pytest.skip("No video output")
    assert len(video) > 8
    assert b'ftyp' in video[:32], "Video doesn't appear to be a valid MP4"
