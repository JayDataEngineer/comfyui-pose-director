"""E2E test: Full pipeline from character generation to video output.

Stages:
  A. Generate character (Z-Image Turbo)
  C. Extract pose → Generate keyframes (MotionDirector via local LLM)
  D. Edit keyframe 0 with VNCCS QWEN Encoder
  E. Assemble video with LTX Video 2.3

Skipped unless RUN_E2E=1 env var is set.
"""

import pytest
import requests
import time

from conftest import submit_workflow, get_output_images, get_output_video, upload_image


pytestmark = [pytest.mark.e2e, pytest.mark.requires_api_key]


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

    # ── Step C: Extract pose + generate keyframes ──────────────────────
    print("[E2E] Step C: Extracting pose and generating motion keyframes...")
    kf_workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/e2e_character.png", "upload": "image"}},
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": "enable", "detect_body": "enable", "detect_face": "disable",
            "resolution": 512,
        }},
        "3": {"class_type": "MotionDirector", "inputs": {
            "keypoints": "[]",
            "motion_description": "Character performs a cute anime dance, waving arms and shifting weight side to side with playful poses",
            "num_keyframes": 3,
            "total_frames": 25,
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
            "width": 512,
            "height": 512,
            "pose_keypoint": ["2", 1],
        }},
        "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "e2e/keyframes", "images": ["3", 0]}},
    }
    kf_result = submit_workflow(kf_workflow, timeout=600, retries=2)
    kf_images = get_output_images(kf_result["outputs"], "4")
    assert len(kf_images) > 0, "E2E: No keyframe images generated"
    print(f"[E2E] Step C done: {len(kf_images)} keyframe image(s)")

    # ── Step D: Edit keyframe 0 with VNCCS QWEN Encoder ───────────────
    print("[E2E] Step D: Editing keyframe 0 with VNCCS QWEN Encoder...")
    edit_workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/e2e_character.png", "upload": "image"}},
        # Re-extract pose + generate keyframes (self-contained workflow)
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": "enable", "detect_body": "enable", "detect_face": "disable",
            "resolution": 512,
        }},
        "3": {"class_type": "MotionDirector", "inputs": {
            "keypoints": "[]",
            "motion_description": "Character performs a cute anime dance, waving arms and shifting weight side to side with playful poses",
            "num_keyframes": 3,
            "total_frames": 25,
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
            "width": 512,
            "height": 512,
            "pose_keypoint": ["2", 1],
        }},
        # Split batch to get individual keyframes
        "20": {"class_type": "ImpactImageBatchToImageList", "inputs": {"image": ["3", 0]}},
        # Load Qwen Image Edit models (GGUF for VRAM efficiency)
        "6": {"class_type": "UnetLoaderGGUF", "inputs": {
            "unet_name": "Qwen-Image-Edit-2511-Q5_0.gguf",
        }},
        "7": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image",
        }},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        # VNCCS encoder: character + skeleton → edited image
        "9": {"class_type": "VNCCS_QWEN_Encoder", "inputs": {
            "instruction": "Modify the character pose to match the dance skeleton guide, keeping the cute anime style.",
            "seed": 1,
            "image2": None, "image3": None, "reference_image_list": None,
            "image1_label": "Character", "image2_label": "Skeleton", "image3_label": "None",
            "resolution": 512, "resize_method": "lanczos", "crop": "center",
            "image1_weight": 1.0, "image2_weight": 1.0, "image3_weight": 1.0,
            "start_step": 384,
            "prompt": "Modify the character pose to match the dance skeleton guide while maintaining identity and cute anime style.",
            "qwen_2511": True,
            "clip": ["7", 0], "vae": ["8", 0],
            "image1": ["1", 0],
            "image2": ["20", 0],  # first keyframe from batch split
        }},
        "10": {"class_type": "KSampler", "inputs": {
            "seed": 42, "steps": 15, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            "model": ["6", 0], "positive": ["9", 0], "negative": ["9", 1],
            "latent_image": ["9", 2],
        }},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["8", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "e2e/edited_kf0", "images": ["11", 0]}},
    }
    edit_result = submit_workflow(edit_workflow, timeout=900, retries=2)
    edited_images = get_output_images(edit_result["outputs"], "12")
    assert len(edited_images) > 0, "E2E: No edited keyframe image generated"
    assert len(edited_images[0]) > 10000, f"E2E: Edited image too small: {len(edited_images[0])} bytes"
    print(f"[E2E] Step D done: edited keyframe {len(edited_images[0])} bytes")

    # ── Step E: Assemble video with LTX Video 2.3 ─────────────────────
    print("[E2E] Step E: Assembling video with LTX Video 2.3...")
    # The LTX pipeline (22B GGUF UNET + 12B Gemma text encoder) needs ~20GB VRAM.
    # Must free VRAM from both Qwen Edit models AND the llama.cpp LLM server.
    import subprocess, os
    print("[E2E] Step E: Stopping llama.cpp to free 17GB VRAM...")
    subprocess.run(["docker", "stop", "orchestrator-llama"], check=False, timeout=30)
    time.sleep(3)
    # Also restart ComfyUI with clean VRAM (unload Qwen Edit from Step D)
    print("[E2E] Step E: Restarting ComfyUI for clean VRAM...")
    subprocess.run(["pkill", "-f", "main.py.*8465"], check=False)
    time.sleep(8)  # Wait for CUDA memory to be released
    cui_dir = "/home/ubuntu/Documents/programs/img/comfyui"
    env = {**os.environ, "PYTHONWARNINGS": "ignore::FutureWarning,ignore::SyntaxWarning"}
    log_f = open("/tmp/cui_e2e_step_e.log", "w")
    subprocess.Popen(
        [os.path.join(cui_dir, ".venv/bin/python"), "main.py",
         "--port", "8465", "--use-flash-attention", "--dont-upcast-attention",
         "--novram"],
        cwd=cui_dir, stdout=log_f, stderr=subprocess.STDOUT, env=env,
    )
    for _ in range(60):
        time.sleep(2)
        try:
            r = requests.get("http://127.0.0.1:8465/system_stats", timeout=2)
            if r.status_code == 200:
                print("[E2E] Step E: ComfyUI restarted OK")
                break
        except Exception:
            pass
    else:
        pytest.skip("ComfyUI failed to restart for Step E")
    time.sleep(2)  # extra settling time

    # Upload keyframe images to ComfyUI input so LoadImage can find them
    # Resize to 256x256 to match the video latent dimensions
    from PIL import Image
    import io
    for i, kf_bytes in enumerate(kf_images):
        img = Image.open(io.BytesIO(kf_bytes)).convert("RGB")
        img = img.resize((256, 256), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        upload_image(buf.getvalue(), f"kf_{i}.png", "PoseDirector")
    video_workflow = {
        # Load LTX models
        "5": {"class_type": "UnetLoaderGGUF", "inputs": {
            "unet_name": "ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf",
        }},
        "6": {"class_type": "LTXVGemmaCLIPModelLoader", "inputs": {
            "gemma_path": "gemma-3-12b-it/model.safetensors",
            "ltxv_path": "ltx-2.3-connectors-textproj.safetensors",
            "max_length": 256,
        }},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": "LTX23_video_vae_bf16.safetensors"}},

        # Text prompt
        "8": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "A cute anime girl performing a playful dance, waving arms and swaying. Smooth natural motion.",
            "clip": ["6", 0],
        }},

        # Latent video (9 frames at 256x256 to fit in VRAM alongside GGUF UNET)
        "9": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": 256, "height": 256, "length": 9, "batch_size": 1,
        }},

        # Load keyframe skeleton images (uploaded from step C output)
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/kf_0.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/kf_1.png"}},

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

        # Sample (few steps for E2E speed)
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
    # Restart llama.cpp LLM server (stopped to free VRAM for Step E)
    subprocess.run(["docker", "start", "orchestrator-llama"], check=False, timeout=30)
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
