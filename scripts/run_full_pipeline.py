#!/usr/bin/env python3
"""Run the full Pose Director end-to-end pipeline.

Stages:
  A. Character generation (Z-Image Turbo)
  C. Pose extraction + MotionDirector keyframes
  D. Per-keyframe VNCCS QWEN editing
  E. LTX Video 2.3 assembly

Usage:
    export GEMINI_API_KEY=your_key
    python scripts/run_full_pipeline.py
"""

import io
import json
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image

COMFYUI_URL = "http://127.0.0.1:8465"
POLL_INTERVAL = 3
API_KEY = os.environ.get("GEMINI_API_KEY", "")


def log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


def submit_workflow(workflow: dict, timeout: int = 600) -> dict:
    """Submit workflow and wait for completion."""
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow},
        timeout=30,
    )
    assert resp.status_code == 200, f"Submit failed: {resp.status_code} {resp.text}"
    data = resp.json()
    prompt_id = data["prompt_id"]
    log(f"  submitted prompt {prompt_id}, waiting...")

    start = time.time()
    while time.time() - start < timeout:
        hist = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=30)
        if hist.status_code == 200 and hist.json().get(prompt_id):
            entry = hist.json()[prompt_id]
            if "outputs" in entry:
                return entry
            if entry.get("status", {}).get("status_str") == "error":
                msgs = entry["status"].get("messages", [])
                raise RuntimeError(f"Workflow failed: {json.dumps(msgs, indent=2)}")
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Workflow {prompt_id} timed out after {timeout}s")


def get_output_images(outputs: dict, node_id: str | int) -> list:
    """Fetch output images from a SaveImage/PreviewImage node."""
    node_id = str(node_id)
    images = []
    for img_info in outputs.get(node_id, {}).get("images", []):
        resp = requests.get(
            f"{COMFYUI_URL}/view",
            params={
                "filename": img_info["filename"],
                "subfolder": img_info.get("subfolder", ""),
                "type": img_info.get("type", "output"),
            },
            timeout=30,
        )
        assert resp.status_code == 200
        images.append(resp.content)
    return images


def get_output_text(outputs: dict, node_id: str | int) -> str | None:
    """Fetch STRING output from a node."""
    node_out = outputs.get(str(node_id), {})
    texts = node_out.get("text", [])
    return texts[0] if texts else None


def upload_image(image_bytes: bytes, filename: str, subfolder: str = "PoseDirector") -> None:
    """Upload an image to ComfyUI's input directory."""
    resp = requests.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (filename, image_bytes, "image/png")},
        data={"overwrite": "true", "subfolder": subfolder},
        timeout=30,
    )
    assert resp.status_code == 200, f"Upload failed: {resp.status_code} {resp.text}"


def stage_a_character() -> bytes:
    """Stage A: Generate a character image with Z-Image Turbo."""
    log("=" * 50)
    log("STAGE A: Character Generation")
    log("=" * 50)

    workflow = {
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "6": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 1024, "batch_size": 1}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "Full body portrait, wide shot from head to toe. 90s OVA anime cel style. A young woman with long blue hair, red eyes, wearing a white school uniform with blue trim. Standing casually with arms at sides, slight smile. Clean white background. Detailed linework, vibrant colors.",
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
        "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pipeline/character", "images": ["13", 0]}},
    }

    result = submit_workflow(workflow, timeout=300)
    images = get_output_images(result["outputs"], "14")
    assert len(images) > 0, "No character image generated"
    img = Image.open(io.BytesIO(images[0]))
    log(f"Character image: {img.size}, {len(images[0])} bytes")
    return images[0]


def stage_c_keyframes(character_png: bytes) -> list:
    """Stage C: Extract pose and generate motion keyframes via MotionDirector.
    Returns list of keypoint JSON strings (one per keyframe).
    """
    log("=" * 50)
    log("STAGE C: Pose Extraction + MotionDirector")
    log("=" * 50)

    upload_image(character_png, "character.png", "PoseDirector")

    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/character.png", "upload": "image"}},
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "mask": "disable", "hand": "enable", "body": "enable", "face": "disable",
            "resolution": 1024,
        }},
        "3": {"class_type": "MotionDirector", "inputs": {
            "keypoints": "[]",
            "motion_description": "Character waves their right hand in a friendly greeting",
            "num_keyframes": 4,
            "total_frames": 32,
            "base_url": "http://127.0.0.1:8001/v1",
            "api_key": "not-needed",
            "model": "Qwen3.5-2B-UD-Q4_K_XL.gguf",
            "width": 512,
            "height": 512,
            "pose_keypoint": ["2", 1],
        }},
        "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pipeline/keyframes", "images": ["3", 0]}},
    }

    result = submit_workflow(workflow, timeout=180)

    # Get keyframes JSON
    keyframes_json = get_output_text(result["outputs"], "3")
    assert keyframes_json, "No keyframes JSON returned"
    keyframes_data = json.loads(keyframes_json)
    keyframes = keyframes_data.get("keyframes", [])
    log(f"Generated {len(keyframes)} keyframes")

    # Download the saved keyframe images
    images = get_output_images(result["outputs"], "4")
    log(f"Downloaded {len(images)} keyframe images")
    for i, img_bytes in enumerate(images):
        upload_image(img_bytes, f"keyframe_{i}.png", "PoseDirector")
        log(f"  uploaded keyframe_{i}.png ({len(img_bytes)} bytes)")

    return keyframes


def stage_d_edit_keyframe(character_png: bytes, keyframe_idx: int) -> bytes:
    """Stage D: Edit a single keyframe with VNCCS QWEN Encoder."""
    log(f"  Editing keyframe {keyframe_idx}...")

    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/character.png", "upload": "image"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": f"PoseDirector/keyframe_{keyframe_idx}.png", "upload": "image"}},
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {
            "ckpt_name": "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui.safetensors",
        }},
        "4": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image",
        }},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "6": {"class_type": "VNCCS_QWEN_Encoder", "inputs": {
            "instruction": "Modify the character pose to match the skeleton guide shown in Picture 2. Maintain the character's exact appearance, clothing, and identity.",
            "seed": 1 + keyframe_idx,
            "image2": None,
            "image3": None,
            "reference_image_list": None,
            "image1_label": "Character Reference",
            "image2_label": "Pose Guide",
            "image3_label": "Picture 3",
            "resolution": 1024,
            "resize_method": "lanczos",
            "crop": "center",
            "image1_weight": 1.0,
            "image2_weight": 1.0,
            "image3_weight": 1.0,
            "start_step": 384,
            "prompt": "Describe the key features of the input image, then modify the character pose to match the skeleton guide while maintaining identity.",
            "qwen_2511": True,
            "clip": ["4", 0],
            "vae": ["5", 0],
            "image1": ["1", 0],
            "image2": ["2", 0],
        }},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": 42 + keyframe_idx, "steps": 30, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple",
            "model": ["3", 0],
            "positive": ["6", 0], "negative": ["6", 1],
            "latent_image": ["6", 2],
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["5", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"pipeline/edited_kf{keyframe_idx}", "images": ["8", 0]}},
    }

    result = submit_workflow(workflow, timeout=300)
    images = get_output_images(result["outputs"], "9")
    assert len(images) > 0, f"No edited image for keyframe {keyframe_idx}"
    upload_image(images[0], f"edited_kf{keyframe_idx}.png", "PoseDirector")
    log(f"  keyframe {keyframe_idx} edited: {len(images[0])} bytes")
    return images[0]


def stage_e_video() -> bytes:
    """Stage E: Assemble LTX Video from edited keyframes."""
    log("=" * 50)
    log("STAGE E: LTX Video Assembly")
    log("=" * 50)

    workflow = {
        # Load keyframes
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/edited_kf0.png", "upload": "image"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/edited_kf1.png", "upload": "image"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/edited_kf2.png", "upload": "image"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/edited_kf3.png", "upload": "image"}},

        # Load models
        "5": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf"}},
        "6": {"class_type": "LTXVGemmaCLIPModelLoader", "inputs": {
            "gemma_path": "gemma-3-12b-it/model.safetensors",
            "ltxv_path": "ltx-2.3_text_projection_bf16.safetensors",
            "max_token_length": 256,
        }},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": "taeltx2_3.safetensors"}},

        # Text prompt
        "8": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "A character waving their right hand in a friendly greeting. Smooth natural motion between poses.",
            "clip": ["6", 0],
        }},

        # Latent video
        "9": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 512, "height": 512, "length": 33, "batch_size": 1}},

        # Guide keyframes
        "10": {"class_type": "LTXVAddGuideAdvanced", "inputs": {
            "frame_idx": 0, "strength": 1.0, "blend": 29, "bias": 0,
            "scaling": "lanczos", "mask": "disabled",
            "positive": ["8", 0], "negative": ["8", 0],
            "vae": ["7", 0], "latent": ["9", 0], "image": ["1", 0],
        }},
        "11": {"class_type": "LTXVAddGuideAdvanced", "inputs": {
            "frame_idx": 8, "strength": 1.0, "blend": 29, "bias": 0,
            "scaling": "lanczos", "mask": "disabled",
            "positive": ["10", 0], "negative": ["10", 1],
            "vae": ["7", 0], "latent": ["10", 2], "image": ["2", 0],
        }},
        "12": {"class_type": "LTXVAddGuideAdvanced", "inputs": {
            "frame_idx": 16, "strength": 1.0, "blend": 29, "bias": 0,
            "scaling": "lanczos", "mask": "disabled",
            "positive": ["11", 0], "negative": ["11", 1],
            "vae": ["7", 0], "latent": ["11", 2], "image": ["3", 0],
        }},
        "13": {"class_type": "LTXVAddGuideAdvanced", "inputs": {
            "frame_idx": 24, "strength": 1.0, "blend": 29, "bias": 0,
            "scaling": "lanczos", "mask": "disabled",
            "positive": ["12", 0], "negative": ["12", 1],
            "vae": ["7", 0], "latent": ["12", 2], "image": ["4", 0],
        }},

        # Sample
        "14": {"class_type": "KSampler", "inputs": {
            "seed": 42, "steps": 30, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple",
            "model": ["5", 0], "positive": ["13", 0], "negative": ["13", 1],
            "latent_image": ["13", 2],
        }},

        # Decode
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["7", 0]}},

        # Combine to video
        "16": {"class_type": "VHSVideoCombine", "inputs": {
            "filename_prefix": "pipeline/output",
            "fps": 24, "loop_count": 0,
            "format": "auto", "codec": "libx264", "save_output": True,
            "images": ["15", 0],
        }},
    }

    result = submit_workflow(workflow, timeout=600)

    # Download video
    gifs = result["outputs"].get("16", {}).get("gifs", [])
    assert len(gifs) > 0, "No video output"
    resp = requests.get(
        f"{COMFYUI_URL}/view",
        params={
            "filename": gifs[0]["filename"],
            "subfolder": gifs[0].get("subfolder", ""),
            "type": gifs[0].get("type", "output"),
        },
        timeout=30,
    )
    assert resp.status_code == 200
    log(f"Video: {len(resp.content)} bytes")
    return resp.content


def main():
    if not API_KEY:
        log("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    # Verify ComfyUI is up
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        assert r.status_code == 200
    except Exception as e:
        log(f"ERROR: ComfyUI not reachable at {COMFYUI_URL}: {e}")
        sys.exit(1)

    log("ComfyUI is ready. Starting pipeline...")
    start_total = time.time()

    # Stage A
    t0 = time.time()
    character = stage_a_character()
    log(f"Stage A completed in {time.time()-t0:.1f}s")

    # Stage C
    t0 = time.time()
    keyframes = stage_c_keyframes(character)
    log(f"Stage C completed in {time.time()-t0:.1f}s")

    # Stage D
    t0 = time.time()
    for i in range(len(keyframes)):
        stage_d_edit_keyframe(character, i)
    log(f"Stage D completed in {time.time()-t0:.1f}s")

    # Stage E
    t0 = time.time()
    video = stage_e_video()
    log(f"Stage E completed in {time.time()-t0:.1f}s")

    # Save locally
    out_path = Path("/home/ubuntu/Documents/programs/img/comfyui-pose-director/pipeline_output.mp4")
    out_path.write_bytes(video)
    log(f"Video saved to: {out_path}")
    log(f"TOTAL TIME: {time.time()-start_total:.1f}s")
    log("Pipeline complete!")


if __name__ == "__main__":
    main()
