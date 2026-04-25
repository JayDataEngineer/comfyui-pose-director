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
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file into environment (if present)
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

from pipeline.comfyui_client import (
    COMFYUI_URL,
    submit_workflow,
    get_output_images,
    get_output_video,
    upload_image,
)
from pipeline.workflows import (
    build_character_workflow,
    build_motion_director_workflow,
    build_vnccs_edit_workflow,
    build_ltx_video_workflow,
    VNCCS_PROMPT,
    VNCCS_INSTRUCTION,
    VNCCS_INSTRUCTION_MESH,
)

API_KEY = os.environ.get("GEMINI_API_KEY", "")
ZAI_KEY = os.environ.get("ZAI_API_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


def get_llm_config():
    """Return (base_url, model, api_key) for LLM calls."""
    # Prefer OpenRouter (cheap, fast, no VRAM)
    if OPENROUTER_KEY:
        log("Using OpenRouter (deepseek-v4-flash)")
        return ("https://openrouter.ai/api/v1", "deepseek/deepseek-v4-flash", OPENROUTER_KEY)

    # ZAI cloud (no VRAM cost)
    if ZAI_KEY:
        log("Using ZAI cloud API (glm-4.5-air)")
        return ("https://api.z.ai/api/coding/paas/v4", "glm-4.5-air", ZAI_KEY)

    # Try local llama.cpp
    try:
        r = requests.get("http://127.0.0.1:8001/health", timeout=2)
        if r.status_code == 200:
            models = requests.get("http://127.0.0.1:8001/v1/models", timeout=2).json()
            model_id = models["data"][0]["id"]
            log(f"Using local LLM: {model_id}")
            return ("http://127.0.0.1:8001/v1", model_id, "sk-no-key-required")
    except Exception:
        pass

    # Fall back to Gemini
    if API_KEY:
        log("Using Gemini API")
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "models/gemini-2.5-flash",
            API_KEY,
        )

    raise RuntimeError("No LLM available. Set OPENROUTER_API_KEY, ZAI_API_KEY, GEMINI_API_KEY, or start llama.cpp on :8001")


def stage_a_character() -> bytes:
    """Stage A: Generate a character image with Z-Image Turbo."""
    log("=" * 50)
    log("STAGE A: Character Generation")
    log("=" * 50)

    import random
    workflow = build_character_workflow(
        seed=random.randint(0, 999999),
        filename_prefix="pipeline/character",
    )
    result = submit_workflow(workflow)
    images = get_output_images(result["outputs"], "14")
    assert len(images) > 0, "No character image generated"
    img = Image.open(io.BytesIO(images[0]))
    log(f"Character image: {img.size}, {len(images[0])} bytes")
    return images[0]


def stage_c_keyframes(character_png: bytes) -> list:
    """Stage C: Extract pose and generate motion keyframes via MotionDirector.
    Uses rotation mode (Anny body mesh renders) for better VNCCS results.
    Returns list of keyframe image bytes.
    """
    log("=" * 50)
    log("STAGE C: Pose Extraction + MotionDirector (rotation mode)")
    log("=" * 50)

    upload_image(character_png, "character.png", "PoseDirector")

    llm_url, llm_model, llm_key = get_llm_config()
    workflow = build_motion_director_workflow(
        image_path="PoseDirector/character.png",
        motion_description="Character waves their right hand in a friendly greeting",
        num_keyframes=4,
        total_frames=32,
        llm_url=llm_url,
        llm_key=llm_key,
        llm_model=llm_model,
        width=1024,
        height=1024,
        filename_prefix="pipeline/keyframes",
        output_mode="rotations",
    )

    result = submit_workflow(workflow)

    # Download the saved keyframe images
    images = get_output_images(result["outputs"], "4")
    log(f"Generated {len(images)} keyframe images")
    for i, img_bytes in enumerate(images):
        upload_image(img_bytes, f"keyframe_{i}.png", "PoseDirector")
        log(f"  uploaded keyframe_{i}.png ({len(img_bytes)} bytes)")

    return images


def stage_d_edit_keyframes(character_png: bytes, keyframe_images: list) -> list:
    """Stage D: Edit all keyframes with VNCCS QWEN Encoder.
    Uses mesh-specific instruction for body mesh keyframe inputs.
    """
    log("=" * 50)
    log("STAGE D: VNCCS QWEN Editing (body mesh mode)")
    log("=" * 50)

    edited = []
    for i, kf_bytes in enumerate(keyframe_images):
        log(f"  Editing keyframe {i}...")
        workflow = build_vnccs_edit_workflow(
            character_image_path="PoseDirector/character.png",
            keyframe_image_path=f"PoseDirector/keyframe_{i}.png",
            seed=42 + i,
            steps=40,
            prompt=VNCCS_PROMPT,
            instruction=VNCCS_INSTRUCTION_MESH,
            filename_prefix=f"pipeline/edited_kf{i}",
        )
        result = submit_workflow(workflow)
        images = get_output_images(result["outputs"], "10")
        assert len(images) > 0, f"No edited image for keyframe {i}"
        upload_image(images[0], f"edited_kf{i}.png", "PoseDirector")
        log(f"  keyframe {i} edited: {len(images[0])} bytes")
        edited.append(images[0])

    return edited


def stage_e_video(edited_keyframes: list) -> bytes:
    """Stage E: Assemble LTX Video from edited keyframes."""
    log("=" * 50)
    log("STAGE E: LTX Video Assembly")
    log("=" * 50)

    # Upload edited keyframes for LTX guide injection
    for i, kf_bytes in enumerate(edited_keyframes):
        upload_image(kf_bytes, f"edited_kf_{i}.png", "PoseDirector")

    image_paths = [f"PoseDirector/edited_kf_{i}.png" for i in range(len(edited_keyframes))]
    workflow = build_ltx_video_workflow(
        keyframe_image_paths=image_paths,
        seed=42,
        width=512,
        height=512,
        length=33,
        steps=30,
        fps=24,
        filename_prefix="pipeline/output",
    )

    result = submit_workflow(workflow)
    video = get_output_video(result["outputs"], "16")
    assert video is not None, "No video output"
    log(f"Video: {len(video)} bytes")
    return video


def main():
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
    keyframe_images = stage_c_keyframes(character)
    log(f"Stage C completed in {time.time()-t0:.1f}s")

    # Stage D
    t0 = time.time()
    edited_keyframes = stage_d_edit_keyframes(character, keyframe_images)
    log(f"Stage D completed in {time.time()-t0:.1f}s")

    # Stage E
    t0 = time.time()
    video = stage_e_video(edited_keyframes)
    log(f"Stage E completed in {time.time()-t0:.1f}s")

    # Save locally
    out_path = Path("/home/ubuntu/Documents/programs/img/comfyui-pose-director/pipeline_output.mp4")
    out_path.write_bytes(video)
    log(f"Video saved to: {out_path}")
    log(f"TOTAL TIME: {time.time()-start_total:.1f}s")
    log("Pipeline complete!")


if __name__ == "__main__":
    main()
