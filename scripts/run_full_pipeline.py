#!/usr/bin/env python3
"""Run the full Pose Director end-to-end pipeline using UI workflow files.

Stages:
  A. Character generation (Z-Image Turbo) — workflows/00_z_image_character.json
  C. Pose extraction + MotionDirector keyframes — workflows/01_motion_keyframes_rotation.json
  D. Per-keyframe VNCCS QWEN editing — workflows/02_vnccs_edit_keyframe.json
  E. LTX Video 2.3 assembly — workflows/04_ltx_video_assembly.json

Usage:
    export GEMINI_API_KEY=your_key
    python scripts/run_full_pipeline.py
"""

import io
import os
import sys
import time
from datetime import datetime
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
from pipeline.ui_to_api import (
    load_ui_workflow,
    patch_ui_workflow,
    patch_connection,
    convert_ui_to_api,
    save_ui_workflow,
    save_api_workflow,
)

API_KEY = os.environ.get("GEMINI_API_KEY", "")
ZAI_KEY = os.environ.get("ZAI_API_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

WORKFLOWS_DIR = PROJECT_ROOT / "workflows"
COMFYUI_OUTPUT_DIR = Path("/home/ubuntu/Documents/programs/img/comfyui/output")

# Default prompts/instructions
VNCCS_PROMPT = "Draw character from image2"
VNCCS_INSTRUCTION_MESH = (
    "Modify the character's pose to match the body shape and pose shown in Picture 2. "
    "Picture 2 shows a 3D body mesh rendering — match the posture, limb positions, and overall body orientation. "
    "Maintain the character's exact appearance, clothing, and identity."
)
DEFAULT_LTX_PROMPT = (
    "A cute anime girl performing a playful dance, waving arms and swaying. "
    "Smooth natural motion. Clean scene with no text, letters, signs, or watermarks."
)


def log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


def get_llm_config():
    """Return (base_url, model, api_key) for LLM calls."""
    if OPENROUTER_KEY:
        log("Using OpenRouter (deepseek-v4-flash)")
        return ("https://openrouter.ai/api/v1", "deepseek/deepseek-v4-flash", OPENROUTER_KEY)

    if ZAI_KEY:
        log("Using ZAI cloud API (glm-4.5-air)")
        return ("https://api.z.ai/api/coding/paas/v4", "glm-4.5-air", ZAI_KEY)

    try:
        r = requests.get("http://127.0.0.1:8001/health", timeout=2)
        if r.status_code == 200:
            models = requests.get("http://127.0.0.1:8001/v1/models", timeout=2).json()
            model_id = models["data"][0]["id"]
            log(f"Using local LLM: {model_id}")
            return ("http://127.0.0.1:8001/v1", model_id, "sk-no-key-required")
    except Exception:
        pass

    if API_KEY:
        log("Using Gemini API")
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "models/gemini-2.5-flash",
            API_KEY,
        )

    raise RuntimeError("No LLM available. Set OPENROUTER_API_KEY, ZAI_API_KEY, GEMINI_API_KEY, or start llama.cpp on :8001")


def _setup_run_dir() -> Path:
    """Create a timestamped run directory under project root."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["stage_a", "stage_c", "stage_d", "stage_e"]:
        (run_dir / sub).mkdir(exist_ok=True)
    return run_dir


def _save_stage_artifacts(
    run_dir: Path,
    stage_name: str,
    ui_workflow: dict,
    api_workflow: dict,
    outputs: list[bytes] | bytes | None = None,
) -> None:
    """Save the patched UI workflow, API workflow, and any output files."""
    stage_dir = run_dir / stage_name
    save_ui_workflow(stage_dir / "workflow_ui.json", ui_workflow)
    save_api_workflow(stage_dir / "workflow_api.json", api_workflow)
    if outputs is not None:
        if isinstance(outputs, list):
            for i, data in enumerate(outputs):
                suffix = ".png" if data[:4] == b"\x89PNG" else ".mp4"
                (stage_dir / f"output_{i}{suffix}").write_bytes(data)
        else:
            suffix = ".png" if outputs[:4] == b"\x89PNG" else ".mp4"
            (stage_dir / f"output{suffix}").write_bytes(outputs)


def stage_a_character(run_dir: Path) -> bytes:
    """Stage A: Generate a character image with Z-Image Turbo."""
    log("=" * 50)
    log("STAGE A: Character Generation")
    log("=" * 50)

    import random
    workflow = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")

    # Fix the missing VAEDecode→SaveImage connection if needed
    save_node = next((n for n in workflow["nodes"] if n["type"] == "SaveImage"), None)
    vaedecode_node = next((n for n in workflow["nodes"] if n["type"] == "VAEDecode"), None)
    if save_node and vaedecode_node and save_node.get("inputs", [{}])[0].get("link") is None:
        patch_connection(
            workflow,
            target_node_id=save_node["id"],
            target_input_name="images",
            source_node_id=vaedecode_node["id"],
            source_output_index=0,
        )

    patch_ui_workflow(workflow, {
        "KSampler": {"seed": random.randint(0, 999999)},
        "SaveImage": {"filename_prefix": "pipeline/character"},
    })

    api = convert_ui_to_api(workflow)
    result = submit_workflow(api)

    # Find SaveImage node id in API
    save_node_id = str(save_node["id"])
    images = get_output_images(result["outputs"], save_node_id)
    assert len(images) > 0, "No character image generated"
    img = Image.open(io.BytesIO(images[0]))
    log(f"Character image: {img.size}, {len(images[0])} bytes")

    _save_stage_artifacts(run_dir, "stage_a", workflow, api, images[0])
    return images[0]


def stage_c_keyframes(run_dir: Path) -> list:
    """Stage C: Generate motion keyframes via MotionDirector (rotation mode)."""
    log("=" * 50)
    log("STAGE C: MotionDirector (rotation mode)")
    log("=" * 50)

    llm_url, llm_model, llm_key = get_llm_config()
    workflow = load_ui_workflow(WORKFLOWS_DIR / "01_motion_keyframes_rotation.json")
    patch_ui_workflow(workflow, {
        "MotionDirector": {
            "motion_description": "Character waves their right hand in a friendly greeting",
            "num_keyframes": 4,
            "total_frames": 32,
            "base_url": llm_url,
            "model": llm_model,
        },
        "PreviewImage": {"filename_prefix": "pipeline/keyframes"},
    })

    api = convert_ui_to_api(workflow)
    result = submit_workflow(api)

    # Find the first PreviewImage node (keyframe output, node 4)
    save_node = next(n for n in workflow["nodes"] if n["type"] == "PreviewImage")
    save_node_id = str(save_node["id"])
    images = get_output_images(result["outputs"], save_node_id)
    log(f"Generated {len(images)} keyframe images")
    for i, img_bytes in enumerate(images):
        upload_image(img_bytes, f"keyframe_{i}.png", "PoseDirector")
        log(f"  uploaded keyframe_{i}.png ({len(img_bytes)} bytes)")

    _save_stage_artifacts(run_dir, "stage_c", workflow, api, images)
    return images


def stage_d_edit_keyframes(character_png: bytes, keyframe_images: list, run_dir: Path) -> list:
    """Stage D: Edit all keyframes with VNCCS QWEN Encoder."""
    log("=" * 50)
    log("STAGE D: VNCCS QWEN Editing (body mesh mode)")
    log("=" * 50)

    upload_image(character_png, "character.png", "PoseDirector")
    edited = []

    for i, kf_bytes in enumerate(keyframe_images):
        log(f"  Editing keyframe {i}...")
        upload_image(kf_bytes, f"keyframe_{i}.png", "PoseDirector")

        workflow = load_ui_workflow(WORKFLOWS_DIR / "02_vnccs_edit_keyframe.json")
        patch_ui_workflow(workflow, {
            "LoadImage": [
                {"image": "PoseDirector/character.png"},
                {"image": f"PoseDirector/keyframe_{i}.png"},
            ],
            "KSampler": {"seed": 42 + i, "steps": 40},
            "VNCCS_QWEN_Encoder": {
                "prompt": VNCCS_PROMPT,
                "instruction": VNCCS_INSTRUCTION_MESH,
            },
            "SaveImage": {"filename_prefix": f"pipeline/edited_kf{i}"},
        })

        api = convert_ui_to_api(workflow)
        result = submit_workflow(api)

        save_node = next(n for n in workflow["nodes"] if n["type"] == "SaveImage")
        save_node_id = str(save_node["id"])
        images = get_output_images(result["outputs"], save_node_id)
        assert len(images) > 0, f"No edited image for keyframe {i}"
        upload_image(images[0], f"edited_kf{i}.png", "PoseDirector")
        log(f"  keyframe {i} edited: {len(images[0])} bytes")
        edited.append(images[0])

        _save_stage_artifacts(run_dir, "stage_d", workflow, api, images[0])

    return edited


def stage_e_video(edited_keyframes: list, run_dir: Path) -> bytes:
    """Stage E: Assemble LTX Video from edited keyframes."""
    log("=" * 50)
    log("STAGE E: LTX Video Assembly")
    log("=" * 50)

    for i, kf_bytes in enumerate(edited_keyframes):
        upload_image(kf_bytes, f"edited_kf_{i}.png", "PoseDirector")

    workflow = load_ui_workflow(WORKFLOWS_DIR / "04_ltx_video_assembly.json")

    # Patch LoadImage nodes with edited keyframe paths
    loadimage_patches = [
        {"image": f"PoseDirector/edited_kf_{i}.png"}
        for i in range(len(edited_keyframes))
    ]
    patch_ui_workflow(workflow, {
        "LoadImage": loadimage_patches,
        "CLIPTextEncode": {"text": DEFAULT_LTX_PROMPT},
        "KSampler": {"seed": 42, "steps": 30},
        "VHS_VideoCombine": {"filename_prefix": "pipeline/output"},
    })

    api = convert_ui_to_api(workflow)
    result = submit_workflow(api)

    vhs_node = next(n for n in workflow["nodes"] if n["type"] == "VHS_VideoCombine")
    vhs_node_id = str(vhs_node["id"])
    video = get_output_video(result["outputs"], vhs_node_id)
    assert video is not None, "No video output"
    log(f"Video: {len(video)} bytes")

    _save_stage_artifacts(run_dir, "stage_e", workflow, api, video)
    return video


def main():
    # Verify ComfyUI is up
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        assert r.status_code == 200
    except Exception as e:
        log(f"ERROR: ComfyUI not reachable at {COMFYUI_URL}: {e}")
        sys.exit(1)

    run_dir = _setup_run_dir()
    log(f"Run directory: {run_dir}")
    log("ComfyUI is ready. Starting pipeline...")
    start_total = time.time()

    # Stage A
    t0 = time.time()
    character = stage_a_character(run_dir)
    log(f"Stage A completed in {time.time()-t0:.1f}s")

    # Stage C
    t0 = time.time()
    keyframe_images = stage_c_keyframes(run_dir)
    log(f"Stage C completed in {time.time()-t0:.1f}s")

    # Stage D
    t0 = time.time()
    edited_keyframes = stage_d_edit_keyframes(character, keyframe_images, run_dir)
    log(f"Stage D completed in {time.time()-t0:.1f}s")

    # Stage E
    t0 = time.time()
    video = stage_e_video(edited_keyframes, run_dir)
    log(f"Stage E completed in {time.time()-t0:.1f}s")

    # Save final video to run dir
    final_path = run_dir / "pipeline_output.mp4"
    final_path.write_bytes(video)
    log(f"Video saved to: {final_path}")
    log(f"TOTAL TIME: {time.time()-start_total:.1f}s")
    log("Pipeline complete!")


if __name__ == "__main__":
    main()
