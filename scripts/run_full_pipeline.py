#!/usr/bin/env python3
"""Run the full Pose Director end-to-end pipeline using UI workflow files.

Stages:
  1. Character  — Z-Image Turbo generates a character portrait
  2. Motion     — HY-Motion 1.0 generates skeleton keyframes from text
  3. Edit       — VNCCS QWEN edits character pose into each keyframe
  4. Video      — LTX Video 2.3 assembles edited keyframes into MP4

Usage:
    python scripts/run_full_pipeline.py
"""

import io
import os
import random
import subprocess
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

WORKFLOWS_DIR = PROJECT_ROOT / "workflows"

# Default prompts/instructions
VNCCS_PROMPT = "Draw character from image2"
VNCCS_INSTRUCTION_MESH = (
    "Match the body pose shown in Picture 1 (3D body mesh). "
    "Picture 2 is the character to draw. Picture 3 shows the skeleton overlay. "
    "Replicate the exact pose, limb positions, and body orientation from Picture 1 "
    "while maintaining the character's identity, clothing, and appearance."
)
DEFAULT_MOTION_DESCRIPTION = "A person waves their right hand in a friendly greeting"
DEFAULT_LTX_PROMPT = (
    "A cute anime girl performing a playful dance, waving arms and swaying. "
    "Smooth natural motion. Clean scene with no text, letters, signs, or watermarks."
)

# Stage directory names
DIR_CHARACTER = "1_character"
DIR_MOTION = "2_motion"
DIR_EDIT = "3_edit"
DIR_VIDEO = "4_video"


def log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


def ensure_vram():
    """Stop llama.cpp container if running to free 19GB VRAM for HY-Motion."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", "orchestrator-llama"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip() == "true":
            log("Stopping llama.cpp container (frees 19GB VRAM)...")
            subprocess.run(["docker", "update", "--restart=no", "orchestrator-llama"],
                           capture_output=True, timeout=10)
            subprocess.run(["docker", "stop", "orchestrator-llama"],
                           capture_output=True, timeout=30)
            time.sleep(3)
            log("llama.cpp stopped")
    except Exception:
        pass  # Docker not available or container doesn't exist


def _setup_run_dir() -> Path:
    """Create a timestamped run directory under project root."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in [DIR_CHARACTER, DIR_MOTION, DIR_EDIT, DIR_VIDEO]:
        (run_dir / sub).mkdir(exist_ok=True)
    return run_dir


def _save_artifacts(
    stage_dir: Path,
    ui_workflow: dict,
    api_workflow: dict,
    *,
    images: dict[str, bytes] | None = None,
    video: bytes | None = None,
    extra_files: dict[str, bytes] | None = None,
) -> None:
    """Save stage artifacts with human-readable names.

    Args:
        stage_dir: The stage subdirectory (e.g. run_dir / "1_character").
        ui_workflow: Patched UI workflow dict.
        api_workflow: Converted API workflow dict.
        images: Optional {filename: bytes} for image outputs.
        video: Optional video bytes.
        extra_files: Optional {filename: bytes} for any other files.
    """
    save_ui_workflow(stage_dir / "workflow_ui.json", ui_workflow)
    save_api_workflow(stage_dir / "workflow_api.json", api_workflow)
    if images:
        for name, data in images.items():
            (stage_dir / name).write_bytes(data)
    if video:
        suffix = ".mp4"
        (stage_dir / f"video{suffix}").write_bytes(video)
    if extra_files:
        for name, data in extra_files.items():
            (stage_dir / name).write_bytes(data)


# ---------------------------------------------------------------------------
# Stage 1: Character
# ---------------------------------------------------------------------------

def stage_character(run_dir: Path) -> bytes:
    """Generate a character portrait with Z-Image Turbo."""
    log("=" * 50)
    log("STAGE 1: Character Generation")
    log("=" * 50)

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

    seed = random.randint(0, 999999)
    patch_ui_workflow(workflow, {
        "KSampler": {"seed": seed},
        "SaveImage": {"filename_prefix": "pipeline/character"},
    })

    api = convert_ui_to_api(workflow)
    result = submit_workflow(api)

    save_node_id = str(save_node["id"])
    images = get_output_images(result["outputs"], save_node_id)
    assert len(images) > 0, "No character image generated"
    img = Image.open(io.BytesIO(images[0]))
    log(f"Character: {img.size[0]}x{img.size[1]}, {len(images[0])} bytes, seed={seed}")

    _save_artifacts(
        run_dir / DIR_CHARACTER, workflow, api,
        images={"character.png": images[0]},
    )
    return images[0]


# ---------------------------------------------------------------------------
# Stage 2: Motion
# ---------------------------------------------------------------------------

def stage_motion(run_dir: Path, motion_description: str = DEFAULT_MOTION_DESCRIPTION) -> list:
    """Generate motion keyframes via HY-Motion 1.0 (text-to-motion DiT)."""
    log("=" * 50)
    log("STAGE 2: HY-Motion Keyframes")
    log("=" * 50)

    seed = random.randint(0, 999999)
    workflow = load_ui_workflow(WORKFLOWS_DIR / "01_hymotion_keyframes.json")
    patch_ui_workflow(workflow, {
        "HYMotionEncodeText": {"text": motion_description},
        "HYMotionGenerate": {"seed": seed, "duration": 3.0},
        "HYMotionSaveNPZ": {"output_dir": "hymotion_npz", "filename_prefix": "motion"},
        "PreviewImage": {"filename_prefix": "pipeline/keyframes"},
    })

    api = convert_ui_to_api(workflow)
    result = submit_workflow(api)

    # Extract skeleton preview images from PreviewImage node
    preview_node = next(n for n in workflow["nodes"] if n["type"] == "PreviewImage")
    preview_node_id = str(preview_node["id"])
    images = get_output_images(result["outputs"], preview_node_id)
    log(f"Generated {len(images)} keyframe images, seed={seed}")

    # Upload for downstream stages
    image_files: dict[str, bytes] = {}
    for i, img_bytes in enumerate(images):
        upload_image(img_bytes, f"keyframe_{i}.png", "PoseDirector")
        image_files[f"keyframe_{i}.png"] = img_bytes
        log(f"  keyframe_{i}.png ({len(img_bytes)} bytes)")

    _save_artifacts(
        run_dir / DIR_MOTION, workflow, api,
        images=image_files,
    )
    return images


# ---------------------------------------------------------------------------
# Stage 3: Edit
# ---------------------------------------------------------------------------

def stage_edit(character_png: bytes, keyframe_images: list, run_dir: Path) -> list:
    """Edit each keyframe with VNCCS QWEN Encoder to apply character pose."""
    log("=" * 50)
    log("STAGE 3: VNCCS Pose Editing")
    log("=" * 50)

    upload_image(character_png, "character.png", "PoseDirector")
    edited: list[bytes] = []

    # All keyframe edits go into the same stage directory
    stage_dir = run_dir / DIR_EDIT
    image_files: dict[str, bytes] = {}

    for i, kf_bytes in enumerate(keyframe_images):
        log(f"  Editing keyframe {i}/{len(keyframe_images) - 1}...")
        upload_image(kf_bytes, f"keyframe_{i}.png", "PoseDirector")

        workflow = load_ui_workflow(WORKFLOWS_DIR / "02_vnccs_edit_keyframe.json")
        patch_ui_workflow(workflow, {
            "LoadImage": [
                {"image": "PoseDirector/character.png"},
                {"image": f"PoseDirector/keyframe_{i}.png"},
            ],
            "KSampler": {"seed": 42 + i},
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
        out_images = get_output_images(result["outputs"], save_node_id)
        assert len(out_images) > 0, f"No edited image for keyframe {i}"
        upload_image(out_images[0], f"edited_kf{i}.png", "PoseDirector")
        log(f"  edited_{i}.png ({len(out_images[0])} bytes)")

        image_files[f"edited_{i}.png"] = out_images[0]
        edited.append(out_images[0])

    # Save all edited keyframes + last workflow snapshot
    _save_artifacts(
        stage_dir, workflow, api,
        images=image_files,
    )
    return edited


# ---------------------------------------------------------------------------
# Stage 4: Video
# ---------------------------------------------------------------------------

def stage_video(edited_keyframes: list, run_dir: Path) -> bytes:
    """Assemble LTX Video from edited keyframes."""
    log("=" * 50)
    log("STAGE 4: LTX Video Assembly")
    log("=" * 50)

    for i, kf_bytes in enumerate(edited_keyframes):
        upload_image(kf_bytes, f"edited_kf_{i}.png", "PoseDirector")

    workflow = load_ui_workflow(WORKFLOWS_DIR / "04_ltx_video_assembly.json")

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

    _save_artifacts(
        run_dir / DIR_VIDEO, workflow, api,
        video=video,
    )
    return video


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Verify ComfyUI is up
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        assert r.status_code == 200
    except Exception as e:
        log(f"ERROR: ComfyUI not reachable at {COMFYUI_URL}: {e}")
        sys.exit(1)

    # Free VRAM by stopping llama.cpp
    ensure_vram()

    run_dir = _setup_run_dir()
    log(f"Run directory: {run_dir}")
    log("ComfyUI is ready. Starting pipeline...")
    start_total = time.time()

    # Stage 1: Character
    t0 = time.time()
    character = stage_character(run_dir)
    log(f"Character done in {time.time() - t0:.1f}s")

    # Stage 2: Motion
    t0 = time.time()
    keyframe_images = stage_motion(run_dir)
    log(f"Motion done in {time.time() - t0:.1f}s")

    # Stage 3: Edit
    t0 = time.time()
    edited_keyframes = stage_edit(character, keyframe_images, run_dir)
    log(f"Edit done in {time.time() - t0:.1f}s")

    # Stage 4: Video
    t0 = time.time()
    video = stage_video(edited_keyframes, run_dir)
    log(f"Video done in {time.time() - t0:.1f}s")

    # Save final video at run root
    (run_dir / "result.mp4").write_bytes(video)
    elapsed = time.time() - start_total
    log(f"Result: {run_dir / 'result.mp4'}")
    log(f"TOTAL: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    log("Pipeline complete!")


if __name__ == "__main__":
    main()
