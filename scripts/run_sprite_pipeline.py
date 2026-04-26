#!/usr/bin/env python3
"""Run the sprite pipeline end-to-end: BodyMeshRenderer → VNCCS_QWEN_Editor → sprite frame.

Usage:
    python scripts/run_sprite_pipeline.py [--pose Hi] [--character PoseDirector/character.png]
    python scripts/run_sprite_pipeline.py --pose 3 --direction right
"""

import io
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
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
    upload_image,
)
from pipeline.ui_to_api import (
    load_ui_workflow,
    patch_ui_workflow,
    convert_ui_to_api,
    save_ui_workflow,
    save_api_workflow,
)

import requests
from PIL import Image

WORKFLOWS_DIR = PROJECT_ROOT / "workflows"
POSE_LIBRARY = Path("/home/ubuntu/Documents/programs/img/comfyui/custom_nodes/vnccs-utils/PoseLibrary")

VNCCS_PROMPT = "Draw character from image2"
VNCCS_INSTRUCTION = (
    "Match the body pose shown in Picture 1 (3D body mesh). "
    "Picture 2 is the character to draw. Picture 3 shows the skeleton overlay. "
    "Replicate the exact pose, limb positions, and body orientation from Picture 1 "
    "while maintaining the character's identity, clothing, and appearance."
)

# VNCCS (MakeHuman) bone names that have direct Anny equivalents.
# anny_compat.py handles the name mapping; we just pass bones through.
# Bones not in this set (finger bones, etc.) are silently skipped.
VNCCS_BONE_NAMES = {
    "upperarm_r", "lowerarm_r", "hand_r",
    "upperarm_l", "lowerarm_l", "hand_l",
    "thigh_r", "calf_r", "foot_r",
    "thigh_l", "calf_l", "foot_l",
    "spine_01", "spine_02", "spine_03",
    "neck_01", "head", "pelvis",
}


def log(msg: str):
    print(f"[sprite] {msg}", flush=True)


def load_pose(pose_name: str) -> dict:
    """Load a pose from VNCCS PoseLibrary."""
    pose_path = POSE_LIBRARY / f"{pose_name}.json"
    if not pose_path.exists():
        available = [p.stem for p in POSE_LIBRARY.glob("*.json")]
        raise FileNotFoundError(
            f"Pose '{pose_name}' not found. Available: {available}"
        )
    with open(pose_path) as f:
        return json.load(f)


def pose_to_rotations_json(pose: dict) -> str:
    """Convert VNCCS PoseLibrary pose to Anny-compatible rotations JSON.

    VNCCS bones (upperarm_r, lowerarm_r, etc.) are handled by anny_compat.py's
    JOINT_NAME_TO_ANNY mapping. Finger bones and other unrecognized names are
    silently skipped during the Anny forward pass.
    """
    bones = pose.get("bones", {})
    # Filter to only bones anny_compat can handle (skip fingers, etc.)
    filtered = {k: v for k, v in bones.items() if k in VNCCS_BONE_NAMES}
    log(f"  Bones: {len(bones)} total, {len(filtered)} mapped to Anny")
    return json.dumps(filtered)


def run_sprite_pipeline(
    character_image: str = "PoseDirector/character.png",
    character_bytes: bytes | None = None,
    pose_name: str = "Hi",
    seed: int | None = None,
    rotations_override: str | None = None,
):
    """Run the sprite pipeline: BodyMeshRenderer → VNCCS Edit → sprite frame."""
    if seed is None:
        seed = random.randint(0, 999999)

    # Setup output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / f"sprite_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Verify ComfyUI is up
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        assert r.status_code == 200
    except Exception as e:
        log(f"ERROR: ComfyUI not reachable at {COMFYUI_URL}: {e}")
        sys.exit(1)
    log(f"ComfyUI ready at {COMFYUI_URL}")

    # Upload character image if provided
    if character_bytes:
        upload_image(character_bytes, "character.png", "PoseDirector")
        log(f"Uploaded character image ({len(character_bytes)} bytes)")

    # Build rotations JSON
    if rotations_override:
        rotations_json = rotations_override
        log(f"Using override rotations: {len(json.loads(rotations_override))} joints")
    else:
        pose = load_pose(pose_name)
        log(f"Loaded pose '{pose_name}': {len(pose.get('bones', {}))} bones")
        rotations_json = pose_to_rotations_json(pose)

    # Load and patch workflow
    workflow = load_ui_workflow(WORKFLOWS_DIR / "05_sprite_bodymesh_edit.json")
    patch_ui_workflow(workflow, {
        "LoadImage": {"image": character_image},
        "BodyMeshRenderer": {"rotations_json": rotations_json},
        "VNCCS_QWEN_Encoder": {
            "prompt": VNCCS_PROMPT,
            "instruction": VNCCS_INSTRUCTION,
        },
        "KSampler": {"seed": seed},
        "SaveImage": {"filename_prefix": "PoseDirector/sprite"},
    })

    # Save UI workflow for debugging
    save_ui_workflow(run_dir / "workflow_ui.json", workflow)

    # Convert to API format
    api = convert_ui_to_api(workflow)
    save_api_workflow(run_dir / "workflow_api.json", api)

    # Submit
    log("Submitting workflow to ComfyUI...")
    start = time.time()
    result = submit_workflow(api)
    elapsed = time.time() - start
    log(f"Workflow completed in {elapsed:.1f}s")

    # Get output images
    save_node = next(n for n in workflow["nodes"] if n["type"] == "SaveImage")
    save_node_id = str(save_node["id"])
    images = get_output_images(result["outputs"], save_node_id)

    if not images:
        log("ERROR: No output images")
        return None

    # Save sprite frames
    for i, img_bytes in enumerate(images):
        img_path = run_dir / f"sprite_frame_{i}.png"
        img_path.write_bytes(img_bytes)
        img = Image.open(io.BytesIO(img_bytes))
        log(f"  Frame {i}: {img.size[0]}x{img.size[1]}, {len(img_bytes)} bytes")

    log(f"Output: {run_dir}")
    return run_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run sprite pipeline E2E")
    parser.add_argument("--pose", default="Hi", help="Pose name from VNCCS PoseLibrary")
    parser.add_argument("--character", default="PoseDirector/character.png", help="Character image path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    run_sprite_pipeline(
        character_image=args.character,
        pose_name=args.pose,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
