#!/usr/bin/env python3
"""Batch sprite pipeline: run multiple poses and directions for a character.

Usage:
    # Use preset animation definitions
    python scripts/run_sprite_batch.py --character PoseDirector/character.png

    # Single animation, specific directions
    python scripts/run_sprite_batch.py --animations idle --directions front right

    # Custom poses from PoseLibrary
    python scripts/run_sprite_batch.py --poses 3 Seeking exercises1

    # HY-Motion NPZ → sprite frames
    python scripts/run_sprite_batch.py --hymotion output/hymotion_wave.npz --directions front
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

VNCCS_BONE_NAMES = {
    "upperarm_r", "lowerarm_r", "hand_r",
    "upperarm_l", "lowerarm_l", "hand_l",
    "thigh_r", "calf_r", "foot_r",
    "thigh_l", "calf_l", "foot_l",
    "spine_01", "spine_02", "spine_03",
    "neck_01", "head", "pelvis",
}

DIRECTION_ROTATIONS = {
    "front": 0.0,
    "right": 90.0,
    "back": 180.0,
    "left": 270.0,
}

# Preset animation definitions using PoseLibrary poses
ANIMATION_PRESETS = {
    "idle": {
        "description": "Subtle idle animation",
        "fps": 8,
        "loop": True,
        "poses": ["exercises1"],
    },
    "wave": {
        "description": "Waving greeting",
        "fps": 8,
        "loop": True,
        "poses": ["Hi"],
    },
    "action": {
        "description": "Dynamic action poses",
        "fps": 12,
        "loop": False,
        "poses": ["3", "Seeking"],
    },
    "combat": {
        "description": "Combat ready stances",
        "fps": 12,
        "loop": False,
        "poses": ["11", "12", "13"],
    },
    "expressive": {
        "description": "Expressive character poses",
        "fps": 10,
        "loop": False,
        "poses": ["5", "8", "9", "exercises2"],
    },
}


def log(msg: str):
    print(f"[batch] {msg}", flush=True)


def load_pose(pose_name: str) -> dict:
    pose_path = POSE_LIBRARY / f"{pose_name}.json"
    if not pose_path.exists():
        available = [p.stem for p in POSE_LIBRARY.glob("*.json")]
        raise FileNotFoundError(f"Pose '{pose_name}' not found. Available: {available}")
    with open(pose_path) as f:
        return json.load(f)


def pose_to_rotations_json(pose: dict) -> str:
    bones = pose.get("bones", {})
    filtered = {k: v for k, v in bones.items() if k in VNCCS_BONE_NAMES}
    return json.dumps(filtered)


def run_single_frame(
    character_image: str,
    rotations_json: str,
    direction: str,
    seed: int,
) -> bytes | None:
    """Run a single sprite frame through the pipeline. Returns image bytes."""
    model_rotation_y = DIRECTION_ROTATIONS.get(direction, 0.0)

    workflow = load_ui_workflow(WORKFLOWS_DIR / "05_sprite_bodymesh_edit.json")
    patch_ui_workflow(workflow, {
        "LoadImage": {"image": character_image},
        "BodyMeshRenderer": {
            "rotations_json": rotations_json,
            "model_rotation_y": model_rotation_y,
        },
        "VNCCS_QWEN_Encoder": {
            "prompt": VNCCS_PROMPT,
            "instruction": VNCCS_INSTRUCTION,
        },
        "KSampler": {"seed": seed},
        "SaveImage": {"filename_prefix": "PoseDirector/sprite"},
    })

    api = convert_ui_to_api(workflow)
    result = submit_workflow(api)

    save_node = next(n for n in workflow["nodes"] if n["type"] == "SaveImage")
    save_node_id = str(save_node["id"])
    images = get_output_images(result["outputs"], save_node_id)
    return images[0] if images else None


def remove_background(img_bytes: bytes) -> bytes:
    """Remove background from image using rembg. Returns RGBA PNG bytes."""
    from rembg import remove
    img = Image.open(io.BytesIO(img_bytes))
    result = remove(img)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def run_hymotion_batch(
    npz_path: str | Path,
    character_image: str = "PoseDirector/character.png",
    character_bytes: bytes | None = None,
    directions: list[str] | None = None,
    num_keyframes: int = 6,
    seed: int | None = None,
    remove_bg: bool = False,
    animation_name: str = "hymotion",
):
    """Run sprite pipeline from HY-Motion NPZ output.

    Extracts keyframes from the NPZ, converts to Anny bone rotations,
    and renders each keyframe through the VNCCS edit pipeline.
    """
    from utils.hymotion_converter import extract_keyframes

    if seed is None:
        seed = random.randint(0, 999999)
    if directions is None:
        directions = ["front"]

    npz_path = Path(npz_path)
    if not npz_path.exists():
        log(f"ERROR: HY-Motion NPZ not found: {npz_path}")
        sys.exit(1)

    log(f"Extracting {num_keyframes} keyframes from {npz_path.name}")
    keyframes = extract_keyframes(npz_path, num_keyframes)
    log(f"Extracted {len(keyframes)} keyframes with {len(keyframes[0])} joints each")

    # Build job list: (animation_name, keyframe_index, direction, rotations_dict)
    jobs: list[tuple[str, int, str, dict]] = []
    for kf_idx, rotations in enumerate(keyframes):
        for d in directions:
            jobs.append((animation_name, kf_idx, d, rotations))

    # Setup output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / f"batch_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Verify ComfyUI is up
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        assert r.status_code == 200
    except Exception as e:
        log(f"ERROR: ComfyUI not reachable at {COMFYUI_URL}: {e}")
        sys.exit(1)
    log(f"ComfyUI ready at {COMFYUI_URL}")

    if character_bytes:
        upload_image(character_bytes, "character.png", "PoseDirector")
        log(f"Uploaded character image ({len(character_bytes)} bytes)")

    log(f"HY-Motion batch: {len(jobs)} frames ({len(keyframes)} keyframes x {len(directions)} directions)")
    start_total = time.time()

    results = []
    for i, (anim_name, kf_idx, direction, rotations) in enumerate(jobs):
        frame_label = f"[{i+1}/{len(jobs)}] kf{kf_idx}/{direction}"
        log(f"{frame_label} ...")

        # Convert Anny bone names to VNCCS bone names for the pipeline
        rotations_json = json.dumps(rotations)
        frame_seed = seed + i
        t0 = time.time()

        try:
            img_bytes = run_single_frame(character_image, rotations_json, direction, frame_seed)
        except Exception as e:
            log(f"  ERROR: {e}")
            continue

        elapsed = time.time() - t0

        if img_bytes:
            if remove_bg:
                try:
                    img_bytes = remove_background(img_bytes)
                except Exception as e:
                    log(f"  BG removal failed ({e}), saving with background")

            anim_dir = run_dir / anim_name
            anim_dir.mkdir(exist_ok=True)
            out_path = anim_dir / f"kf{kf_idx:03d}_{direction}.png"
            out_path.write_bytes(img_bytes)

            img = Image.open(io.BytesIO(img_bytes))
            log(f"  OK: {img.size[0]}x{img.size[1]} {img.mode}, {elapsed:.1f}s")
            results.append({
                "animation": anim_name,
                "keyframe": kf_idx,
                "direction": direction,
                "path": str(out_path),
                "size": list(img.size),
                "transparent": remove_bg,
            })
        else:
            log(f"  ERROR: No output image")

    total_elapsed = time.time() - start_total

    manifest = {
        "character": character_image,
        "source": str(npz_path),
        "keyframes": len(keyframes),
        "seed_base": seed,
        "total_frames": len(results),
        "total_time_s": round(total_elapsed, 1),
        "frames": results,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"Manifest: {manifest_path}")

    log(f"Done: {len(results)}/{len(jobs)} frames in {total_elapsed:.1f}s")
    log(f"Output: {run_dir}")
    return run_dir


def run_batch(
    character_image: str = "PoseDirector/character.png",
    character_bytes: bytes | None = None,
    animations: list[str] | None = None,
    poses: list[str] | None = None,
    directions: list[str] | None = None,
    seed: int | None = None,
    remove_bg: bool = False,
):
    """Run batch sprite generation for multiple animations/directions."""
    if seed is None:
        seed = random.randint(0, 999999)

    if directions is None:
        directions = ["front"]

    # Build job list: list of (animation_name, pose_name, direction)
    jobs: list[tuple[str, str, str]] = []

    if animations:
        for anim_name in animations:
            preset = ANIMATION_PRESETS.get(anim_name)
            if not preset:
                log(f"WARNING: Unknown animation '{anim_name}', skipping")
                continue
            for pose_name in preset["poses"]:
                for d in directions:
                    jobs.append((anim_name, pose_name, d))
    elif poses:
        for pose_name in poses:
            for d in directions:
                jobs.append(("custom", pose_name, d))
    else:
        log("ERROR: Specify --animations, --poses, or --hymotion")
        sys.exit(1)

    if not jobs:
        log("ERROR: No jobs to run")
        sys.exit(1)

    # Setup output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / f"batch_{ts}"
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

    log(f"Batch: {len(jobs)} frames ({len(set(j[0] for j in jobs))} animations, {len(directions)} directions)")
    start_total = time.time()

    results = []
    for i, (anim_name, pose_name, direction) in enumerate(jobs):
        frame_label = f"[{i+1}/{len(jobs)}] {anim_name}/{pose_name}/{direction}"
        log(f"{frame_label} ...")

        try:
            pose = load_pose(pose_name)
            rotations_json = pose_to_rotations_json(pose)
        except FileNotFoundError as e:
            log(f"  SKIP: {e}")
            continue

        frame_seed = seed + i
        t0 = time.time()

        try:
            img_bytes = run_single_frame(character_image, rotations_json, direction, frame_seed)
        except Exception as e:
            log(f"  ERROR: {e}")
            continue

        elapsed = time.time() - t0

        if img_bytes:
            # Background removal (optional)
            if remove_bg:
                try:
                    img_bytes = remove_background(img_bytes)
                except Exception as e:
                    log(f"  BG removal failed ({e}), saving with background")

            # Save to animation directory
            anim_dir = run_dir / anim_name
            anim_dir.mkdir(exist_ok=True)
            out_path = anim_dir / f"{pose_name}_{direction}.png"
            out_path.write_bytes(img_bytes)

            img = Image.open(io.BytesIO(img_bytes))
            log(f"  OK: {img.size[0]}x{img.size[1]} {img.mode}, {elapsed:.1f}s")
            results.append({
                "animation": anim_name,
                "pose": pose_name,
                "direction": direction,
                "path": str(out_path),
                "size": list(img.size),
                "transparent": remove_bg,
            })
        else:
            log(f"  ERROR: No output image")

    total_elapsed = time.time() - start_total

    # Save manifest
    manifest = {
        "character": character_image,
        "seed_base": seed,
        "total_frames": len(results),
        "total_time_s": round(total_elapsed, 1),
        "frames": results,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"Manifest: {manifest_path}")

    log(f"Done: {len(results)}/{len(jobs)} frames in {total_elapsed:.1f}s")
    log(f"Output: {run_dir}")
    return run_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch sprite pipeline")
    parser.add_argument("--character", default="PoseDirector/character.png", help="Character image")
    parser.add_argument("--animations", nargs="+", default=None,
                       help=f"Animation presets: {list(ANIMATION_PRESETS.keys())}")
    parser.add_argument("--poses", nargs="+", default=None, help="Custom pose names from PoseLibrary")
    parser.add_argument("--hymotion", default=None, help="HY-Motion NPZ file to convert to sprites")
    parser.add_argument("--num-keyframes", type=int, default=6, help="Keyframes to extract from HY-Motion")
    parser.add_argument("--hymotion-name", default="hymotion", help="Animation name for HY-Motion output")
    parser.add_argument("--directions", nargs="+", default=["front"],
                       help="Directions: front right back left")
    parser.add_argument("--remove-bg", action="store_true", help="Remove background (transparent sprites)")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    args = parser.parse_args()

    if args.hymotion:
        run_hymotion_batch(
            npz_path=args.hymotion,
            character_image=args.character,
            directions=args.directions,
            num_keyframes=args.num_keyframes,
            seed=args.seed,
            remove_bg=args.remove_bg,
            animation_name=args.hymotion_name,
        )
    else:
        run_batch(
            character_image=args.character,
            animations=args.animations,
            poses=args.poses,
            directions=args.directions,
            seed=args.seed,
            remove_bg=args.remove_bg,
        )


if __name__ == "__main__":
    main()
