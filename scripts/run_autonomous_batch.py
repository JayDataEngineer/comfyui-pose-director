#!/usr/bin/env python3
"""Autonomous batch pipeline with vision evaluation and retry loops.

Orchestrates the full pipeline (character → motion → edit → sprites)
for multiple NPCs with automatic quality checking between stages.

Usage:
    # From NPC definition YAML
    python scripts/run_autonomous_batch.py --npcs npcs.yaml

    # Single NPC inline
    python scripts/run_autonomous_batch.py --name dark_mage --description "hooded figure, dark robes, glowing eyes"

    # Skip character generation (use existing portrait)
    python scripts/run_autonomous_batch.py --npcs npcs.yaml --skip-character
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
from utils.pose_evaluator import (
    evaluate_keyframe_renders,
    is_evaluation_passing,
)

import requests
from PIL import Image

WORKFLOWS_DIR = PROJECT_ROOT / "workflows"

# Animation presets for game characters
ANIMATION_DEFS = {
    "idle": {"description": "subtle breathing, slight weight shift", "frames": 4, "fps": 8, "loop": True},
    "walk": {"description": "natural walk cycle, full stride", "frames": 6, "fps": 12, "loop": True},
    "attack": {"description": "overhead strike, wind-up to follow-through", "frames": 8, "fps": 12, "loop": False},
    "hurt": {"description": "knocked back, stumbling, recovering", "frames": 4, "fps": 10, "loop": False},
    "death": {"description": "collapsing to the ground, dramatic fall", "frames": 6, "fps": 8, "loop": False},
}

# Quality thresholds per stage
QUALITY_THRESHOLD = {
    "character": 0.7,
    "edit": 0.7,
}

MAX_RETRIES = 3


def log(msg: str):
    print(f"[auto] {msg}", flush=True)


# ---------------------------------------------------------------------------
# NPC Definition
# ---------------------------------------------------------------------------

def load_npc_defs(path: str | Path) -> list[dict]:
    """Load NPC definitions from YAML or JSON file."""
    path = Path(path)
    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(content)
        except ImportError:
            log("ERROR: PyYAML required for YAML files. Install: uv add pyyaml")
            sys.exit(1)
    else:
        data = json.loads(content)

    # Normalize to list
    if isinstance(data, dict):
        if "npcs" in data:
            data = data["npcs"]
        else:
            data = [data]

    # Validate
    for npc in data:
        if "name" not in npc:
            log(f"ERROR: NPC missing 'name' field: {npc}")
            sys.exit(1)
        npc.setdefault("description", "")
        npc.setdefault("animations", list(ANIMATION_DEFS.keys()))
        npc.setdefault("directions", ["front"])
        npc.setdefault("max_retries", MAX_RETRIES)

    return data


# ---------------------------------------------------------------------------
# Vision Evaluation
# ---------------------------------------------------------------------------

def evaluate_character(img_bytes: bytes, description: str) -> dict:
    """Evaluate character portrait quality using vision model."""
    from utils.api_client import call_llm_vision

    img = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img) / 255.0

    prompt = f"""Evaluate this AI-generated character portrait.

Character description: "{description}"

Rate the image on these criteria (0-1 scale):
1. anatomical_coherence: Are proportions correct? No extra/missing limbs?
2. face_quality: Clear, well-defined face with eyes/mouth/nose?
3. full_body_visible: Is the entire body visible head to toe?
4. style_consistency: Consistent art style throughout?
5. background_clean: Simple/clean background suitable for background removal?

Also note any specific issues (extra fingers, floating elements, etc).

Respond in JSON:
{{
  "scores": {{
    "anatomical_coherence": 0.0-1.0,
    "face_quality": 0.0-1.0,
    "full_body_visible": 0.0-1.0,
    "style_consistency": 0.0-1.0,
    "background_clean": 0.0-1.0
  }},
  "overall": 0.0-1.0,
  "issues": ["list of specific problems"],
  "pass": true/false,
  "adjustments": "suggested prompt changes if score is low"
}}"""

    try:
        from utils.api_client import resolve_api_key
        api_key = resolve_api_key(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="models/gemini-2.5-flash",
        )
        response = call_llm_vision(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
            model="models/gemini-2.5-flash",
            system_prompt="You evaluate AI-generated character portraits for game asset quality.",
            text_prompt=prompt,
            images=[img_array],
            temperature=0.3,
        )
        from utils.json_parser import parse_llm_json
        result = parse_llm_json(response)
        if isinstance(result, list):
            result = result[0] if result else {}
        return result
    except Exception as e:
        log(f"  Vision eval failed: {e}")
        return {"overall": 0.5, "pass": True, "issues": [f"eval error: {e}"]}


def evaluate_edit(img_bytes: bytes, instruction: str) -> dict:
    """Evaluate VNCCS edit quality."""
    from utils.api_client import call_llm_vision

    img = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img) / 255.0

    prompt = f"""Evaluate this AI-edited character sprite frame.

The instruction was: "{instruction}"

Rate the image:
1. character_identity: Does the character look consistent (not distorted)?
2. pose_accuracy: Does the pose look natural and physically plausible?
3. artifact_free: No extra limbs, floating elements, warping, or blurring?
4. overall_quality: Would this be usable as a game sprite frame?

Respond in JSON:
{{
  "scores": {{
    "character_identity": 0.0-1.0,
    "pose_accuracy": 0.0-1.0,
    "artifact_free": 0.0-1.0,
    "overall_quality": 0.0-1.0
  }},
  "overall": 0.0-1.0,
  "issues": ["list of problems"],
  "pass": true/false
}}"""

    try:
        from utils.api_client import resolve_api_key
        api_key = resolve_api_key(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="models/gemini-2.5-flash",
        )
        response = call_llm_vision(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
            model="models/gemini-2.5-flash",
            system_prompt="You evaluate AI-edited character sprite frames for game asset quality.",
            text_prompt=prompt,
            images=[img_array],
            temperature=0.3,
        )
        from utils.json_parser import parse_llm_json
        result = parse_llm_json(response)
        if isinstance(result, list):
            result = result[0] if result else {}
        return result
    except Exception as e:
        log(f"  Vision eval failed: {e}")
        return {"overall": 0.5, "pass": True, "issues": [f"eval error: {e}"]}


# ---------------------------------------------------------------------------
# Pipeline Stages (with retry)
# ---------------------------------------------------------------------------

def stage_character_with_retry(
    description: str,
    run_dir: Path,
    max_retries: int = MAX_RETRIES,
    existing_character: bytes | None = None,
) -> bytes:
    """Generate character portrait with vision evaluation and retry."""
    import numpy as np

    if existing_character:
        log("  Using existing character portrait")
        return existing_character

    for attempt in range(max_retries):
        seed = random.randint(0, 999999)
        log(f"  Attempt {attempt + 1}/{max_retries}, seed={seed}")

        workflow = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")

        # Fix missing VAEDecode→SaveImage connection
        save_node = next((n for n in workflow["nodes"] if n["type"] == "SaveImage"), None)
        vaedecode_node = next((n for n in workflow["nodes"] if n["type"] == "VAEDecode"), None)
        if save_node and vaedecode_node and save_node.get("inputs", [{}])[0].get("link") is None:
            from pipeline.ui_to_api import patch_connection
            patch_connection(
                workflow,
                target_node_id=save_node["id"],
                target_input_name="images",
                source_node_id=vaedecode_node["id"],
                source_output_index=0,
            )

        patch_ui_workflow(workflow, {
            "KSampler": {"seed": seed},
            "SaveImage": {"filename_prefix": f"auto/{run_dir.name}/character"},
        })

        api = convert_ui_to_api(workflow)
        result = submit_workflow(api)

        save_node_id = str(save_node["id"])
        images = get_output_images(result["outputs"], save_node_id)
        if not images:
            log("  No character image generated, retrying...")
            continue

        img_bytes = images[0]

        # Vision evaluation
        eval_result = evaluate_character(img_bytes, description)
        overall = eval_result.get("overall", 0)
        passed = eval_result.get("pass", overall >= QUALITY_THRESHOLD["character"])
        issues = eval_result.get("issues", [])

        log(f"  Quality: {overall:.2f} ({'PASS' if passed else 'FAIL'})")
        if issues:
            for issue in issues[:3]:
                log(f"    - {issue}")

        if passed:
            # Save
            char_dir = run_dir / "character"
            char_dir.mkdir(exist_ok=True)
            (char_dir / "character.png").write_bytes(img_bytes)
            upload_image(img_bytes, "character.png", "PoseDirector")
            return img_bytes

        log(f"  Below threshold ({overall:.2f} < {QUALITY_THRESHOLD['character']}), retrying...")

    log(f"  WARNING: Character quality below threshold after {max_retries} attempts, using best result")
    # Use last result anyway
    char_dir = run_dir / "character"
    char_dir.mkdir(exist_ok=True)
    (char_dir / "character.png").write_bytes(img_bytes)
    upload_image(img_bytes, "character.png", "PoseDirector")
    return img_bytes


def stage_sprites_for_npc(
    character_image: str,
    npc: dict,
    run_dir: Path,
    seed_base: int | None = None,
    evaluate: bool = True,
) -> dict:
    """Generate sprite frames for one NPC's animation set.

    Returns manifest dict with frame info.
    """
    if seed_base is None:
        seed_base = random.randint(0, 999999)

    animations = npc.get("animations", list(ANIMATION_DEFS.keys()))
    directions = npc.get("directions", ["front"])
    npc_name = npc["name"]

    all_results = []

    for anim_name in animations:
        anim_def = ANIMATION_DEFS.get(anim_name)
        if not anim_def:
            log(f"  Unknown animation '{anim_name}', skipping")
            continue

        # Load poses from PoseLibrary
        pose_library = Path("/home/ubuntu/Documents/programs/img/comfyui/custom_nodes/vnccs-utils/PoseLibrary")
        pose_files = sorted(pose_library.glob("*.json"))

        # Use first available poses for now (HY-Motion integration would go here)
        # Pick poses that match the animation count
        if len(pose_files) >= anim_def["frames"]:
            selected_poses = pose_files[:anim_def["frames"]]
        else:
            selected_poses = pose_files[:max(1, len(pose_files))]

        VNCCS_BONE_NAMES = {
            "upperarm_r", "lowerarm_r", "hand_r",
            "upperarm_l", "lowerarm_l", "hand_l",
            "thigh_r", "calf_r", "foot_r",
            "thigh_l", "calf_l", "foot_l",
            "spine_01", "spine_02", "spine_03",
            "neck_01", "head", "pelvis",
        }

        VNCCS_PROMPT = "Draw character from image2"
        VNCCS_INSTRUCTION = (
            "Match the body pose shown in Picture 1 (3D body mesh). "
            "Picture 2 is the character to draw. Picture 3 shows the skeleton overlay. "
            "Replicate the exact pose, limb positions, and body orientation from Picture 1 "
            "while maintaining the character's identity, clothing, and appearance."
        )

        dir_map = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}

        for direction in directions:
            for pf_idx, pose_path in enumerate(selected_poses):
                frame_label = f"{npc_name}/{anim_name}/{direction}/{pose_path.stem}"
                log(f"  [{len(all_results)+1}] {frame_label}")

                # Load pose
                with open(pose_path) as f:
                    pose = json.load(f)
                bones = pose.get("bones", {})
                rotations_json = json.dumps({k: v for k, v in bones.items() if k in VNCCS_BONE_NAMES})

                model_rotation_y = dir_map.get(direction, 0.0)
                frame_seed = seed_base + len(all_results)

                # Run through sprite pipeline
                try:
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
                        "KSampler": {"seed": frame_seed},
                        "SaveImage": {"filename_prefix": f"auto/{run_dir.name}/sprite"},
                    })

                    api = convert_ui_to_api(workflow)
                    result = submit_workflow(api)

                    save_node = next(n for n in workflow["nodes"] if n["type"] == "SaveImage")
                    images = get_output_images(result["outputs"], str(save_node["id"]))

                    if not images:
                        log(f"    No output")
                        continue

                    img_bytes = images[0]

                    # Optional evaluation
                    if evaluate and len(all_results) == 0:
                        eval_result = evaluate_edit(img_bytes, VNCCS_INSTRUCTION)
                        overall = eval_result.get("overall", 0)
                        log(f"    Edit quality: {overall:.2f}")
                        if overall < QUALITY_THRESHOLD["edit"]:
                            log(f"    Low quality, but continuing batch")

                except Exception as e:
                    log(f"    ERROR: {e}")
                    continue

                # Save frame
                anim_dir = run_dir / npc_name / anim_name / direction
                anim_dir.mkdir(parents=True, exist_ok=True)
                out_path = anim_dir / f"frame_{pf_idx:03d}.png"
                out_path.write_bytes(img_bytes)

                img = Image.open(io.BytesIO(img_bytes))
                all_results.append({
                    "npc": npc_name,
                    "animation": anim_name,
                    "direction": direction,
                    "frame": pf_idx,
                    "pose": pose_path.stem,
                    "path": str(out_path),
                    "size": list(img.size),
                })
                log(f"    OK: {img.size[0]}x{img.size[1]}")

    return {
        "npc": npc_name,
        "total_frames": len(all_results),
        "animations": animations,
        "directions": directions,
        "frames": all_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_autonomous_batch(
    npc_defs: list[dict],
    output_dir: str | Path | None = None,
    skip_character: bool = False,
    character_dir: str | Path | None = None,
    evaluate: bool = True,
):
    """Run autonomous batch pipeline for multiple NPCs."""
    import numpy as np

    # Setup output
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "output" / f"auto_{ts}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify ComfyUI
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        assert r.status_code == 200
    except Exception as e:
        log(f"ERROR: ComfyUI not reachable at {COMFYUI_URL}: {e}")
        sys.exit(1)
    log(f"ComfyUI ready at {COMFYUI_URL}")

    start_total = time.time()
    results = []

    for npc_idx, npc in enumerate(npc_defs):
        npc_name = npc["name"]
        npc_dir = output_dir / npc_name
        npc_dir.mkdir(exist_ok=True)

        log(f"\n{'='*60}")
        log(f"NPC {npc_idx+1}/{len(npc_defs)}: {npc_name}")
        log(f"  Description: {npc.get('description', '(none)')}")
        log(f"  Animations: {npc.get('animations', list(ANIMATION_DEFS.keys()))}")
        log(f"  Directions: {npc.get('directions', ['front'])}")
        log(f"{'='*60}")

        # Stage 1: Character
        existing_character = None
        if skip_character and character_dir:
            char_path = Path(character_dir) / npc_name / "character.png"
            if char_path.exists():
                existing_character = char_path.read_bytes()

        if skip_character and not existing_character:
            # Try to find in output dir
            char_path = npc_dir / "character" / "character.png"
            if char_path.exists():
                existing_character = char_path.read_bytes()

        t0 = time.time()
        character_bytes = stage_character_with_retry(
            description=npc.get("description", ""),
            run_dir=npc_dir,
            max_retries=npc.get("max_retries", MAX_RETRIES),
            existing_character=existing_character,
        )
        log(f"  Character: {time.time() - t0:.1f}s")

        # Stage 2: Sprites
        t0 = time.time()
        character_image = "PoseDirector/character.png"
        manifest = stage_sprites_for_npc(
            character_image=character_image,
            npc=npc,
            run_dir=npc_dir,
            evaluate=evaluate,
        )
        log(f"  Sprites: {time.time() - t0:.1f}s ({manifest['total_frames']} frames)")

        # Save NPC manifest
        manifest_path = npc_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        results.append(manifest)

    # Save global manifest
    total_frames = sum(r["total_frames"] for r in results)
    total_elapsed = time.time() - start_total
    global_manifest = {
        "npcs": len(npc_defs),
        "total_frames": total_frames,
        "total_time_s": round(total_elapsed, 1),
        "npc_details": results,
    }
    (output_dir / "batch_manifest.json").write_text(json.dumps(global_manifest, indent=2))

    log(f"\n{'='*60}")
    log(f"BATCH COMPLETE")
    log(f"  NPCs: {len(npc_defs)}")
    log(f"  Total frames: {total_frames}")
    log(f"  Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    log(f"  Output: {output_dir}")
    log(f"  Manifest: {output_dir / 'batch_manifest.json'}")
    log(f"{'='*60}")

    return output_dir


def main():
    import argparse
    import numpy as np  # noqa: ensure available

    parser = argparse.ArgumentParser(description="Autonomous batch NPC sprite generation")
    parser.add_argument("--npcs", default=None, help="YAML/JSON file with NPC definitions")
    parser.add_argument("--name", default=None, help="Single NPC name (inline mode)")
    parser.add_argument("--description", default="", help="Single NPC description (inline mode)")
    parser.add_argument("--animations", nargs="+", default=None,
                       help=f"Animations to generate: {list(ANIMATION_DEFS.keys())}")
    parser.add_argument("--directions", nargs="+", default=["front"],
                       help="Directions: front right back left")
    parser.add_argument("--skip-character", action="store_true",
                       help="Skip character generation, use existing portraits")
    parser.add_argument("--character-dir", default=None,
                       help="Directory with existing character portraits")
    parser.add_argument("--no-evaluate", action="store_true",
                       help="Skip vision evaluation (faster but no quality checks)")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    args = parser.parse_args()

    if args.npcs:
        npc_defs = load_npc_defs(args.npcs)
    elif args.name:
        npc_defs = [{
            "name": args.name,
            "description": args.description,
            "animations": args.animations or list(ANIMATION_DEFS.keys()),
            "directions": args.directions,
        }]
    else:
        parser.error("Specify --npcs <file> or --name <name> --description <desc>")

    run_autonomous_batch(
        npc_defs=npc_defs,
        output_dir=args.output,
        skip_character=args.skip_character,
        character_dir=args.character_dir,
        evaluate=not args.no_evaluate,
    )


if __name__ == "__main__":
    main()
