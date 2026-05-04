#!/usr/bin/env python3
"""Declarative character asset build system.

Reads a character YAML manifest, checks build state, and generates only what's missing.
Supports composable packages (generation, portraits, sprites, outfit, state) with
selective rebuild and incremental addition.

Usage:
    # Build a character (generates only what's missing)
    uv run python scripts/build_character.py characters/marty.yaml

    # Force rebuild a specific package
    uv run python scripts/build_character.py characters/marty.yaml --rebuild sprites.casual.normal.idle
    uv run python scripts/build_character.py characters/marty.yaml --rebuild portraits
    uv run python scripts/build_character.py characters/marty.yaml --rebuild trellis

    # Override seed
    uv run python scripts/build_character.py characters/marty.yaml --seed 99

    # Dry run (show what would be generated)
    uv run python scripts/build_character.py characters/marty.yaml --dry-run

    # Nuclear option — delete all outputs and rebuild
    uv run python scripts/build_character.py characters/marty.yaml --clean

    # Build multiple characters
    uv run python scripts/build_character.py characters/*.yaml
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Run: uv pip install pyyaml")
    sys.exit(1)

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
    save_api_workflow,
)
import requests

WORKFLOWS_DIR = PROJECT_ROOT / "workflows"
OUTPUT_DIR = PROJECT_ROOT / "output"
MOTIONS_DIR = PROJECT_ROOT / "motions"

VALID_EMOTIONS = {
    "neutral", "happy", "sad", "angry", "confused", "determined", "disgusted",
    "excited", "exhausted", "nervous", "pout", "scared", "serious", "surprised",
    "thinking", "grin", "evil grin", "light smile", "sad smile", "forced smile",
    "doyagao", "smirk", "smug", "troll face", "horrified", "screaming", "sobbing",
    "traumatized", "despair", "suspicious",
}

VALID_DIRECTIONS = {"front", "back", "left", "right"}
VALID_MOTION_SOURCES = {"shared", "custom", "manual", "file"}
VALID_TIERS = {"t0", "t1", "t1.5", "t2", "t3"}
VALID_QUALITIES = {"turbo", "base"}

VNCCS_PROMPT = "Draw character from image2"
VNCCS_INSTRUCTION = (
    "Match the body pose shown in Picture 1 (3D body mesh). "
    "Picture 2 is the character to draw. Picture 3 shows the skeleton overlay. "
    "Replicate the exact pose, limb positions, and body orientation from Picture 1 "
    "while maintaining the character's identity, clothing, and appearance."
)

DIRECTION_ROTATIONS = {
    "front": 0.0,
    "right": 90.0,
    "back": 180.0,
    "left": 270.0,
}

TIER_DEFAULTS = {
    "t0": {
        "directions": ["front"],
        "body_source": "import",
        "motion_source": "shared",
    },
    "t1": {
        "directions": ["front"],
        "body_source": "anny",
        "motion_source": "shared",
    },
    "t1.5": {
        "directions": ["front"],
        "body_source": "anny",
        "motion_source": "shared",
    },
    "t2": {
        "directions": ["front", "back", "left", "right"],
        "body_source": "trellis",
        "motion_source": "custom",
    },
    "t3": {
        "directions": ["front", "back", "left", "right"],
        "body_source": "trellis",
        "motion_source": "custom",
    },
}


def log(msg: str):
    print(f"[build] {msg}", flush=True)


def die(msg: str):
    log(f"ERROR: {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# YAML Parsing & Validation
# ---------------------------------------------------------------------------

def load_character_yaml(path: Path) -> dict:
    """Load and validate a character YAML manifest."""
    if not path.exists():
        die(f"Character YAML not found: {path}")
    path = path.resolve()
    with open(path) as f:
        data = yaml.safe_load(f)
    data["_yaml_path"] = str(path)  # For resolving relative import paths
    validate_character(data, path)
    return data


def validate_character(data: dict, path: Path):
    """Validate character YAML structure."""
    char = data.get("character", {})
    if not char.get("name"):
        die(f"{path}: character.name is required")
    name = char["name"]
    if not name.replace("_", "").replace("-", "").isalnum():
        die(f"{path}: character.name must be filesystem-safe (alphanumeric, dash, underscore)")

    gen = data.get("generation", {})
    source = gen.get("source", "generate")
    if source not in ("generate", "import"):
        die(f"{path}: generation.source must be 'generate' or 'import'")
    if source == "generate" and not gen.get("prompt"):
        die(f"{path}: generation.prompt is required when source=generate")
    if source == "import":
        img = gen.get("image")
        if img and not (path.parent / img).exists():
            log(f"WARNING: {path}: import image not found: {img}")

    quality = gen.get("quality", "turbo")
    if quality not in VALID_QUALITIES:
        die(f"{path}: generation.quality must be one of {VALID_QUALITIES}")

    tier = char.get("tier", "t1")
    if tier not in VALID_TIERS:
        die(f"{path}: character.tier must be one of {VALID_TIERS}")

    packages = data.get("packages", {})
    for pkg_name in ("portraits", "sprites", "outfit", "state"):
        if pkg_name in packages and packages[pkg_name] is None:
            die(f"{path}: packages.{pkg_name} is declared but empty")

    portraits = packages.get("portraits", {})
    if portraits:
        emotions = portraits.get("emotions", [])
        unknown = set(emotions) - VALID_EMOTIONS
        if unknown:
            log(f"WARNING: {path}: unknown emotions (will try anyway): {unknown}")

    sprites = packages.get("sprites", {})
    if sprites:
        directions = sprites.get("directions", TIER_DEFAULTS[tier]["directions"])
        invalid = set(directions) - VALID_DIRECTIONS
        if invalid:
            die(f"{path}: invalid directions: {invalid}")

        # Support both "animations" (legacy) and "animated" (new tier-based schema)
        animations = sprites.get("animations") or sprites.get("animated") or []
        for anim in animations:
            if anim.get("motion") and anim["motion"] not in VALID_MOTION_SOURCES:
                die(f"{path}: invalid motion source '{anim['motion']}' for animation '{anim.get('name')}'")
            if anim.get("motion") == "file" and not anim.get("path"):
                die(f"{path}: motion=file requires a 'path' for animation '{anim.get('name')}'")
            if anim.get("motion") == "manual" and not anim.get("frames"):
                die(f"{path}: motion=manual requires 'frames' for animation '{anim.get('name')}'")


# ---------------------------------------------------------------------------
# Build State Management
# ---------------------------------------------------------------------------

def compute_yaml_hash(data: dict) -> str:
    """Hash the YAML content for change detection."""
    content = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def load_build_state(output_dir: Path) -> dict:
    """Load build state from character output directory."""
    state_file = output_dir / "build_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"character": "", "yaml_hash": "", "last_build": ""}


def save_build_state(output_dir: Path, state: dict):
    """Save build state."""
    state["last_build"] = datetime.now().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "build_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def file_hash(path: Path) -> str:
    """SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Build Plan
# ---------------------------------------------------------------------------

class BuildPlan:
    """Tracks what needs to be generated."""

    def __init__(self):
        self.steps: list[dict] = []

    def add(self, stage: str, action: str, details: dict, reason: str = "missing"):
        self.steps.append({
            "stage": stage,
            "action": action,
            "details": details,
            "reason": reason,
        })

    def __len__(self):
        return len(self.steps)

    def __bool__(self):
        return len(self.steps) > 0

    def summary(self) -> str:
        stages = {}
        for step in self.steps:
            s = step["stage"]
            stages[s] = stages.get(s, 0) + 1
        parts = [f"{count} {stage}" for stage, count in stages.items()]
        return ", ".join(parts)


def make_plan(
    data: dict,
    state: dict,
    rebuild_targets: list[str] | None = None,
) -> BuildPlan:
    """Determine what needs building based on YAML, state, and rebuild flags."""
    plan = BuildPlan()
    char = data["character"]
    name = char["name"]
    tier = char.get("tier", "t1")
    gen = data.get("generation", {})
    packages = data.get("packages", {})
    new_hash = compute_yaml_hash(data)

    # Determine rebuild scope
    rebuild_all = rebuild_targets and "all" in rebuild_targets
    rebuild_generation = rebuild_all or (rebuild_targets and "generation" in rebuild_targets)
    rebuild_trellis = rebuild_all or (rebuild_targets and "trellis" in rebuild_targets)
    rebuild_portraits = rebuild_all or (rebuild_targets and "portraits" in rebuild_targets)
    rebuild_sprites_prefixes = []
    if rebuild_targets:
        for t in rebuild_targets:
            if t.startswith("sprites"):
                rebuild_sprites_prefixes.append(t)

    # --- Stage 1: Generation ---
    gen_state = state.get("generation", {})
    source = gen.get("source", "generate")

    needs_gen = rebuild_generation
    if not needs_gen:
        if gen_state.get("status") != "complete":
            needs_gen = True
        elif new_hash != state.get("yaml_hash") and source == "generate":
            # Prompt changed — need to regenerate
            old_prompt = gen_state.get("prompt", "")
            new_prompt = gen.get("prompt", "")
            if old_prompt != new_prompt:
                needs_gen = True
        elif source == "import":
            img_path = gen.get("image")
            if img_path:
                old_hash = gen_state.get("input_hash", "")
                try:
                    new_hash_val = file_hash(Path(img_path))
                    if old_hash != new_hash_val:
                        needs_gen = True
                except FileNotFoundError:
                    pass

    if needs_gen:
        plan.add("generation", source, {
            "source": source,
            "prompt": gen.get("prompt", ""),
            "quality": gen.get("quality", "turbo"),
            "seed": gen.get("seed", 42),
            "image": gen.get("image"),
        })

    # --- Stage 2: TRELLIS (t2+ only) ---
    trellis_state = state.get("trellis", {})
    needs_trellis = rebuild_trellis and tier in ("t2", "t3")
    if not needs_trellis and tier in ("t2", "t3"):
        if trellis_state.get("status") != "complete":
            needs_trellis = True

    if needs_trellis:
        plan.add("trellis", "generate", {
            "tier": tier,
        })

    # --- Stage 3: Portraits ---
    portraits_pkg = packages.get("portraits")
    if portraits_pkg:
        emotions = portraits_pkg.get("emotions", [])
        portraits_state = state.get("portraits", {})

        for emotion in emotions:
            outfit = "default"
            # Check if this emotion needs generating
            emotion_key = f"{outfit}/{emotion}"
            emo_state = portraits_state.get(emotion_key, {})

            needs = rebuild_portraits
            if not needs:
                if emo_state.get("status") != "complete":
                    needs = True
                # Check if file exists
                output_path = OUTPUT_DIR / name / "portraits" / outfit / f"{emotion}.png"
                if not output_path.exists():
                    needs = True

            if needs:
                plan.add("portraits", "generate", {
                    "outfit": outfit,
                    "emotion": emotion,
                })

    # --- Stage 4: Sprites ---
    sprites_pkg = packages.get("sprites")
    if sprites_pkg:
        outfit = sprites_pkg.get("outfit", "default")
        directions = sprites_pkg.get("directions", TIER_DEFAULTS[tier]["directions"])
        body_source = TIER_DEFAULTS[tier]["body_source"]

        for anim in sprites_pkg.get("animations") or sprites_pkg.get("animated") or []:
            anim_name = anim["name"]
            motion_source = anim.get("motion", TIER_DEFAULTS[tier]["motion_source"])
            sprite_key = f"{outfit}/normal/{anim_name}"
            anim_state = state.get("sprites", {}).get(sprite_key, {})

            # Check if rebuild flag matches
            # Normalize: sprites.casual.normal.idle -> sprites/casual/normal/idle
            sprite_path_str = f"sprites/{sprite_key}"
            needs = rebuild_all
            if not needs and rebuild_sprites_prefixes:
                for prefix in rebuild_sprites_prefixes:
                    normalized_prefix = prefix.replace(".", "/")
                    if sprite_path_str.startswith(normalized_prefix):
                        needs = True
                        break

            if not needs:
                if anim_state.get("status") == "complete":
                    # Trust build state — don't recheck filesystem
                    # (filesystem check only used when state is missing)
                    needs = False
                else:
                    needs = True
                # Check motion prompt change
                if not needs and motion_source == "custom":
                    old_prompt = anim_state.get("motion_prompt", "")
                    new_prompt = anim.get("prompt", "")
                    if old_prompt and old_prompt != new_prompt:
                        needs = True
                # Check filesystem only when state has no entry
                if not needs and not anim_state:
                    for d in directions:
                        frame_dir = OUTPUT_DIR / name / "sprites" / outfit / "normal" / anim_name / d
                        if not frame_dir.exists() or not list(frame_dir.glob("frame_*.png")):
                            needs = True
                            break

            if needs:
                plan.add("sprites", "generate", {
                    "outfit": outfit,
                    "state": "normal",
                    "animation": anim_name,
                    "directions": directions,
                    "motion_source": motion_source,
                    "motion_prompt": anim.get("prompt", ""),
                    "motion_path": anim.get("path"),
                    "motion_frames": anim.get("frames"),
                    "body_source": body_source,
                })

    # --- Stage 5: Outfit variants ---
    outfit_pkg = packages.get("outfit")
    if outfit_pkg and sprites_pkg:
        outfit_name = outfit_pkg["name"]
        outfit_desc = outfit_pkg.get("description", "")
        directions = sprites_pkg.get("directions", TIER_DEFAULTS[tier]["directions"])

        for anim in sprites_pkg.get("animations") or sprites_pkg.get("animated") or []:
            anim_name = anim["name"]
            sprite_key = f"{outfit_name}/normal/{anim_name}"
            anim_state = state.get("sprites", {}).get(sprite_key, {})

            needs = rebuild_all
            if not needs and rebuild_sprites_prefixes:
                for prefix in rebuild_sprites_prefixes:
                    if f"sprites/{sprite_key}".startswith(prefix.replace(".", "/")):
                        needs = True
                        break

            if not needs:
                if anim_state.get("status") != "complete":
                    needs = True
                for d in directions:
                    frame_dir = OUTPUT_DIR / name / "sprites" / outfit_name / "normal" / anim_name / d
                    if not frame_dir.exists() or not list(frame_dir.glob("frame_*.png")):
                        needs = True
                        break

            if needs:
                plan.add("outfit", "generate", {
                    "outfit": outfit_name,
                    "description": outfit_desc,
                    "animation": anim_name,
                    "directions": directions,
                    "motion_source": anim.get("motion", "shared"),
                    "motion_prompt": anim.get("prompt", ""),
                    "body_source": TIER_DEFAULTS[tier]["body_source"],
                })

    # --- Stage 6: State variants ---
    state_pkg = packages.get("state")
    if state_pkg and sprites_pkg:
        state_name = state_pkg["name"]
        state_desc = state_pkg.get("description", "")
        apply_to = state_pkg.get("apply_to", [])

        for target in apply_to:
            if isinstance(target, str) and target == "portraits":
                # Re-edit all portraits with state
                portraits_pkg = packages.get("portraits", {})
                for emotion in portraits_pkg.get("emotions", []):
                    outfit = "default"
                    portrait_key = f"{outfit}_{state_name}/{emotion}"
                    portrait_state = state.get("portraits", {}).get(portrait_key, {})

                    needs = rebuild_all
                    if not needs:
                        if portrait_state.get("status") != "complete":
                            needs = True
                        output_path = OUTPUT_DIR / name / "portraits" / f"{outfit}_{state_name}" / f"{emotion}.png"
                        if not output_path.exists():
                            needs = True

                    if needs:
                        plan.add("state", "portrait", {
                            "state": state_name,
                            "description": state_desc,
                            "outfit": outfit,
                            "emotion": emotion,
                        })

            elif isinstance(target, dict) and "animations" in target:
                # State applied to specific animations
                outfit = sprites_pkg.get("outfit", "default")
                for anim_name in target["animations"]:
                    directions = sprites_pkg.get("directions", TIER_DEFAULTS[tier]["directions"])
                    sprite_key = f"{outfit}/{state_name}/{anim_name}"

                    needs = rebuild_all
                    if not needs:
                        for d in directions:
                            frame_dir = OUTPUT_DIR / name / "sprites" / outfit / state_name / anim_name / d
                            if not frame_dir.exists() or not list(frame_dir.glob("frame_*.png")):
                                needs = True
                                break

                    if needs:
                        plan.add("state", "sprite", {
                            "state": state_name,
                            "description": state_desc,
                            "outfit": outfit,
                            "animation": anim_name,
                            "directions": directions,
                        })

    return plan


# ---------------------------------------------------------------------------
# Build Execution
# ---------------------------------------------------------------------------

def check_comfyui():
    """Verify ComfyUI is reachable."""
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        assert r.status_code == 200
    except Exception as e:
        die(f"ComfyUI not reachable at {COMFYUI_URL}: {e}")
    log(f"ComfyUI ready at {COMFYUI_URL}")


def stage_generate(data: dict, details: dict, state: dict, seed_override: int | None) -> dict:
    """Generate character image via Z-Image."""
    source = details["source"]
    name = data["character"]["name"]
    char_dir = OUTPUT_DIR / name

    if source == "import":
        img_path = Path(details["image"])
        # Resolve relative to YAML file location, not CWD
        if not img_path.is_absolute():
            yaml_dir = Path(data.get("_yaml_path", ".")).parent
            img_path = yaml_dir / img_path
        if not img_path.exists():
            die(f"Import image not found: {img_path}")
        design_dir = char_dir / "design"
        design_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(img_path, design_dir / "approved.png")
        log(f"  Imported: {design_dir / 'approved.png'}")
        return {
            "status": "complete",
            "source": "import",
            "input_image": str(img_path),
            "input_hash": file_hash(img_path),
            "output": str(design_dir / "approved.png"),
        }

    # Generate via Z-Image
    quality = details.get("quality", "turbo")
    prompt = details["prompt"]
    seed = seed_override or details.get("seed", 42)

    log(f"  Generating character (quality={quality}, seed={seed})")
    workflow = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")

    patches = {
        "CLIPTextEncode": {"text": prompt},
        "KSampler": {"seed": seed},
        "SaveImage": {"filename_prefix": f"PoseDirector/{name}/character"},
    }
    if quality == "base":
        patches["UNETLoader"] = {"unet_name": "z_image_bf16.safetensors"}
        patches["KSampler"] = {"seed": seed, "steps": 50, "cfg": 4.0, "sampler_name": "euler", "scheduler": "normal"}

    patch_ui_workflow(workflow, patches)
    api = convert_ui_to_api(workflow)
    result = submit_workflow(api, timeout=600)

    # Get output image
    save_node = next(n for n in workflow["nodes"] if n["type"] == "SaveImage")
    images = get_output_images(result["outputs"], save_node["id"])

    if not images:
        die("Z-Image produced no output")

    # Save to design directory
    design_dir = char_dir / "design"
    design_dir.mkdir(parents=True, exist_ok=True)
    approved_path = design_dir / "approved.png"
    approved_path.write_bytes(images[0])
    log(f"  Saved: {approved_path}")

    return {
        "status": "complete",
        "source": "generate",
        "quality": quality,
        "prompt": prompt,
        "seed": seed,
        "output": str(approved_path),
    }


def stage_trellis(data: dict, state: dict, seed_override: int | None) -> dict:
    """Generate 3D model via TRELLIS and render reference views."""
    name = data["character"]["name"]
    char_image_path = OUTPUT_DIR / name / "design" / "approved.png"
    if not char_image_path.exists():
        die(f"Character image not found for TRELLIS: {char_image_path}")

    char_dir = OUTPUT_DIR / name
    glb_path = char_dir / "reference" / "trellis_model.glb"

    # Check if GLB already exists
    if glb_path.exists():
        log(f"  TRELLIS model already exists: {glb_path}")
        return {"status": "complete", "model_path": str(glb_path)}

    log(f"  Running TRELLIS pipeline...")
    from scripts.run_trellis_pipeline import run_trellis_pipeline

    try:
        trellis_dir = run_trellis_pipeline(
            image_path=char_image_path,
            num_angles=4,
            resolution=512,
            output_dir=char_dir / "reference",
            use_ray=False,  # Use subprocess fallback for local
        )

        # The GLB should be at output_dir/model.glb
        result_glb = char_dir / "reference" / "model.glb"
        if result_glb.exists():
            # Rename to expected path
            ref_dir = char_dir / "reference"
            ref_dir.mkdir(parents=True, exist_ok=True)
            target = ref_dir / "trellis_model.glb"
            import shutil
            shutil.move(str(result_glb), str(target))

            # Save reference renders
            renders = {}
            frames_dir = trellis_dir / "frames" if isinstance(trellis_dir, Path) else None
            if frames_dir and frames_dir.exists():
                for f in sorted(frames_dir.glob("*.png")):
                    angle = f.stem.replace("turn_", "")
                    renders[f"{int(angle):03d}deg"] = str(f)

            return {
                "status": "complete",
                "model_path": str(target),
                "reference_images": renders,
            }
        else:
            log(f"  WARNING: TRELLIS did not produce model.glb")
            return {"status": "failed"}
    except Exception as e:
        log(f"  TRELLIS failed: {e}")
        log(f"  Continuing without 3D model — using Anny body instead")
        return {"status": "failed", "error": str(e)}


def stage_portraits(data: dict, details: dict, state: dict, seed_override: int | None) -> dict:
    """Generate a single portrait with emotion via VNCCS.

    For now, uses the sprite pipeline with a neutral pose as the portrait generator.
    TODO: Create a dedicated emotion_portrait_workflow.json using EmotionGenerator.
    """
    name = data["character"]["name"]
    emotion = details["emotion"]
    outfit = details["outfit"]
    seed = seed_override or data.get("generation", {}).get("seed", 42)

    # Get the character's approved image
    char_image_path = OUTPUT_DIR / name / "design" / "approved.png"
    if not char_image_path.exists():
        die(f"Character image not found: {char_image_path}")

    # For now, use VNCCS edit with emotion instruction
    # Load character into ComfyUI
    char_bytes = char_image_path.read_bytes()
    upload_image(char_bytes, f"{name}_char.png", "PoseDirector")

    # Use a simple front-facing neutral pose
    neutral_pose = {
        "upperarm_r": [0, 0, 0], "lowerarm_r": [0, 0, 0], "hand_r": [0, 0, 0],
        "upperarm_l": [0, 0, 0], "lowerarm_l": [0, 0, 0], "hand_l": [0, 0, 0],
        "thigh_r": [0, 0, 0], "calf_r": [0, 0, 0], "foot_r": [0, 0, 0],
        "thigh_l": [0, 0, 0], "calf_l": [0, 0, 0], "foot_l": [0, 0, 0],
        "spine_01": [0, 0, 0], "spine_02": [0, 0, 0], "spine_03": [0, 0, 0],
        "neck_01": [0, 0, 0], "head": [0, 0, 0], "pelvis": [0, 0, 0],
    }

    log(f"  Generating portrait: {outfit}/{emotion}")

    workflow = load_ui_workflow(WORKFLOWS_DIR / "05_sprite_bodymesh_edit.json")
    patches = {
        "LoadImage": {"image": f"PoseDirector/{name}_char.png"},
        "BodyMeshRenderer": {
            "rotations_json": json.dumps(neutral_pose),
            "model_rotation_y": 0.0,
        },
        "VNCCS_QWEN_Encoder": {
            "prompt": f"Draw character from image2 with {emotion} expression",
            "instruction": f"Draw the character from Picture 2 with a {emotion} expression. "
                          f"Maintain the character's identity and appearance. "
                          f"The expression should clearly convey {emotion}.",
        },
        "KSampler": {"seed": seed},
        "SaveImage": {"filename_prefix": f"PoseDirector/{name}/portrait_{outfit}_{emotion}"},
    }
    patch_ui_workflow(workflow, patches)
    api = convert_ui_to_api(workflow)
    result = submit_workflow(api, timeout=300)

    save_node = next(n for n in workflow["nodes"] if n["type"] == "SaveImage")
    images = get_output_images(result["outputs"], save_node["id"])

    if not images:
        die(f"Portrait generation failed for {emotion}")

    # Save to portraits directory
    portrait_dir = OUTPUT_DIR / name / "portraits" / outfit
    portrait_dir.mkdir(parents=True, exist_ok=True)
    portrait_path = portrait_dir / f"{emotion}.png"
    portrait_path.write_bytes(images[0])
    log(f"  Saved: {portrait_path}")

    return {
        "status": "complete",
        "emotion": emotion,
        "seed": seed,
    }


def stage_sprites(data: dict, details: dict, state: dict, seed_override: int | None) -> dict:
    """Generate sprite frames for an animation."""
    from utils.hymotion_converter import extract_keyframes

    name = data["character"]["name"]
    outfit = details["outfit"]
    anim_name = details["animation"]
    directions = details["directions"]
    motion_source = details["motion_source"]
    motion_prompt = details.get("motion_prompt", "")
    body_source = details.get("body_source", "anny")
    seed = seed_override or data.get("generation", {}).get("seed", 42)

    char_image_path = OUTPUT_DIR / name / "design" / "approved.png"
    if not char_image_path.exists():
        die(f"Character image not found: {char_image_path}")

    # Upload character
    char_bytes = char_image_path.read_bytes()
    upload_image(char_bytes, f"{name}_char.png", "PoseDirector")

    # --- Resolve motion data ---
    npz_path = None
    if motion_source == "shared":
        npz_path = MOTIONS_DIR / "shared" / f"{anim_name}.npz"
        if not npz_path.exists():
            # Fallback to idle.npz for non-standard animation names
            fallback = MOTIONS_DIR / "shared" / "idle.npz"
            if fallback.exists():
                log(f"  No shared motion for '{anim_name}', falling back to idle.npz")
                npz_path = fallback
            else:
                die(f"Shared motion not found: {npz_path} (no idle fallback either)")
    elif motion_source == "custom":
        npz_path = MOTIONS_DIR / "custom" / f"{name}_{anim_name}.npz"
        if not npz_path.exists():
            # Generate via HY-Motion
            log(f"  Generating motion: {motion_prompt}")
            npz_path = run_hymotion(motion_prompt, name, anim_name, seed)
    elif motion_source == "file":
        npz_path = Path(details["motion_path"])
        if not npz_path.exists():
            die(f"Motion file not found: {npz_path}")
    elif motion_source == "manual":
        npz_path = build_manual_npz(details["motion_frames"], name, anim_name)

    if not npz_path or not npz_path.exists():
        die(f"No motion data for {anim_name}")

    # Extract keyframes
    num_keyframes = 6
    keyframes = extract_keyframes(npz_path, num_keyframes)
    log(f"  {len(keyframes)} keyframes from {npz_path.name}")

    # Generate frames
    total_frames = 0
    for kf_idx, rotations in enumerate(keyframes):
        for direction in directions:
            frame_seed = seed + kf_idx * len(directions) + directions.index(direction)
            rotations_json = json.dumps(rotations)
            model_rotation_y = DIRECTION_ROTATIONS.get(direction, 0.0)

            workflow = load_ui_workflow(WORKFLOWS_DIR / "05_sprite_bodymesh_edit.json")
            patches = {
                "LoadImage": {"image": f"PoseDirector/{name}_char.png"},
                "BodyMeshRenderer": {
                    "rotations_json": rotations_json,
                    "model_rotation_y": model_rotation_y,
                },
                "VNCCS_QWEN_Encoder": {
                    "prompt": VNCCS_PROMPT,
                    "instruction": VNCCS_INSTRUCTION,
                },
                "KSampler": {"seed": frame_seed},
                "SaveImage": {"filename_prefix": f"PoseDirector/{name}/sprite"},
            }
            patch_ui_workflow(workflow, patches)
            api = convert_ui_to_api(workflow)
            result = submit_workflow(api, timeout=300)

            save_node = next(n for n in workflow["nodes"] if n["type"] == "SaveImage")
            images = get_output_images(result["outputs"], save_node["id"])

            if images:
                frame_dir = OUTPUT_DIR / name / "sprites" / outfit / "normal" / anim_name / direction
                frame_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frame_dir / f"frame_{kf_idx + 1:03d}.png"
                frame_path.write_bytes(images[0])
                total_frames += 1

    log(f"  Generated {total_frames} frames for {anim_name}")

    return {
        "status": "complete",
        "outfit": outfit,
        "state": "normal",
        "animation": anim_name,
        "directions": directions,
        "frames_per_dir": len(keyframes),
        "total_frames": total_frames,
        "motion_source": motion_source,
        "motion_prompt": motion_prompt,
        "npz_path": str(npz_path),
        "seed": seed,
    }


def run_hymotion(prompt: str, char_name: str, anim_name: str, seed: int) -> Path:
    """Run HY-Motion via ComfyUI workflow to generate motion NPZ."""
    npz_target = MOTIONS_DIR / "custom" / f"{char_name}_{anim_name}.npz"
    if npz_target.exists():
        log(f"  Using cached NPZ: {npz_target}")
        return npz_target

    npz_target.parent.mkdir(parents=True, exist_ok=True)

    workflow = load_ui_workflow(WORKFLOWS_DIR / "01_hymotion_keyframes.json")
    patches = {
        "HYMotionEncodeText": {"text": prompt},
        "HYMotionGenerate": {"seed": seed},
        "HYMotionSaveNPZ": {
            "output_dir": f"motions_custom",
            "filename_prefix": f"{char_name}_{anim_name}",
        },
    }
    patch_ui_workflow(workflow, patches)
    api = convert_ui_to_api(workflow)
    result = submit_workflow(api, timeout=600)

    # The NPZ is saved inside ComfyUI's output directory
    # Find it via the history output
    from pipeline.comfyui_client import COMFYUI_URL as CU_URL
    import requests as _req

    # Check ComfyUI output dir for the NPZ
    comfyui_output = Path("/home/ubuntu/Documents/programs/img/comfyui/output")
    npz_files = sorted(comfyui_output.rglob(f"{char_name}_{anim_name}*.npz"), key=lambda p: p.stat().st_mtime)

    if not npz_files:
        # Also check the output_dir specified in the node
        comfyui_output2 = comfyui_output / "motions_custom"
        npz_files = sorted(comfyui_output2.glob(f"{char_name}_{anim_name}*.npz"), key=lambda p: p.stat().st_mtime)

    if not npz_files:
        die(f"HY-Motion did not produce NPZ output for {char_name}_{anim_name}")

    # Copy to our motions dir
    import shutil
    shutil.copy2(npz_files[-1], npz_target)
    log(f"  Cached NPZ: {npz_target}")
    return npz_target


def build_manual_npz(frames: list, char_name: str, anim_name: str) -> Path:
    """Build an NPZ file from manual joint rotation data."""
    import numpy as np

    npz_path = MOTIONS_DIR / "custom" / f"{char_name}_{anim_name}_manual.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Store as motion data
    np.savez(npz_path, frames=frames)
    log(f"  Built manual NPZ: {npz_path}")
    return npz_path


def stage_outfit(data: dict, details: dict, state: dict, seed_override: int | None) -> dict:
    """Generate outfit variant sprites.

    First generates an outfit reference image via VNCCS edit, then re-runs sprite generation
    using the outfit reference instead of the base character image.
    """
    name = data["character"]["name"]
    outfit_name = details["outfit"]
    outfit_desc = details["description"]
    anim_name = details["animation"]
    directions = details["directions"]

    log(f"  Outfit variant: {outfit_name} — {anim_name}")

    seed = seed_override or data.get("generation", {}).get("seed", 42)

    # Generate or reuse outfit reference image
    char_image_path = OUTPUT_DIR / name / "design" / "approved.png"
    if not char_image_path.exists():
        raise RuntimeError(f"Character image not found: {char_image_path}")

    outfit_dir = OUTPUT_DIR / name / "design"
    outfit_ref_path = outfit_dir / f"{outfit_name}.png"

    if not outfit_ref_path.exists():
        # Generate outfit reference via VNCCS edit
        char_bytes = char_image_path.read_bytes()
        upload_image(char_bytes, f"{name}_char.png", "PoseDirector")

        neutral_pose = {
            "upperarm_r": [0, 0, 0], "lowerarm_r": [0, 0, 0], "hand_r": [0, 0, 0],
            "upperarm_l": [0, 0, 0], "lowerarm_l": [0, 0, 0], "hand_l": [0, 0, 0],
            "thigh_r": [0, 0, 0], "calf_r": [0, 0, 0], "foot_r": [0, 0, 0],
            "thigh_l": [0, 0, 0], "calf_l": [0, 0, 0], "foot_l": [0, 0, 0],
            "spine_01": [0, 0, 0], "spine_02": [0, 0, 0], "spine_03": [0, 0, 0],
            "neck_01": [0, 0, 0], "head": [0, 0, 0], "pelvis": [0, 0, 0],
        }

        workflow = load_ui_workflow(WORKFLOWS_DIR / "05_sprite_bodymesh_edit.json")
        patches = {
            "LoadImage": {"image": f"PoseDirector/{name}_char.png"},
            "BodyMeshRenderer": {
                "rotations_json": json.dumps(neutral_pose),
                "model_rotation_y": 0.0,
            },
            "VNCCS_QWEN_Encoder": {
                "prompt": f"Draw character from image2 wearing {outfit_desc}",
                "instruction": f"Redraw the character from Picture 2 wearing {outfit_desc}. "
                              f"Maintain the character's facial features, body type, and identity.",
            },
            "KSampler": {"seed": seed},
            "SaveImage": {"filename_prefix": f"PoseDirector/{name}/outfit_{outfit_name}"},
        }
        patch_ui_workflow(workflow, patches)
        api = convert_ui_to_api(workflow)
        result = submit_workflow(api, timeout=300)

        save_node = next(n for n in workflow["nodes"] if n["type"] == "SaveImage")
        images = get_output_images(result["outputs"], save_node["id"])

        if not images:
            raise RuntimeError(f"Outfit reference generation failed for {outfit_name}")

        outfit_dir.mkdir(parents=True, exist_ok=True)
        outfit_ref_path.write_bytes(images[0])
        log(f"  Outfit reference: {outfit_ref_path}")

    # Upload outfit reference
    outfit_bytes = outfit_ref_path.read_bytes()
    upload_image(outfit_bytes, f"{name}_{outfit_name}.png", "PoseDirector")

    # Generate sprite frames using the outfit reference

    # Generate sprite frames using the outfit reference
    motion_source = details.get("motion_source", "shared")
    motion_prompt = details.get("motion_prompt", "")

    # Find the base animation's NPZ
    npz_path = MOTIONS_DIR / "custom" / f"{name}_{anim_name}.npz"
    if not npz_path.exists():
        npz_path = MOTIONS_DIR / "shared" / f"{anim_name}.npz"
    if not npz_path.exists():
        # Fallback to idle.npz
        fallback = MOTIONS_DIR / "shared" / "idle.npz"
        if fallback.exists():
            log(f"  No shared motion for '{anim_name}', falling back to idle.npz")
            npz_path = fallback
        else:
            log(f"  WARNING: No motion data for {anim_name}, skipping outfit sprites")
            return {"status": "pending", "outfit": outfit_name, "animation": anim_name}

    from utils.hymotion_converter import extract_keyframes
    keyframes = extract_keyframes(npz_path, 6)

    total_frames = 0
    for kf_idx, rotations in enumerate(keyframes):
        for direction in directions:
            frame_seed = seed + kf_idx * len(directions) + directions.index(direction)
            rotations_json = json.dumps(rotations)
            model_rotation_y = DIRECTION_ROTATIONS.get(direction, 0.0)

            workflow = load_ui_workflow(WORKFLOWS_DIR / "05_sprite_bodymesh_edit.json")
            patches = {
                "LoadImage": {"image": f"PoseDirector/{name}_{outfit_name}.png"},
                "BodyMeshRenderer": {
                    "rotations_json": rotations_json,
                    "model_rotation_y": model_rotation_y,
                },
                "VNCCS_QWEN_Encoder": {
                    "prompt": VNCCS_PROMPT,
                    "instruction": VNCCS_INSTRUCTION,
                },
                "KSampler": {"seed": frame_seed},
                "SaveImage": {"filename_prefix": f"PoseDirector/{name}/sprite_{outfit_name}"},
            }
            patch_ui_workflow(workflow, patches)
            api = convert_ui_to_api(workflow)
            result = submit_workflow(api, timeout=300)

            save_node = next(n for n in workflow["nodes"] if n["type"] == "SaveImage")
            frame_images = get_output_images(result["outputs"], save_node["id"])

            if frame_images:
                frame_dir = OUTPUT_DIR / name / "sprites" / outfit_name / "normal" / anim_name / direction
                frame_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frame_dir / f"frame_{kf_idx + 1:03d}.png"
                frame_path.write_bytes(frame_images[0])
                total_frames += 1

    log(f"  Generated {total_frames} outfit frames for {outfit_name}/{anim_name}")

    return {
        "status": "complete",
        "outfit": outfit_name,
        "animation": anim_name,
        "frames_per_dir": len(keyframes),
        "total_frames": total_frames,
    }


def stage_state_portrait(data: dict, details: dict, state: dict, seed_override: int | None) -> dict:
    """Re-edit a portrait with a state overlay (bruises, wet, etc)."""
    name = data["character"]["name"]
    state_name = details["state"]
    state_desc = details["description"]
    emotion = details["emotion"]
    outfit = details["outfit"]

    # Find the base portrait
    base_portrait = OUTPUT_DIR / name / "portraits" / outfit / f"{emotion}.png"
    if not base_portrait.exists():
        die(f"Base portrait not found: {base_portrait}")

    char_image_path = OUTPUT_DIR / name / "design" / "approved.png"
    seed = seed_override or data.get("generation", {}).get("seed", 42)

    # Upload both
    upload_image(base_portrait.read_bytes(), f"{name}_base.png", "PoseDirector")
    upload_image(char_image_path.read_bytes(), f"{name}_char.png", "PoseDirector")

    log(f"  State portrait: {state_name}/{emotion}")

    # Use VNCCS edit to apply state
    neutral_pose = {
        "upperarm_r": [0, 0, 0], "lowerarm_r": [0, 0, 0], "hand_r": [0, 0, 0],
        "upperarm_l": [0, 0, 0], "lowerarm_l": [0, 0, 0], "hand_l": [0, 0, 0],
        "thigh_r": [0, 0, 0], "calf_r": [0, 0, 0], "foot_r": [0, 0, 0],
        "thigh_l": [0, 0, 0], "calf_l": [0, 0, 0], "foot_l": [0, 0, 0],
        "spine_01": [0, 0, 0], "spine_02": [0, 0, 0], "spine_03": [0, 0, 0],
        "neck_01": [0, 0, 0], "head": [0, 0, 0], "pelvis": [0, 0, 0],
    }

    workflow = load_ui_workflow(WORKFLOWS_DIR / "05_sprite_bodymesh_edit.json")
    patches = {
        "LoadImage": {"image": f"PoseDirector/{name}_char.png"},
        "BodyMeshRenderer": {
            "rotations_json": json.dumps(neutral_pose),
            "model_rotation_y": 0.0,
        },
        "VNCCS_QWEN_Encoder": {
            "prompt": f"Draw character from image2 with {emotion} expression",
            "instruction": f"Draw the character from Picture 2 with a {emotion} expression and {state_desc}. "
                          f"Maintain the character's identity but add visible {state_name} effects.",
        },
        "KSampler": {"seed": seed},
        "SaveImage": {"filename_prefix": f"PoseDirector/{name}/portrait_{outfit}_{state_name}_{emotion}"},
    }
    patch_ui_workflow(workflow, patches)
    api = convert_ui_to_api(workflow)
    result = submit_workflow(api, timeout=300)

    save_node = next(n for n in workflow["nodes"] if n["type"] == "SaveImage")
    images = get_output_images(result["outputs"], save_node["id"])

    if images:
        portrait_dir = OUTPUT_DIR / name / "portraits" / f"{outfit}_{state_name}"
        portrait_dir.mkdir(parents=True, exist_ok=True)
        portrait_path = portrait_dir / f"{emotion}.png"
        portrait_path.write_bytes(images[0])
        log(f"  Saved: {portrait_path}")
        return {"status": "complete", "state": state_name, "emotion": emotion}

    return {"status": "failed", "state": state_name, "emotion": emotion}


def stage_state_sprite(data: dict, details: dict, state: dict, seed_override: int | None) -> dict:
    """Re-edit sprite frames with a state overlay."""
    name = data["character"]["name"]
    state_name = details["state"]
    state_desc = details["description"]
    outfit = details["outfit"]
    anim_name = details["animation"]
    directions = details["directions"]
    seed = seed_override or data.get("generation", {}).get("seed", 42)

    char_image_path = OUTPUT_DIR / name / "design" / "approved.png"
    upload_image(char_image_path.read_bytes(), f"{name}_char.png", "PoseDirector")

    log(f"  State sprites: {state_name}/{outfit}/{anim_name}")

    # Find base frames
    base_dir = OUTPUT_DIR / name / "sprites" / outfit / "normal" / anim_name
    if not base_dir.exists():
        die(f"Base sprite frames not found: {base_dir}")

    total_frames = 0
    for direction in directions:
        frame_files = sorted((base_dir / direction).glob("frame_*.png"))
        for frame_path in frame_files:
            # Re-edit each frame with state
            upload_image(frame_path.read_bytes(), f"{name}_base_frame.png", "PoseDirector")

            workflow = load_ui_workflow(WORKFLOWS_DIR / "05_sprite_bodymesh_edit.json")
            patches = {
                "LoadImage": {"image": f"PoseDirector/{name}_base_frame.png"},
                "BodyMeshRenderer": {
                    "rotations_json": json.dumps({}),
                    "model_rotation_y": DIRECTION_ROTATIONS.get(direction, 0.0),
                },
                "VNCCS_QWEN_Encoder": {
                    "prompt": VNCCS_PROMPT,
                    "instruction": f"Apply {state_desc} to the character in this image. "
                                  f"Keep the exact same pose and composition, only add the state effects.",
                },
                "KSampler": {"seed": seed},
                "SaveImage": {"filename_prefix": f"PoseDirector/{name}/state_{state_name}"},
            }
            patch_ui_workflow(workflow, patches)
            api = convert_ui_to_api(workflow)
            result = submit_workflow(api, timeout=300)

            save_node = next(n for n in workflow["nodes"] if n["type"] == "SaveImage")
            images = get_output_images(result["outputs"], save_node["id"])

            if images:
                out_dir = OUTPUT_DIR / name / "sprites" / outfit / state_name / anim_name / direction
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / frame_path.name
                out_path.write_bytes(images[0])
                total_frames += 1

    log(f"  Generated {total_frames} state frames for {state_name}/{anim_name}")
    return {
        "status": "complete",
        "state": state_name,
        "outfit": outfit,
        "animation": anim_name,
        "total_frames": total_frames,
    }


# ---------------------------------------------------------------------------
# Main Build Orchestrator
# ---------------------------------------------------------------------------

def build_character(
    yaml_path: Path,
    rebuild_targets: list[str] | None = None,
    seed_override: int | None = None,
    dry_run: bool = False,
    clean: bool = False,
):
    """Build a character from its YAML manifest."""
    data = load_character_yaml(yaml_path)
    char = data["character"]
    name = char["name"]
    char_dir = OUTPUT_DIR / name

    log(f"Character: {name} (tier={char.get('tier', 't1')})")

    # Handle --clean
    if clean:
        import shutil
        if char_dir.exists():
            shutil.rmtree(char_dir)
            log(f"  Cleaned: {char_dir}")
        state_file = char_dir / "build_state.json"
        if state_file.exists():
            state_file.unlink()

    # Load state
    state = load_build_state(char_dir)
    new_hash = compute_yaml_hash(data)

    # Make plan
    plan = make_plan(data, state, rebuild_targets)

    if not plan:
        log(f"Nothing to build — all assets up to date")
        return

    log(f"Build plan: {plan.summary()}")

    if dry_run:
        for step in plan.steps:
            log(f"  [{step['stage']}] {step['action']}: {step['details']}")
        return

    # Execute plan
    check_comfyui()
    start = time.time()
    failed = 0

    for step in plan.steps:
        stage = step["stage"]
        action = step["action"]
        details = step["details"]
        log(f"[{stage}] {action} ...")

        try:
            if stage == "generation":
                result = stage_generate(data, details, state, seed_override)
                state["generation"] = result

            elif stage == "trellis":
                # TRELLIS is only needed for t2+ characters
                tier = char.get("tier", "t1")
                if tier in ("t2", "t3"):
                    result = stage_trellis(data, state, seed_override)
                    state["trellis"] = result
                else:
                    log(f"  TRELLIS skipped (tier={tier})")
                    result = {"status": "skipped", "tier": tier}

            elif stage == "portraits":
                result = stage_portraits(data, details, state, seed_override)
                outfit = details["outfit"]
                emotion = details["emotion"]
                if "portraits" not in state:
                    state["portraits"] = {}
                state["portraits"][f"{outfit}/{emotion}"] = result

            elif stage == "sprites":
                result = stage_sprites(data, details, state, seed_override)
                sprite_key = f"{details['outfit']}/{details.get('state', 'normal')}/{details['animation']}"
                if "sprites" not in state:
                    state["sprites"] = {}
                state["sprites"][sprite_key] = result

            elif stage == "outfit":
                result = stage_outfit(data, details, state, seed_override)
                sprite_key = f"{details['outfit']}/normal/{details['animation']}"
                if "sprites" not in state:
                    state["sprites"] = {}
                state["sprites"][sprite_key] = result

            elif stage == "state":
                if action == "portrait":
                    result = stage_state_portrait(data, details, state, seed_override)
                elif action == "sprite":
                    result = stage_state_sprite(data, details, state, seed_override)
                else:
                    result = {"status": "skipped"}

            # Save state after each step
            state["yaml_hash"] = new_hash
            save_build_state(char_dir, state)

        except Exception as e:
            log(f"  FAILED: {e}")
            failed += 1
            continue

    elapsed = time.time() - start
    succeeded = len(plan) - failed
    log(f"Build complete: {succeeded}/{len(plan)} steps in {elapsed:.1f}s")
    if failed:
        log(f"  {failed} steps failed")


def main():
    parser = argparse.ArgumentParser(
        description="Declarative character asset build system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s characters/marty.yaml                    # Build everything missing
  %(prog)s characters/marty.yaml --dry-run           # Show what would be built
  %(prog)s characters/marty.yaml --rebuild portraits # Force regenerate portraits
  %(prog)s characters/marty.yaml --rebuild sprites.casual.normal.idle
  %(prog)s characters/marty.yaml --clean             # Delete all and rebuild
  %(prog)s characters/*.yaml                          # Build multiple characters
""",
    )
    parser.add_argument("yaml_files", nargs="+", help="Character YAML manifest(s)")
    parser.add_argument("--rebuild", nargs="*", default=None,
                       help="Force rebuild specific packages (e.g. portraits, sprites.outfit.state.animation)")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for this run")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated, don't execute")
    parser.add_argument("--clean", action="store_true", help="Delete all outputs and rebuild from scratch")

    args = parser.parse_args()

    for yaml_file in args.yaml_files:
        path = Path(yaml_file)
        if not path.exists():
            die(f"File not found: {path}")

        build_character(
            yaml_path=path,
            rebuild_targets=args.rebuild,
            seed_override=args.seed,
            dry_run=args.dry_run,
            clean=args.clean,
        )


if __name__ == "__main__":
    main()
