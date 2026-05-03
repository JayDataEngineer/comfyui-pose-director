#!/usr/bin/env python3
"""Generate animated sprite frames using HY-Motion + VNCCS sprite pipeline.

Takes a text description of an animation and a character image, then:
1. OpenRouter rewrites the prompt for HY-Motion
2. HY-Motion Ray service generates physically realistic motion → NPZ
3. NPZ is converted to keyframe poses via hymotion_converter
4. Each pose x each direction is rendered through the VNCCS sprite pipeline
5. Outputs transparent PNG sprite frames

Usage:
    python scripts/gen_motion_sprites.py \
        --character character.png \
        --prompt "A person wiping down a bar counter" \
        --directions front right back left \
        --num-keyframes 6

    # With custom output dir and seed
    python scripts/gen_motion_sprites.py \
        --character luna.png \
        --prompt "A person waving hello" \
        --directions front \
        --num-keyframes 4 \
        --seed 42 \
        --output-dir output/sprites/luna/

    # Use existing NPZ (skip motion generation)
    python scripts/gen_motion_sprites.py \
        --character luna.png \
        --prompt "waving" \
        --npz-path output/motion.npz \
        --directions front
"""

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

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

from pipeline.comfyui_client import COMFYUI_URL, submit_workflow, get_output_images, upload_image
from pipeline.ui_to_api import load_ui_workflow, patch_ui_workflow, convert_ui_to_api
from utils.hymotion_converter import extract_keyframes

COMFYUI = os.environ.get("COMFYUI_URL", "http://100.86.69.57:18800/comfyui")
RAY_PROXY = os.environ.get("RAY_PROXY", "http://100.86.69.57:18800")
HY_MOTION_URL = os.environ.get("HY_MOTION_URL", f"{RAY_PROXY}/3d/hy-motion")
WORKFLOW_PATH = PROJECT_ROOT / "workflows" / "05_sprite_bodymesh_edit.json"

DIRECTION_MAP = {"front": 0, "right": 90, "back": 180, "left": 270}

SYSTEM_PROMPT_REWRITE = """You rewrite text descriptions into optimal prompts for HY-Motion, a text-to-3D-motion generation model.

Rules:
- Output JSON with "prompt" and "duration_seconds" fields
- Prompt: under 60 words, English only
- Focus on ACTION descriptions: limb movements, torso motion, body mechanics
- DO NOT include: emotions, clothing, appearance, environment, camera angles, multi-person
- DO NOT include: looping, in-place, or seamless instructions
- Duration: estimate in seconds (1-10s range). Simple actions = 2-3s, complex = 4-6s
- Be specific about body parts and direction of movement

Examples:
Input: "bartender wiping counter"
Output: {"prompt": "A person stands behind a surface, extending their right arm forward and sweeping it left to right repeatedly in smooth horizontal arcs, pulling back slightly between each sweep", "duration_seconds": 3.0}

Input: "friendly wave"
Output: {"prompt": "A person raises their right arm with hand open, then oscillates the hand back and forth from the wrist in a rhythmic greeting motion", "duration_seconds": 2.5}

Input: "combat idle"
Output: {"prompt": "A person stands with feet shoulder-width apart, knees slightly bent, both fists raised to chin level with elbows tucked, shifting weight between feet in a subtle rhythmic bounce", "duration_seconds": 4.0}

Now rewrite:"""


def rewrite_prompt(prompt: str, model: str = "deepseek/deepseek-v4-pro") -> tuple[str, float]:
    """Use OpenRouter to rewrite prompt for HY-Motion. Returns (prompt, duration_seconds)."""
    from utils.api_client import call_llm, resolve_api_key

    api_key = resolve_api_key("", "https://openrouter.ai/api/v1")
    if not api_key or api_key == "not-needed":
        print("  [WARN] No API key for prompt rewriting, using raw prompt")
        return prompt, 3.0

    try:
        response = call_llm(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model,
            system_prompt=SYSTEM_PROMPT_REWRITE,
            user_message=prompt,
            temperature=0.5,
        )
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return result["prompt"], float(result.get("duration_seconds", 3.0))
    except Exception as e:
        print(f"  [WARN] Prompt rewrite failed ({e}), using raw prompt")
        return prompt, 3.0


def generate_motion(prompt: str, duration: float) -> Path | None:
    """Call HY-Motion Ray service to generate motion. Returns path to local NPZ file."""
    print(f"  Calling HY-Motion service: {HY_MOTION_URL}")
    try:
        resp = requests.post(
            HY_MOTION_URL,
            json={"prompt": prompt, "duration": duration, "format": "npz"},
            timeout=300,
        )
        if resp.status_code != 200:
            print(f"  HY-Motion error: {resp.status_code} {resp.text[:200]}")
            return None

        # Save response body as NPZ
        output_dir = PROJECT_ROOT / "output" / "hymotion"
        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / f"motion_{int(time.time())}.npz"
        npz_path.write_bytes(resp.content)
        print(f"  NPZ saved: {npz_path} ({len(resp.content) // 1024}KB)")
        return npz_path

    except requests.exceptions.ConnectionError:
        print(f"  HY-Motion service not reachable at {HY_MOTION_URL}")
        print(f"  Is the Ray service deployed? Check: curl {RAY_PROXY}/-/routes")
        return None
    except Exception as e:
        print(f"  HY-Motion failed: {e}")
        return None


def generate_sprite(character_image: str, pose: dict, direction: float,
                    seed: int, prefix: str) -> bytes | None:
    """Generate a single sprite frame. Returns PNG bytes or None."""
    wf = load_ui_workflow(str(WORKFLOW_PATH))
    patch_ui_workflow(wf, {
        "LoadImage": {"image": character_image},
        "BodyMeshRenderer": {
            "rotations_json": json.dumps(pose),
            "model_rotation_y": direction,
        },
        "KSampler": {"seed": seed},
        "SaveImage": {"filename_prefix": f"PoseDirector/{prefix}"},
    })
    api = convert_ui_to_api(wf)

    try:
        result = submit_workflow(api, timeout=300)
        outputs = result.get("outputs", {})
        images = get_output_images(outputs, 12)
        return images[0] if images else None
    except Exception as e:
        print(f"    Sprite generation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate animated sprites using HY-Motion + VNCCS")
    parser.add_argument("--character", required=True, help="Path to character image PNG")
    parser.add_argument("--prompt", required=True, help="Animation description")
    parser.add_argument("--directions", nargs="+", default=["front"],
                        choices=["front", "right", "back", "left"],
                        help="Viewing directions (default: front)")
    parser.add_argument("--num-keyframes", type=int, default=6, help="Sprite frames to extract")
    parser.add_argument("--seed", type=int, default=42, help="Seed for consistency")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--rewrite-model", type=str, default="deepseek/deepseek-v4-pro",
                        help="OpenRouter model for prompt rewriting")
    parser.add_argument("--npz-path", type=str, default=None,
                        help="Use existing NPZ file instead of generating motion")
    args = parser.parse_args()

    # Output dir
    char_name = Path(args.character).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "output" / "sprites" / f"{char_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"ComfyUI:   {COMFYUI}")
    print(f"Character: {args.character}")
    print(f"Prompt:    {args.prompt}")
    print(f"Directions:{args.directions}")
    print(f"Output:    {output_dir}")
    print()

    # Step 1: Rewrite prompt
    print("[1/4] Rewriting prompt via OpenRouter...")
    rewritten, duration = rewrite_prompt(args.prompt, model=args.rewrite_model)
    print(f"  Rewritten: {rewritten}")
    print(f"  Duration:  {duration}s")
    print()

    # Step 2: Generate motion
    if args.npz_path:
        npz_path = Path(args.npz_path)
        print(f"[2/4] Using existing NPZ: {npz_path}")
    else:
        print("[2/4] Generating motion via HY-Motion Ray service...")
        npz_path = generate_motion(rewritten, duration)

    if not npz_path or not npz_path.exists():
        print("FAILED: No motion data. Cannot proceed.")
        sys.exit(1)
    print()

    # Step 3: Extract keyframes
    print(f"[3/4] Extracting {args.num_keyframes} keyframes...")
    keyframes = extract_keyframes(npz_path, num_keyframes=args.num_keyframes)
    print(f"  Got {len(keyframes)} keyframe poses")
    for i, kf in enumerate(keyframes):
        joints = [k for k, v in kf.items() if any(abs(a) > 5 for a in v)]
        print(f"    Frame {i}: active = {joints}")
    print()

    # Step 4: Generate sprites
    print("[4/4] Generating sprite frames via ComfyUI...")
    char_path = Path(args.character)
    if not char_path.exists():
        print(f"FAILED: Character image not found: {char_path}")
        sys.exit(1)

    upload_image(char_path.read_bytes(), char_path.name)
    char_comfyui_name = f"PoseDirector/{char_path.name}"
    print(f"  Uploaded: {char_comfyui_name}")

    manifest = {
        "character": char_name,
        "prompt_original": args.prompt,
        "prompt_rewritten": rewritten,
        "duration_seconds": duration,
        "npz_path": str(npz_path),
        "num_keyframes": len(keyframes),
        "seed": args.seed,
        "directions": args.directions,
        "timestamp": timestamp,
        "frames": [],
    }

    total = len(keyframes) * len(args.directions)
    done = 0
    for direction_name in args.directions:
        direction_angle = DIRECTION_MAP[direction_name]
        for fi, pose in enumerate(keyframes):
            done += 1
            prefix = f"{char_name}_{direction_name}_f{fi}"
            print(f"  [{done}/{total}] {direction_name} frame {fi}...", end=" ", flush=True)

            png_bytes = generate_sprite(
                character_image=char_comfyui_name,
                pose=pose,
                direction=direction_angle,
                seed=args.seed,
                prefix=prefix,
            )

            if png_bytes:
                out_path = output_dir / f"{prefix}.png"
                out_path.write_bytes(png_bytes)
                print(f"OK ({len(png_bytes) // 1024}KB)")
                manifest["frames"].append({
                    "file": out_path.name,
                    "direction": direction_name,
                    "frame_index": fi,
                    "pose": pose,
                })
            else:
                print("FAILED")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone! {len(manifest['frames'])} frames saved to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
