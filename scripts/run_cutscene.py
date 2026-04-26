#!/usr/bin/env python3
"""Generate cutscene MP4 from sprite frames using LTX Video 2.3.

Takes selected sprite frames as keyframe guides and generates a smooth
video transition between them. Outputs MP4 for Godot VideoStreamPlayer.

Usage:
    # From batch output directory
    python scripts/run_cutscene.py output/batch_20260426_125130 --animation action

    # From specific frames
    python scripts/run_cutscene.py --frames frame1.png frame2.png frame3.png --prompt "Dramatic overhead strike"

    # With custom FPS and frame count
    python scripts/run_cutscene.py output/batch_20260426 --animation wave --fps 30 --total-frames 65
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

import requests
from PIL import Image

WORKFLOWS_DIR = PROJECT_ROOT / "workflows"


def log(msg: str):
    print(f"[cutscene] {msg}", flush=True)


def collect_frames_from_batch(
    batch_dir: str | Path,
    animation: str,
    direction: str = "front",
) -> list[Path]:
    """Collect sprite frame paths from a batch output directory."""
    batch_dir = Path(batch_dir)
    anim_dir = batch_dir / animation

    if not anim_dir.exists():
        raise FileNotFoundError(f"Animation directory not found: {anim_dir}")

    # Find frames matching the direction
    frames = sorted(
        p for p in anim_dir.glob("*.png")
        if direction in p.name
    )

    if not frames:
        # Try without direction filter
        frames = sorted(anim_dir.glob("*.png"))

    if not frames:
        raise FileNotFoundError(f"No frames found in {anim_dir}")

    return frames


def resize_for_ltx(img_path: Path, size: int = 512) -> bytes:
    """Resize image to LTX-compatible resolution (must be divisible by 8)."""
    img = Image.open(img_path)
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_cutscene(
    frame_paths: list[str | Path],
    prompt: str = "Smooth character animation",
    fps: int = 24,
    total_frames: int = 33,
    steps: int = 30,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    resolution: int = 512,
) -> Path:
    """Generate cutscene MP4 from keyframe sprite images.

    Args:
        frame_paths: Paths to keyframe images (2-4 recommended).
        prompt: Motion description for LTX Video conditioning.
        fps: Output video FPS.
        total_frames: Total video frames (must satisfy keyframe spacing).
        steps: KSampler steps.
        seed: Random seed.
        output_dir: Where to save output.
        resolution: Video resolution (must be divisible by 8).

    Returns:
        Path to output MP4.
    """
    if seed is None:
        seed = random.randint(0, 999999)

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "output" / f"cutscene_{ts}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate frame count
    num_keyframes = len(frame_paths)
    if num_keyframes < 2:
        log("ERROR: Need at least 2 keyframes for cutscene")
        sys.exit(1)
    if num_keyframes > 4:
        log(f"WARNING: LTX workflow supports 4 keyframe guides, using first 4 of {num_keyframes}")
        frame_paths = frame_paths[:4]
        num_keyframes = 4

    # Calculate keyframe frame indices (evenly spaced, divisible by 8)
    # Total frames must allow last guide + room for interpolation
    last_frame = total_frames - 1
    # Ensure last_frame is divisible by 8
    last_frame = (last_frame // 8) * 8
    if last_frame < 8:
        last_frame = 8

    frame_indices = [
        int(i * last_frame / (num_keyframes - 1))
        for i in range(num_keyframes)
    ]
    # Ensure all indices are divisible by 8
    frame_indices = [(idx // 8) * 8 for idx in frame_indices]
    frame_indices[0] = 0  # First frame is always 0

    log(f"Keyframe indices: {frame_indices} (total {last_frame + 1} frames)")

    # Verify ComfyUI is up
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        assert r.status_code == 200
    except Exception as e:
        log(f"ERROR: ComfyUI not reachable at {COMFYUI_URL}: {e}")
        sys.exit(1)
    log(f"ComfyUI ready at {COMFYUI_URL}")

    # Upload keyframe images
    uploaded_names = []
    for i, fp in enumerate(frame_paths):
        fp = Path(fp)
        if not fp.exists():
            log(f"ERROR: Frame not found: {fp}")
            sys.exit(1)

        img_bytes = resize_for_ltx(fp, resolution)
        name = f"edited_kf_{i}.png"
        upload_image(img_bytes, name, "PoseDirector")
        uploaded_names.append(f"PoseDirector/{name}")
        log(f"Uploaded keyframe {i}: {fp.name} → {name}")

    # Load and patch LTX video workflow
    workflow = load_ui_workflow(WORKFLOWS_DIR / "04_ltx_video_assembly.json")

    # Patch keyframe images
    load_image_nodes = [n for n in workflow["nodes"] if n["type"] == "LoadImage"]
    guide_nodes = [n for n in workflow["nodes"] if n["type"] == "LTXVAddGuideAdvanced"]

    for i, name in enumerate(uploaded_names):
        if i < len(load_image_nodes):
            load_image_nodes[i]["widgets_values"][0] = name

    # Patch prompt
    patch_ui_workflow(workflow, {
        "CLIPTextEncode": {"text": prompt},
        "KSampler": {
            "seed": seed,
            "steps": steps,
        },
        "EmptyLTXVLatentVideo": {
            "width": resolution,
            "height": resolution,
            "length": last_frame + 1,
        },
    })

    # Save workflow for debugging
    save_ui_workflow(output_dir / "workflow_ui.json", workflow)

    # Convert and submit
    api = convert_ui_to_api(workflow)
    save_api_workflow(output_dir / "workflow_api.json", api)

    log(f"Generating cutscene: {num_keyframes} keyframes, {last_frame + 1} frames, {steps} steps")
    start = time.time()

    result = submit_workflow(api)
    elapsed = time.time() - start
    log(f"LTX Video completed in {elapsed:.1f}s")

    # Get video output
    vhs_node = next(n for n in workflow["nodes"] if n["type"] == "VHS_VideoCombine")
    vhs_node_id = str(vhs_node["id"])
    video_bytes = get_output_video(result["outputs"], vhs_node_id)

    if not video_bytes:
        log("ERROR: No video output")
        return output_dir

    # Save video
    video_path = output_dir / "cutscene.mp4"
    video_path.write_bytes(video_bytes)
    log(f"Saved: {video_path} ({len(video_bytes)} bytes, {elapsed:.1f}s)")

    # Save metadata for Godot
    metadata = {
        "animation": "cutscene",
        "prompt": prompt,
        "keyframes": [str(fp) for fp in frame_paths],
        "keyframe_indices": frame_indices,
        "total_frames": last_frame + 1,
        "fps": fps,
        "resolution": resolution,
        "seed": seed,
        "steps": steps,
        "duration_s": round((last_frame + 1) / fps, 2),
    }
    meta_path = output_dir / "cutscene_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    log(f"Metadata: {meta_path}")
    log(f"Duration: {metadata['duration_s']}s at {fps} FPS")
    return output_dir


def export_cutscene_godot(
    cutscene_dir: str | Path,
    character_name: str,
    project_dir: str | Path,
):
    """Export cutscene MP4 to Godot project structure."""
    cutscene_dir = Path(cutscene_dir)
    project_dir = Path(project_dir)

    video_src = cutscene_dir / "cutscene.mp4"
    meta_src = cutscene_dir / "cutscene_meta.json"

    if not video_src.exists():
        log(f"ERROR: No cutscene.mp4 in {cutscene_dir}")
        sys.exit(1)

    # Create Godot cutscene directory
    cs_dir = project_dir / "cutscenes" / character_name
    cs_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    dst_video = cs_dir / video_src.name
    shutil.copy2(video_src, dst_video)

    if meta_src.exists():
        shutil.copy2(meta_src, cs_dir / "cutscene_meta.json")

    # Generate Godot scene for VideoStreamPlayer
    scene_content = f"""[gd_scene load_steps=2 format=3]

[ext_resource type="VideoStream" path="res://cutscenes/{character_name}/cutscene.mp4" id="1_video"]

[node name="Cutscene_{character_name}" type="Control"]
layout_mode = 3
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0
grow_horizontal = 2
grow_vertical = 2

[node name="VideoStreamPlayer" type="VideoStreamPlayer" parent="."]
layout_mode = 1
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0
grow_horizontal = 2
grow_vertical = 2
stream = ExtResource("1_video")
"""
    scene_path = cs_dir / f"cutscene_{character_name}.tscn"
    scene_path.write_text(scene_content)

    # Generate GDScript helper
    gdscript = '''@tool
extends Control
## Cutscene player with fade transitions.

@export var character_name: String = ""
@export var auto_play: bool = false
@export var fade_duration: float = 0.5

var _player: VideoStreamPlayer
signal cutscene_finished


func _ready():
    _player = $VideoStreamPlayer
    _player.connect("finished", _on_finished)
    if auto_play and not Engine.is_editor_hint():
        play()


func play():
    modulate.a = 0.0
    _player.play()
    _fade_in()


func stop():
    _fade_out(func():
        _player.stop()
        cutscene_finished.emit()
    )


func _on_finished():
    _fade_out(func():
        cutscene_finished.emit()
    )


func _fade_in():
    var tween = create_tween()
    tween.tween_property(self, "modulate:a", 1.0, fade_duration)


func _fade_out(callback: Callable):
    var tween = create_tween()
    tween.tween_property(self, "modulate:a", 0.0, fade_duration)
    tween.tween_callback(callback)
'''
    script_path = cs_dir / f"cutscene_{character_name}.gd"
    script_path.write_text(gdscript)

    log(f"Exported to: {cs_dir}")
    log(f"  Video: {dst_video}")
    log(f"  Scene: {scene_path}")
    log(f"  Script: {script_path}")
    log(f"\nIn Godot: Copy {cs_dir} → res://cutscenes/{character_name}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate cutscene from sprite frames")
    parser.add_argument("batch_dir", nargs="?", default=None,
                       help="Batch output directory with manifest.json")
    parser.add_argument("--animation", default=None,
                       help="Animation name within batch dir")
    parser.add_argument("--direction", default="front",
                       help="Direction to use for frames")
    parser.add_argument("--frames", nargs="+", default=None,
                       help="Specific frame image paths")
    parser.add_argument("--prompt", default="Smooth character animation",
                       help="Motion description for LTX Video")
    parser.add_argument("--fps", type=int, default=24, help="Output FPS")
    parser.add_argument("--total-frames", type=int, default=33,
                       help="Total video frames (adjusted to be divisible by 8+1)")
    parser.add_argument("--steps", type=int, default=30, help="KSampler steps")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Video resolution (must be divisible by 8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--export-godot", default=None,
                       help="Export to Godot project dir after generation")
    parser.add_argument("--name", default="character",
                       help="Character name for Godot export")
    args = parser.parse_args()

    # Collect frame paths
    if args.frames:
        frame_paths = [Path(f) for f in args.frames]
    elif args.batch_dir and args.animation:
        frame_paths = collect_frames_from_batch(
            args.batch_dir, args.animation, args.direction
        )
        log(f"Found {len(frame_paths)} frames for {args.animation}/{args.direction}")
    else:
        parser.error("Specify --frames OR --batch-dir with --animation")

    output_dir = run_cutscene(
        frame_paths=frame_paths,
        prompt=args.prompt,
        fps=args.fps,
        total_frames=args.total_frames,
        steps=args.steps,
        seed=args.seed,
        output_dir=args.output,
        resolution=args.resolution,
    )

    if args.export_godot:
        export_cutscene_godot(output_dir, args.name, args.export_godot)


if __name__ == "__main__":
    main()
