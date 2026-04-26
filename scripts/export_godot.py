#!/usr/bin/env python3
"""Export batch sprite output to Godot 4.x project structure.

Takes a batch manifest.json and creates:
  - Organized sprite directory for Godot import
  - SpriteFrames .tres resource per character
  - CharacterSprite3D GDScript controller
  - character_config.json with animation metadata

Usage:
    python scripts/export_godot.py output/batch_20260426_125130 --name luna --project ~/game
    python scripts/export_godot.py output/batch_20260426_125130 --name luna --project ~/game --resize 512
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Godot directions in draw order (for mirror optimization)
DIRECTIONS = ["front", "right", "back", "left"]

GDSCRIPT_CONTROLLER = '''\
@tool
extends Sprite3D
class_name CharacterSprite3D
## Direction-aware sprite animation controller for 2.5D characters.
##
## Attach to a Sprite3D node. Set character_name and the controller
## loads all sprite frames and handles animation playback with
## direction switching.

@export var character_name: String = "":
    set(v):
        character_name = v
        if Engine.is_editor_hint():
            _load_animations()

@export var fps: float = 12.0
@export var pixel_size: float = 0.01

var animations: Dictionary = {}  # anim_name -> {direction -> [textures]}
var current_anim: String = "idle"
var current_dir: String = "front"
var frame: int = 0
var frame_timer: float = 0.0
var _config: Dictionary = {}

# Mirror left/right to halve sprite count
var mirror_enabled: bool = true


func _ready():
    _load_animations()
    play("idle", "front")


func _load_animations():
    if character_name.is_empty():
        return

    var base_path = "res://sprites/%s" % character_name
    var config_path = base_path.path_join("character_config.json")

    if FileAccess.file_exists(config_path):
        var f = FileAccess.open(config_path, FileAccess.READ)
        _config = JSON.parse_string(f.get_as_text())
        f.close()

    animations.clear()

    # Load from organized directory structure: sprites/char/anim_dir/frame.png
    var dir = DirAccess.open(base_path)
    if dir == null:
        push_warning("CharacterSprite3D: no sprites at %s" % base_path)
        return

    dir.list_dir_begin()
    var anim_name = dir.get_next()
    while anim_name != "":
        if dir.current_is_dir() and anim_name != "." and anim_name != "..":
            _load_animation_dir(base_path.path_join(anim_name), anim_name)
        anim_name = dir.get_next()
    dir.list_dir_end()


func _load_animation_dir(anim_path: String, anim_name: String):
    animations[anim_name] = {}

    var dir = DirAccess.open(anim_path)
    if dir == null:
        return

    dir.list_dir_begin()
    var dir_name = dir.get_next()
    while dir_name != "":
        if dir.current_is_dir() and dir_name in ["front", "right", "back", "left"]:
            var textures = _load_frame_textures(anim_path.path_join(dir_name))
            if textures.size() > 0:
                animations[anim_name][dir_name] = textures
        dir_name = dir.get_next()
    dir.list_dir_end()


func _load_frame_textures(frames_dir: String) -> Array:
    var textures = []
    var sorted_files = []

    var dir = DirAccess.open(frames_dir)
    if dir == null:
        return textures

    dir.list_dir_begin()
    var fname = dir.get_next()
    while fname != "":
        if fname.ends_with(".png") or fname.ends_with(".webp"):
            sorted_files.append(fname)
        fname = dir.get_next()
    dir.list_dir_end()

    sorted_files.sort()

    for f in sorted_files:
        var tex = load(frames_dir.path_join(f))
        if tex:
            textures.append(tex)

    return textures


func play(anim_name: String, direction: String = "front"):
    if current_anim == anim_name and current_dir == direction:
        return

    if not animations.has(anim_name):
        push_warning("CharacterSprite3D: unknown animation '%s'" % anim_name)
        return

    current_anim = anim_name
    current_dir = direction
    frame = 0
    frame_timer = 0.0
    _update_texture()


func set_direction(direction: String):
    if current_dir == direction:
        return
    current_dir = direction
    _update_texture()


func _process(delta):
    if not animations.has(current_anim):
        return

    var anim_fps = _config.get(current_anim, {}).get("fps", fps)
    var loop = _config.get(current_anim, {}).get("loop", true)

    frame_timer += delta
    if frame_timer >= 1.0 / anim_fps:
        frame_timer = 0.0
        var dir_frames = _get_current_frames()
        if dir_frames.size() == 0:
            return

        frame += 1
        if frame >= dir_frames.size():
            if loop:
                frame = 0
            else:
                frame = dir_frames.size() - 1
                return
        _update_texture()


func _get_current_frames() -> Array:
    if not animations.has(current_anim):
        return []

    var anim = animations[current_anim]

    # Direct lookup
    if anim.has(current_dir):
        return anim[current_dir]

    # Mirror: right <-> left
    if mirror_enabled:
        if current_dir == "left" and anim.has("right"):
            return anim["right"]
        if current_dir == "right" and anim.has("left"):
            return anim["left"]

    # Fallback: any available direction
    for d in DIRECTIONS:
        if anim.has(d):
            return anim[d]

    return []


func _update_texture():
    var dir_frames = _get_current_frames()
    if dir_frames.size() == 0:
        return

    var idx = mini(frame, dir_frames.size() - 1)
    texture = dir_frames[idx]

    # Mirror horizontally for left direction if using right sprites
    if mirror_enabled and current_dir == "left":
        if animations.has(current_anim) and not animations[current_anim].has("left"):
            flip_h = true
            return
    flip_h = false
'''


def export_godot(
    manifest_path: str | Path,
    character_name: str,
    project_dir: str | Path,
    resize: int | None = None,
):
    """Export batch sprites to Godot project structure."""
    manifest_path = Path(manifest_path)
    project_dir = Path(project_dir)

    with open(manifest_path) as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    if not frames:
        print(f"[godot] ERROR: No frames in manifest {manifest_path}")
        sys.exit(1)

    # Create Godot sprite directory
    sprite_dir = project_dir / "sprites" / character_name
    sprite_dir.mkdir(parents=True, exist_ok=True)
    print(f"[godot] Export: {manifest_path} → {sprite_dir}")

    # Group frames by animation and direction
    anim_dirs: dict[str, dict[str, list[dict]]] = {}
    for frame_info in frames:
        anim = frame_info["animation"]
        direction = frame_info["direction"]
        anim_dirs.setdefault(anim, {}).setdefault(direction, []).append(frame_info)

    # Config for animation metadata
    config: dict = {}

    copied_count = 0
    for anim_name, directions in anim_dirs.items():
        config[anim_name] = {
            "fps": 12,
            "loop": True,
            "directions": list(directions.keys()),
            "frames_per_direction": len(next(iter(directions.values()))),
        }

        for direction, dir_frames in directions.items():
            # Create frame directory
            frame_dir = sprite_dir / anim_name / direction
            frame_dir.mkdir(parents=True, exist_ok=True)

            for i, frame_info in enumerate(dir_frames):
                src = Path(frame_info["path"])
                if not src.exists():
                    print(f"[godot] WARNING: Missing {src}")
                    continue

                dst = frame_dir / f"frame_{i:03d}.png"

                if resize:
                    from PIL import Image
                    img = Image.open(src)
                    img = img.resize((resize, resize), Image.LANCZOS)
                    img.save(dst)
                else:
                    shutil.copy2(src, dst)
                copied_count += 1

    # Write character config
    config_path = sprite_dir / "character_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # Write GDScript controller
    script_path = sprite_dir / f"{character_name}_sprite.gd"
    script_path.write_text(GDSCRIPT_CONTROLLER)

    # Write scene file (.tscn)
    scene = _build_scene_file(character_name)
    scene_path = sprite_dir / f"{character_name}.tscn"
    scene_path.write_text(scene)

    print(f"[godot] Copied {copied_count} frames across {len(anim_dirs)} animations")
    print(f"[godot] Config: {config_path}")
    print(f"[godot] Scene: {scene_path}")
    print(f"[godot] Script: {script_path}")
    print(f"\n[godot] In Godot: Copy {sprite_dir} → res://sprites/{character_name}/")


def _build_scene_file(character_name: str) -> str:
    """Generate a Godot .tscn scene file for the character."""
    uid = "uid://c_sprite_{name}".format(name=character_name[:8])
    return f"""[gd_scene load_steps=2 format=3 uid="{uid}"]

[ext_resource type="Script" path="res://sprites/{character_name}/{character_name}_sprite.gd" id="1_script"]

[node name="{character_name.capitalize()}" type="Sprite3D"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
pixel_size = 0.01
billboard = 2
texture_filter = 0
script = ExtResource("1_script")
character_name = "{character_name}"
fps = 12.0
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export sprites to Godot 4.x")
    parser.add_argument("manifest", help="Path to batch manifest.json")
    parser.add_argument("--name", required=True, help="Character name (e.g. luna)")
    parser.add_argument("--project", default=".", help="Godot project root directory")
    parser.add_argument("--resize", type=int, default=None, help="Resize frames (e.g. 512)")
    args = parser.parse_args()

    export_godot(args.manifest, args.name, args.project, args.resize)


if __name__ == "__main__":
    main()
