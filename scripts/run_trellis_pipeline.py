#!/usr/bin/env python3
"""TRELLIS 3D model pipeline: character image → 3D GLB → rendered keyframes.

Uses the Tech Noir Ray infrastructure for 3D generation, then renders
turntable keyframes locally with pyrender.

Usage:
    # From existing character image
    python scripts/run_trellis_pipeline.py --image output/character.png

    # With custom camera angles
    python scripts/run_trellis_pipeline.py --image character.png --angles 8

    # Skip 3D generation (use existing GLB)
    python scripts/run_trellis_pipeline.py --glb output/model.glb --image character.png

    # Export to Godot
    python scripts/run_trellis_pipeline.py --image character.png --export-godot ~/game --name luna
"""

import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

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


def log(msg: str):
    print(f"[trellis] {msg}", flush=True)


def generate_3d_via_ray(image_bytes: bytes, ray_url: str = "http://localhost:8000") -> bytes:
    """Generate 3D GLB from image using Ray TRELLIS deployment."""
    import httpx

    log(f"Calling Ray TRELLIS at {ray_url}/3d/trellis...")
    with httpx.Client(timeout=600) as client:
        resp = client.post(
            f"{ray_url}/3d/trellis",
            files={"image": ("input.png", image_bytes, "image/png")},
            data={"output_format": "glb", "resolution": "512"},
        )
        resp.raise_for_status()
        return resp.content


def generate_3d_via_subprocess(image_bytes: bytes) -> bytes:
    """Generate 3D GLB from image using TRELLIS subprocess (fallback)."""
    import tempfile
    import subprocess

    trellis_dir = Path.home() / "Documents/programs/ray/infra/repos/TRELLIS.2"
    trellis_python = trellis_dir / ".venv/bin/python"
    wrapper = Path.home() / "Documents/programs/ray/services/creative/wrappers/trellis_cli.py"

    if not trellis_dir.exists():
        trellis_dir = Path.home() / "Documents/programs/TRELLIS.2"
        trellis_python = trellis_dir / ".venv/bin/python"

    if not trellis_python.exists():
        log("ERROR: TRELLIS not found. Install via Ray setup or manually.")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="trellis_") as tmpdir:
        input_path = Path(tmpdir) / "input.png"
        output_path = Path(tmpdir) / "output.glb"
        input_path.write_bytes(image_bytes)

        cmd = [
            str(trellis_python), str(wrapper),
            "--image", str(input_path),
            "--output", str(output_path),
            "--resolution", "512",
        ]
        log(f"Running TRELLIS subprocess: {trellis_dir.name}")
        result = subprocess.run(cmd, cwd=str(trellis_dir), capture_output=True, timeout=600)

        if result.returncode != 0:
            log(f"STDERR: {result.stderr.decode()[-500:]}")
            raise RuntimeError(f"TRELLIS failed with code {result.returncode}")

        if not output_path.exists():
            raise RuntimeError("TRELLIS did not produce output")

        return output_path.read_bytes()


def render_turntable(
    glb_bytes: bytes,
    num_angles: int = 8,
    resolution: int = 512,
) -> list[tuple[float, bytes]]:
    """Render GLB from multiple camera angles (turntable).

    Returns list of (angle_degrees, png_bytes) tuples.
    """
    import trimesh
    from PIL import Image as PILImage

    log(f"Loading GLB mesh ({len(glb_bytes)} bytes)...")
    scene = trimesh.load(
        io.BytesIO(glb_bytes),
        file_type="glb",
        force="scene",
    )

    # Get combined geometry bounds
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2
    extent = bounds[1] - bounds[0]
    max_extent = max(extent)

    # Camera distance with padding
    cam_dist = max_extent * 2.0

    renders = []
    for i in range(num_angles):
        angle = i * (360.0 / num_angles)
        theta = np.radians(angle)

        # Camera position (orbiting around Y-up center)
        cam_x = center[0] + cam_dist * np.sin(theta)
        cam_z = center[2] + cam_dist * np.cos(theta)
        cam_y = center[1] + max_extent * 0.3  # Slightly above center

        # Create a copy of the scene for this render
        render_scene = scene.copy()

        try:
            import pyrender
            # Set up pyrender scene
            py_scene = pyrender.Scene(bg_color=[40, 40, 40, 255])

            # Add all meshes
            for node in render_scene.graph.nodes:
                mesh = render_scene.get_mesh(node)
                if mesh is not None:
                    py_mesh = pyrender.Mesh.from_trimesh(mesh)
                    py_scene.add(py_mesh)

            # Camera
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)

            cam_pose = np.eye(4)
            cam_pose[:3, 3] = [cam_x, cam_y, cam_z]
            # Look at center
            forward = np.array([center[0] - cam_x, center[1] - cam_y, center[2] - cam_z])
            forward = forward / np.linalg.norm(forward)
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            if np.linalg.norm(right) < 1e-6:
                right = np.array([1, 0, 0])
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            cam_pose[:3, 0] = right
            cam_pose[:3, 1] = up
            cam_pose[:3, 2] = -forward

            py_scene.add(camera, pose=cam_pose)
            light_pose = cam_pose.copy()
            py_scene.add(light, pose=light_pose)

            # Render
            renderer = pyrender.OffscreenRenderer(resolution, resolution)
            color, _ = renderer.render(py_scene)
            renderer.delete()

            img = PILImage.fromarray(color)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            renders.append((angle, buf.getvalue()))

        except ImportError:
            log("WARNING: pyrender not available, using trimesh rendering (lower quality)")
            # Fallback: trimesh simple render
            try:
                png = scene.save_image(resolution=resolution)
                renders.append((angle, png))
            except Exception as e:
                log(f"  Render failed for angle {angle}: {e}")
                continue

    return renders


def run_trellis_pipeline(
    image_path: str | Path,
    glb_path: str | Path | None = None,
    num_angles: int = 8,
    resolution: int = 512,
    output_dir: str | Path | None = None,
    use_ray: bool = True,
    ray_url: str = "http://localhost:8000",
) -> Path:
    """Run the TRELLIS pipeline: image → 3D GLB → rendered keyframes."""
    from PIL import Image as PILImage

    image_path = Path(image_path)
    if not image_path.exists():
        log(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "output" / f"trellis_{ts}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read character image
    image_bytes = image_path.read_bytes()
    log(f"Character image: {image_path.name} ({len(image_bytes)} bytes)")

    # Stage 1: Generate 3D model
    if glb_path:
        glb_path = Path(glb_path)
        if not glb_path.exists():
            log(f"ERROR: GLB not found: {glb_path}")
            sys.exit(1)
        glb_bytes = glb_path.read_bytes()
        log(f"Using existing GLB: {glb_path.name} ({len(glb_bytes)} bytes)")
    else:
        log("Generating 3D model...")
        t0 = time.time()

        try:
            if use_ray:
                glb_bytes = generate_3d_via_ray(image_bytes, ray_url)
            else:
                glb_bytes = generate_3d_via_subprocess(image_bytes)
        except Exception as e:
            if use_ray:
                log(f"Ray failed ({e}), trying subprocess fallback...")
                glb_bytes = generate_3d_via_subprocess(image_bytes)
            else:
                raise

        elapsed = time.time() - t0
        log(f"3D model generated: {len(glb_bytes)} bytes in {elapsed:.1f}s")

        # Save GLB
        (output_dir / "model.glb").write_bytes(glb_bytes)

    # Stage 2: Render turntable
    log(f"Rendering turntable ({num_angles} angles, {resolution}x{resolution})...")
    t0 = time.time()
    renders = render_turntable(glb_bytes, num_angles, resolution)
    elapsed = time.time() - t0
    log(f"Rendered {len(renders)} views in {elapsed:.1f}s")

    # Save renders
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    manifest_frames = []
    for angle, png_bytes in renders:
        filename = f"turn_{int(angle):03d}.png"
        (frames_dir / filename).write_bytes(png_bytes)
        manifest_frames.append({
            "angle": angle,
            "path": str(frames_dir / filename),
            "size": resolution,
        })
        log(f"  {filename} ({angle:.0f}°)")

    # Save manifest
    manifest = {
        "source_image": str(image_path),
        "glb_file": str(output_dir / "model.glb"),
        "num_angles": num_angles,
        "resolution": resolution,
        "frames": manifest_frames,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Copy character image for reference
    import shutil
    shutil.copy2(image_path, output_dir / "character.png")

    log(f"Output: {output_dir}")
    log(f"GLB: {output_dir / 'model.glb'}")
    log(f"Frames: {frames_dir}")
    return output_dir


def export_trellis_godot(
    trellis_dir: str | Path,
    character_name: str,
    project_dir: str | Path,
):
    """Export TRELLIS turntable renders to Godot project structure."""
    import shutil

    trellis_dir = Path(trellis_dir)
    project_dir = Path(project_dir)

    frames_dir = trellis_dir / "frames"
    manifest_path = trellis_dir / "manifest.json"
    glb_path = trellis_dir / "model.glb"

    if not frames_dir.exists():
        log(f"ERROR: No frames in {trellis_dir}")
        sys.exit(1)

    # Create Godot directory
    char_dir = project_dir / "characters_3d" / character_name
    char_dir.mkdir(parents=True, exist_ok=True)

    # Copy GLB
    if glb_path.exists():
        shutil.copy2(glb_path, char_dir / f"{character_name}.glb")

    # Copy frames
    godot_frames = char_dir / "turntable"
    godot_frames.mkdir(exist_ok=True)
    for frame in sorted(frames_dir.glob("*.png")):
        shutil.copy2(frame, godot_frames / frame.name)

    # Copy manifest
    if manifest_path.exists():
        shutil.copy2(manifest_path, char_dir / "manifest.json")

    # Generate scene for 3D character viewer
    scene = f"""[gd_scene load_steps=3 format=3]

[ext_resource type="PackedScene" path="res://characters_3d/{character_name}/{character_name}.glb" id="1_model"]
[ext_resource type="Script" path="res://characters_3d/{character_name}/turntable.gd" id="2_script"]

[node name="{character_name.capitalize()}3D" type="Node3D"]

[node name="Model" parent="." instance=ExtResource("1_model")]

[node name="Camera3D" type="Camera3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1.5, 3)
current = true

[node name="Turntable" type="Node3D" parent="."]
script = ExtResource("2_script")
"""
    (char_dir / f"{character_name}.tscn").write_text(scene)

    # GDScript turntable controller
    gdscript = '''@tool
extends Node3D
## Turntable controller for 3D character preview.

@export var rotation_speed: float = 30.0  # degrees per second
@export var auto_rotate: bool = true

func _process(delta):
    if auto_rotate:
        rotate_y(deg_to_rad(rotation_speed * delta))

func stop():
    auto_rotate = false

func start():
    auto_rotate = true
'''
    (char_dir / "turntable.gd").write_text(gdscript)

    log(f"Exported to: {char_dir}")
    log(f"  GLB: {char_dir / f'{character_name}.glb'}")
    log(f"  Frames: {godot_frames}")
    log(f"  Scene: {char_dir / f'{character_name}.tscn'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TRELLIS 3D model pipeline")
    parser.add_argument("--image", required=True, help="Character image path")
    parser.add_argument("--glb", default=None, help="Existing GLB file (skip 3D generation)")
    parser.add_argument("--angles", type=int, default=8, help="Number of turntable angles")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--no-ray", action="store_true", help="Skip Ray, use subprocess")
    parser.add_argument("--ray-url", default="http://localhost:8000", help="Ray gateway URL")
    parser.add_argument("--export-godot", default=None, help="Export to Godot project dir")
    parser.add_argument("--name", default="character", help="Character name for Godot export")
    args = parser.parse_args()

    output_dir = run_trellis_pipeline(
        image_path=args.image,
        glb_path=args.glb,
        num_angles=args.angles,
        resolution=args.resolution,
        output_dir=args.output,
        use_ray=not args.no_ray,
        ray_url=args.ray_url,
    )

    if args.export_godot:
        export_trellis_godot(output_dir, args.name, args.export_godot)


if __name__ == "__main__":
    main()
