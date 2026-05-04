"""Godot export stage and placeholder base class.

GodotStage is fully implemented. Other stages that were previously
placeholders have been moved to their own modules:
- state.py (StateStage)
- trellis.py (TrellisStage)
- video.py (VideoStage)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from build_v2.stages.base import Stage, StageResult


class _Placeholder(Stage):
    """Stage that reports skipped without error."""
    reason: str = "not yet implemented"

    def is_complete(self, state: dict) -> bool:
        return False

    def run(self, state: dict) -> StageResult:
        return StageResult(status="skipped", error=self.reason)


class GodotStage(Stage):
    """Export sprites to Godot 4.x project structure.

    Builds a manifest.json from sprite files on disk, then calls
    scripts/export_godot.py to create Godot sprite directories,
    SpriteFrames .tres resources, and CharacterSprite3D GDScript.
    Only runs when sprites exist and a Godot project path is configured.
    """

    name = "godot"
    tier_required = "t1"
    depends_on = ["sprites_static"]

    def is_complete(self, state: dict) -> bool:
        godot_dir = self.ctx.output_dir / "godot"
        return godot_dir.exists() and bool(list(godot_dir.rglob("*.tres")))

    def _build_manifest(self) -> Path | None:
        """Build manifest.json from sprite files on disk."""
        import json as json_mod

        ctx = self.ctx
        sprites_pkg = ctx.character_data.get("packages", {}).get("sprites", {})
        if not sprites_pkg:
            sprites_pkg = ctx.character_data.get("sprites", {})
        outfit = sprites_pkg.get("outfit", "default")
        directions = sprites_pkg.get("directions", ["front"])

        frames = []

        # Static sprites
        static_dir = ctx.sprites_dir / "static"
        if static_dir.exists():
            for sprite_path in sorted(static_dir.glob("*.png")):
                for direction in directions:
                    frames.append({
                        "animation": "idle",
                        "direction": direction,
                        "path": str(sprite_path),
                    })

        # Animated sprites (if they exist)
        for anim in sprites_pkg.get("animated", []):
            anim_name = anim["name"]
            for direction in directions:
                frame_dir = ctx.sprites_dir / outfit / "normal" / anim_name / direction
                if frame_dir.exists():
                    for frame_path in sorted(frame_dir.glob("frame_*.png")):
                        frames.append({
                            "animation": anim_name,
                            "direction": direction,
                            "path": str(frame_path),
                        })

        if not frames:
            return None

        manifest_path = ctx.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json_mod.dump({"frames": frames, "character": ctx.character_name}, f, indent=2)
        return manifest_path

    def run(self, state: dict) -> StageResult:
        import json as json_mod

        ctx = self.ctx
        godot_pkg = ctx.character_data.get("packages", {}).get("godot", {})
        godot_project = godot_pkg.get("project_path", "")

        if not godot_project:
            return StageResult(status="skipped", error="No Godot project path configured")

        sprites_dir = ctx.sprites_dir
        if not sprites_dir.exists():
            return StageResult(status="skipped", error="No sprites directory found")

        export_script = ctx.project_root / "scripts" / "export_godot.py"
        if not export_script.exists():
            return StageResult(status="skipped", error="export_godot.py not found")

        # Build manifest from sprite files
        manifest_path = self._build_manifest()
        if not manifest_path:
            return StageResult(status="skipped", error="No sprite frames found for manifest")

        godot_dir = ctx.output_dir / "godot"
        godot_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                [
                    sys.executable, str(export_script),
                    str(manifest_path),
                    "--name", ctx.character_name,
                    "--project", godot_project,
                ],
                capture_output=True, text=True, timeout=120,
                cwd=str(ctx.project_root),
            )
            if result.returncode != 0:
                return StageResult(
                    status="failed",
                    error=f"Godot export failed: {result.stderr[:500]}",
                )
        except Exception as e:
            return StageResult(status="failed", error=f"Godot export error: {e}")

        # Copy godot output to our output dir
        godot_src = Path(godot_project) / "sprites" / ctx.character_name
        if godot_src.exists():
            import shutil
            for f in godot_src.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(godot_src)
                    target = godot_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, target)

        artifacts = [str(p.relative_to(ctx.output_dir)) for p in godot_dir.rglob("*") if p.is_file()]
        return StageResult(
            status="complete" if artifacts else "skipped",
            artifacts=artifacts,
            meta={"project": godot_project},
        )
