"""Placeholder stages for not-yet-implemented pipeline steps.

These stages exist so the DAG resolves and the build system runs without error.
They report "skipped" with an explanation rather than failing.
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


class TrellisStage(_Placeholder):
    name = "trellis"
    tier_required = "t2"
    depends_on = ["generate"]
    reason = "TRELLIS 3D model generation not yet implemented"


class StateStage(_Placeholder):
    """Re-edit character with state condition (bruised, wet, etc).

    Uses VNCCS Step 1.1 with modified instruction to apply visual state changes.
    The state package in YAML defines name + description of the condition.
    """
    name = "state"
    tier_required = "t1"
    depends_on = ["sheet"]
    reason = "State condition re-edit not yet implemented — needs VNCCS Step 1.1 with state prompt"


class VideoStage(_Placeholder):
    """LTX Video 2.3 assembly from sprite frames into animation clips.

    Uses 04_ltx_video_assembly.json workflow to combine sprite frames
    into smooth video animations with keyframe guides.
    """
    name = "video"
    tier_required = "t1.5"
    depends_on = ["sprites_static"]
    reason = "LTX Video assembly not yet implemented"


class GodotStage(Stage):
    """Export sprites to Godot 4.x project structure.

    Uses scripts/export_godot.py to create organized sprite directories,
    SpriteFrames .tres resources, and CharacterSprite3D GDScript controller.
    Only runs when sprites exist and a Godot project path is configured.
    """

    name = "godot"
    tier_required = "t1"
    depends_on = ["sprites_static"]

    def is_complete(self, state: dict) -> bool:
        godot_dir = self.ctx.output_dir / "godot"
        return godot_dir.exists() and bool(list(godot_dir.rglob("*.tres")))

    def run(self, state: dict) -> StageResult:
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

        godot_dir = ctx.output_dir / "godot"
        godot_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                [
                    sys.executable, str(export_script),
                    str(ctx.output_dir),
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

        artifacts = [str(p.relative_to(ctx.output_dir)) for p in godot_dir.rglob("*") if p.is_file()]
        return StageResult(
            status="complete" if artifacts else "skipped",
            artifacts=artifacts,
            meta={"project": godot_project},
        )
