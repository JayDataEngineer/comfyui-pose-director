"""State stage — Apply visual state conditions (bruised, wet, etc.) to character.

Uses the sprite body-mesh edit workflow (05_sprite_bodymesh_edit.json) with
a VNCCS instruction that applies state effects while preserving identity and pose.
"""

from __future__ import annotations

from pathlib import Path

from build_v2.stages.base import Stage, StageResult


class StateStage(Stage):
    """Apply visual state condition to character portraits and sprites.

    Takes the state package (name + description) from character YAML and
    runs VNCCS QWEN edit to apply the state effect (bruises, weathering,
    damage, etc.) while keeping character identity and pose intact.

    Uses the 05_sprite_bodymesh_edit.json UI workflow with a state-specific
    VNCCS instruction, similar to how the sprite pipeline works but without
    pose changes.
    """

    name = "state"
    tier_required = "t1"
    depends_on = ["sheet"]

    # Node IDs in the pre-exported API workflow
    _LOAD_IMAGE_NODE = "638"
    _SAVE_IMAGE_NODE = "15"
    _CHARACTER_CREATOR_NODE = "499"

    # Remote deployment overrides
    _CHECKPOINT = "sd_xl_base_1.0.safetensors"
    _RMBG_MODEL = "BEN2"

    def is_complete(self, state: dict) -> bool:
        state_dir = self.ctx.output_dir / "state"
        return state_dir.exists() and bool(list(state_dir.glob("*.png")))

    def _get_state_package(self) -> dict:
        """Extract state package from character data."""
        pkg = self.ctx.character_data.get("packages", {}).get("state", {})
        if not pkg:
            pkg = self.ctx.character_data.get("state", {})
        return pkg

    def _build_instruction(self, state_name: str, state_desc: str) -> str:
        """Build VNCCS instruction for state application."""
        return (
            f"Apply {state_name} condition to the character: {state_desc}. "
            "Keep the exact same character identity, facial features, pose, "
            "and composition. Only add the state effects described above. "
            "Maintain the same art style, color palette, and lighting."
        )

    def run(self, state: dict) -> StageResult:
        ctx = self.ctx
        state_pkg = self._get_state_package()
        state_name = state_pkg.get("name", "")
        state_desc = state_pkg.get("description", "")

        if not state_name:
            return StageResult(status="skipped", error="No state condition declared")

        # Need sheet as input
        sheet_dir = ctx.sheet_dir
        if not sheet_dir.exists():
            return StageResult(status="failed", error="Character sheet not found")

        sheet_images = sorted(sheet_dir.glob("*.png"))
        if not sheet_images:
            return StageResult(status="failed", error="No sheet images found")

        state_dir = ctx.output_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        instruction = self._build_instruction(state_name, state_desc)

        # Use the first sheet image as input
        source_image = sheet_images[0]
        comfyui_name = self.executor.upload_character(
            source_image, f"{ctx.character_name}_state_source"
        )

        # VNCCS bug: character registered as "None" (see sheet stage)
        vnccs_char_name = "None"

        patches = {
            self._LOAD_IMAGE_NODE: {"image": comfyui_name},
            self._CHARACTER_CREATOR_NODE: {
                "existing_character": vnccs_char_name,
                "new_character_name": ctx.character_name,
            },
            self._SAVE_IMAGE_NODE: {
                "filename_prefix": f"PoseDirector/{ctx.character_name}/state/{state_name}",
            },
        }

        try:
            images = self.executor.run_api_workflow(
                workflow_path=ctx.api_workflows_dir / "vnccs_step11_clone_existing.json",
                patches=patches,
                output_node_id=self._SAVE_IMAGE_NODE,
                timeout=1800,
                checkpoint_override=self._CHECKPOINT,
                rmbg_override=self._RMBG_MODEL,
            )
        except Exception as e:
            return StageResult(status="failed", error=f"State generation failed: {e}")

        if not images:
            return StageResult(status="failed", error="State edit produced no output")

        artifacts = []
        for i, img_bytes in enumerate(images):
            out_path = state_dir / f"{state_name}_{i:03d}.png"
            out_path.write_bytes(img_bytes)
            artifacts.append(str(out_path.relative_to(ctx.output_dir)))

        return StageResult(
            status="complete",
            artifacts=artifacts,
            meta={
                "state_name": state_name,
                "description": state_desc,
                "apply_to": state_pkg.get("apply_to", []),
            },
        )
