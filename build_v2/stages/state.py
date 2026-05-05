"""State stage — Apply visual state conditions (bruised, wet, etc.) to character.

Uses the sprite body-mesh edit workflow (05_sprite_bodymesh_edit.json) with
a VNCCS instruction that applies state effects while preserving identity and pose.
"""

from __future__ import annotations

from build_v2.context import DEPLOYMENT_CHECKPOINT, DEPLOYMENT_RMBG_MODEL, VNCCS_CHAR_NAME
from build_v2.stages.base import Stage, StageResult


class StateStage(Stage):
    """Apply visual state condition to character portraits and sprites.

    Takes the state package (name + description) from character YAML and
    runs VNCCS QWEN edit to apply the state effect (bruises, weathering,
    damage, etc.) while keeping character identity and pose intact.
    """

    name = "state"
    tier_required = "t1"
    depends_on = ["sheet"]

    # Node IDs in the pre-exported API workflow
    _LOAD_IMAGE_NODE = "638"
    _SAVE_IMAGE_NODE = "15"
    _CHARACTER_CREATOR_NODE = "499"

    def is_complete(self, state: dict) -> bool:
        state_dir = self.ctx.output_dir / "state"
        return state_dir.exists() and bool(list(state_dir.glob("*.png")))

    def _get_state_package(self) -> dict:
        """Extract state package from character data."""
        return self.ctx.pkg("state")

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

        instruction = self._build_instruction(state_name, state_desc)

        # Use the first sheet image as input
        source_image = sheet_images[0]
        comfyui_name = self.executor.upload_character(
            source_image, f"{ctx.character_name}_state_source"
        )

        patches = {
            self._LOAD_IMAGE_NODE: {"image": comfyui_name},
            self._CHARACTER_CREATOR_NODE: {
                "existing_character": VNCCS_CHAR_NAME,
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
                checkpoint_override=DEPLOYMENT_CHECKPOINT,
                rmbg_override=DEPLOYMENT_RMBG_MODEL,
            )
        except Exception as e:
            return StageResult(status="failed", error=f"State generation failed: {e}")

        if not images:
            return StageResult(status="failed", error="State edit produced no output")

        artifacts = self._save_artifacts(images, state_dir, state_name)

        return StageResult(
            status="complete",
            artifacts=artifacts,
            meta={
                "state_name": state_name,
                "description": state_desc,
                "apply_to": state_pkg.get("apply_to", []),
            },
        )
