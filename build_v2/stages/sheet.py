"""Sheet stage — VNCCS Step 1.1 Clone to create character sheet from approved image."""

from __future__ import annotations

from pathlib import Path

from build_v2.stages.base import Stage, StageResult


class SheetStage(Stage):
    """Generate VNCCS character sheet from the approved character image.

    Uses VNCCS Step 1.1 (Clone Existing Character) API workflow.
    Uploads approved.png → runs VNCCS Clone → saves character sheet.
    """

    name = "sheet"
    tier_required = "t1"
    depends_on = ["generate"]

    # Node IDs in the pre-exported API workflow
    _LOAD_IMAGE_NODE = "638"
    _SAVE_IMAGE_NODE = "15"
    _CHARACTER_CREATOR_NODE = "499"

    # Remote deployment overrides (see build_v2/api_patcher.py for why)
    _CHECKPOINT = "sd_xl_base_1.0.safetensors"  # ILFlatMix fails mmap on Docker
    _RMBG_MODEL = "BEN2"  # RMBG-2.0 needs writeable cache, INSPYRENET needs extra pkg

    def is_complete(self, state: dict) -> bool:
        sheet_dir = self.ctx.sheet_dir
        return sheet_dir.exists() and bool(list(sheet_dir.glob("*.png")))

    def run(self, state: dict) -> StageResult:
        ctx = self.ctx

        if not ctx.approved_image.exists():
            return StageResult(status="failed", error="Approved image not found")

        sheet_dir = ctx.sheet_dir
        sheet_dir.mkdir(parents=True, exist_ok=True)

        comfyui_name = self.executor.upload_character(
            ctx.approved_image, f"{ctx.character_name}_source"
        )

        # VNCCS bug: CharacterCreator ignores new_character_name parameter.
        # It always uses existing_character as the character name.
        # With existing_character="None", character is registered as "None".
        # Then emotions/sprites stages must also use "None" as character name.
        vnccs_char_name = "None"

        patches = {
            self._LOAD_IMAGE_NODE: {"image": comfyui_name},
            self._CHARACTER_CREATOR_NODE: {
                "existing_character": vnccs_char_name,
                "new_character_name": ctx.character_name,
            },
            self._SAVE_IMAGE_NODE: {
                "filename_prefix": f"PoseDirector/{ctx.character_name}/sheet",
            },
        }

        try:
            images = self.executor.run_api_workflow(
                workflow_path=ctx.api_workflows_dir / "vnccs_step11_clone_existing.json",
                patches=patches,
                output_node_id=self._SAVE_IMAGE_NODE,
                timeout=1800,  # ~22 min on RTX 4090 (3 QWEN encoder passes)
                checkpoint_override=self._CHECKPOINT,
                rmbg_override=self._RMBG_MODEL,
            )
        except Exception as e:
            return StageResult(status="failed", error=f"Sheet generation failed: {e}")

        if not images:
            return StageResult(status="failed", error="Sheet produced no output")

        artifacts = []
        for i, img_bytes in enumerate(images):
            out_path = sheet_dir / f"sheet_{i:03d}.png"
            out_path.write_bytes(img_bytes)
            artifacts.append(str(out_path.relative_to(ctx.output_dir)))

        return StageResult(
            status="complete",
            artifacts=artifacts,
            meta={"source": "vnccs_step11_clone"},
        )
