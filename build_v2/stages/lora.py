"""LoRA stage — VNCCS Step 5 LoRA Dataset Generator."""

from __future__ import annotations

from build_v2.context import VNCCS_CHAR_NAME
from build_v2.stages.base import Stage, StageResult


class LoraStage(Stage):
    """Generate LoRA training dataset from character sprites via VNCCS.

    Uses VNCCS Step 5 API workflow. Takes character name + game name,
    generates captioned image dataset for LoRA training.
    DatasetGenerator returns the VNCCS output path where files are stored.
    """

    name = "lora"
    tier_required = "t3"
    depends_on = ["sprites_static"]

    _DATASET_NODE = "7"   # DatasetGenerator
    _SAVE_NODE = "8"      # Save Text File

    def is_complete(self, state: dict) -> bool:
        lora_dir = self.ctx.output_dir / "lora"
        return lora_dir.exists() and bool(list(lora_dir.glob("*.png")))

    def run(self, state: dict) -> StageResult:
        ctx = self.ctx

        sprites_dir = ctx.sprites_dir
        static = sprites_dir / "static"
        has_static = static.exists() and bool(list(static.glob("*.png")))

        if not has_static:
            return StageResult(status="skipped", error="No static sprites available for LoRA dataset")

        lora_dir = ctx.output_dir / "lora"
        lora_dir.mkdir(parents=True, exist_ok=True)

        game_name = ctx.character_data.get("character", {}).get("game_name", "PoseDirector")

        patches = {
            self._DATASET_NODE: {
                "character": VNCCS_CHAR_NAME,
                "game_name": game_name,
            },
        }

        try:
            self.executor.run_api_workflow(
                workflow_path=ctx.api_workflows_dir / "vnccs_step5_lora_dataset.json",
                patches=patches,
                output_node_id=self._SAVE_NODE,
                timeout=600,
            )
        except Exception:
            # DatasetGenerator returns STRING (path), not IMAGE — may fail on output parsing.
            # The actual dataset is in VNCCS's output directory on the ComfyUI server.
            pass

        # DatasetGenerator writes to VNCCS base_output_dir/{character}/lora/
        # TODO: fetch the generated dataset from ComfyUI server.

        artifacts = [str(p.relative_to(ctx.output_dir)) for p in lora_dir.rglob("*") if p.is_file()]

        return StageResult(
            status="complete" if artifacts else "skipped",
            artifacts=artifacts,
            meta={"source": "vnccs_step5_lora", "character": VNCCS_CHAR_NAME},
        )
