"""LoRA stage — VNCCS Step 5 LoRA Dataset Generator."""

from __future__ import annotations

from build_v2.stages.base import Stage, StageResult


class LoraStage(Stage):
    """Generate LoRA training dataset from character sprites via VNCCS.

    Uses VNCCS Step 5 API workflow. Takes character name,
    generates a zipped dataset for LoRA training.
    """

    name = "lora"
    tier_required = "t3"
    depends_on = ["sprites_static", "sprites_animated"]

    _SAVE_NODE = "2"

    def is_complete(self, state: dict) -> bool:
        lora_dir = self.ctx.output_dir / "lora"
        return lora_dir.exists() and bool(list(lora_dir.glob("*")))

    def run(self, state: dict) -> StageResult:
        ctx = self.ctx

        sprites_dir = ctx.sprites_dir
        static = sprites_dir / "static"
        has_static = static.exists() and bool(list(static.glob("*.png")))

        if not has_static:
            return StageResult(status="skipped", error="No static sprites available for LoRA dataset")

        lora_dir = ctx.output_dir / "lora"
        lora_dir.mkdir(parents=True, exist_ok=True)

        patches = {
            self._SAVE_NODE: {
            },
        }

        try:
            result = self.executor.run_api_workflow(
                workflow_path=ctx.api_workflows_dir / "vnccs_step5_lora_dataset.json",
                patches=patches,
                output_node_id=self._SAVE_NODE,
                timeout=600,
            )
        except Exception as e:
            return StageResult(status="failed", error=f"LoRA generation failed: {e}")

        if result:
            for i, content in enumerate(result):
                out_path = lora_dir / f"dataset_{i:03d}"
                out_path.mkdir(parents=True, exist_ok=True)

        return StageResult(
            status="complete",
            artifacts=[str(p.relative_to(ctx.output_dir)) for p in lora_dir.rglob("*") if p.is_file()],
            meta={"source": "vnccs_step5_lora"},
        )
