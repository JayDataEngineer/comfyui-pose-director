"""Outfit stage — VNCCS Step 2 ClothesGenerator for outfit variants."""

from __future__ import annotations

from build_v2.stages.base import Stage, StageResult


class OutfitStage(Stage):
    """Generate outfit variant from character sheet via VNCCS ClothesGenerator.

    Uses VNCCS Step 2 API workflow. Takes character name + outfit description,
    generates clothing variant sprites.
    """

    name = "outfit"
    tier_required = "t2"
    depends_on = ["sheet"]

    # Node IDs in the pre-exported API workflow
    _ASSET_NODE = "579"  # CharacterAssetSelectorQWEN
    _SAVE_NODE = "15"    # SaveImage

    # Remote deployment overrides
    _CHECKPOINT = "sd_xl_base_1.0.safetensors"
    _RMBG_MODEL = "BEN2"

    def is_complete(self, state: dict) -> bool:
        outfit_dir = self.ctx.output_dir / "outfits"
        return outfit_dir.exists() and bool(list(outfit_dir.glob("*.png")))

    def run(self, state: dict) -> StageResult:
        ctx = self.ctx
        outfit_pkg = ctx.character_data.get("packages", {}).get("outfit", {})
        outfit_name = outfit_pkg.get("name", "")
        outfit_desc = outfit_pkg.get("description", "")

        if not outfit_name:
            return StageResult(status="skipped", error="No outfit declared")

        outfit_dir = ctx.output_dir / "outfits"
        outfit_dir.mkdir(parents=True, exist_ok=True)

        # VNCCS bug: character registered as "None" (see sheet stage)
        vnccs_char_name = "None"

        patches = {
            self._ASSET_NODE: {
                "character": vnccs_char_name,
                "top": outfit_desc or outfit_name,
            },
            # filename_prefix linked to VNCCS_String output → list crash
            self._SAVE_NODE: {
                "filename_prefix": f"PoseDirector/{ctx.character_name}/outfit/{outfit_name}",
            },
        }

        try:
            images = self.executor.run_api_workflow(
                workflow_path=ctx.api_workflows_dir / "vnccs_step2_clothes.json",
                patches=patches,
                output_node_id=self._SAVE_NODE,
                timeout=1800,  # Can be slow — SeedVR2 upscale + multiple passes
                checkpoint_override=self._CHECKPOINT,
                rmbg_override=self._RMBG_MODEL,
            )
        except Exception as e:
            return StageResult(status="failed", error=f"Outfit generation failed: {e}")

        if not images:
            return StageResult(status="failed", error="Outfit produced no output")

        artifacts = []
        for i, img_bytes in enumerate(images):
            out_path = outfit_dir / f"{outfit_name}_{i:03d}.png"
            out_path.write_bytes(img_bytes)
            artifacts.append(str(out_path.relative_to(ctx.output_dir)))

        return StageResult(
            status="complete",
            artifacts=artifacts,
            meta={"outfit": outfit_name, "description": outfit_desc},
        )
