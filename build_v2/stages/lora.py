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
        except Exception as e:
            return StageResult(status="failed", error=f"LoRA dataset generation failed: {e}")

        # Fetch the generated dataset from ComfyUI output directory
        self._fetch_dataset(ctx, lora_dir)

        artifacts = [str(p.relative_to(ctx.output_dir)) for p in lora_dir.rglob("*") if p.is_file()]

        return StageResult(
            status="complete" if artifacts else "failed",
            artifacts=artifacts,
            meta={"source": "vnccs_step5_lora", "character": VNCCS_CHAR_NAME},
        )

    def _fetch_dataset(self, ctx, lora_dir: "Path") -> None:
        """Download generated LoRA dataset files from ComfyUI output.

        VNCCS DatasetGenerator saves files to ComfyUI's output directory.
        We download them to our local output dir via the /view API.
        """
        import requests
        from pathlib import Path

        url = ctx.comfyui_url.rstrip("/")
        try:
            # List files in ComfyUI output directory for this character
            resp = requests.get(
                f"{url}/view",
                params={"type": "output"},
                timeout=30,
            )
            # The /view endpoint serves individual files, not listings.
            # VNCCS DatasetGenerator typically outputs to a known path pattern.
            # Try to fetch from the standard VNCCS output location.
            resp = requests.get(
                f"{url}/api/list",
                params={"type": "output"},
                timeout=30,
            )
            if resp.status_code != 200:
                print("  [WARN] Could not list ComfyUI output directory")
                return

            listing = resp.json()
            # Find files matching the character's LoRA dataset pattern
            prefix = f"PoseDirector/{ctx.character_name}"
            for item in listing.get("files", []):
                # VNCCS outputs are typically under the output root
                filename = item.get("name", "") if isinstance(item, dict) else str(item)
                if not filename:
                    continue

                # Download matching files
                for suffix in (".png", ".jpg", ".txt", ".caption"):
                    if filename.endswith(suffix) and ctx.character_name in filename:
                        dl = requests.get(
                            f"{url}/view",
                            params={"filename": filename, "type": "output"},
                            timeout=30,
                        )
                        if dl.status_code == 200:
                            out_path = lora_dir / filename.rsplit("/", 1)[-1]
                            out_path.write_bytes(dl.content)

        except Exception as e:
            print(f"  [WARN] LoRA dataset fetch failed: {e}")
