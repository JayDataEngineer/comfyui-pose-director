"""Generate stage — Z-Image character generation or image import."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from build_v2.stages.base import Stage, StageResult
from build_v2.state import file_hash


class GenerateStage(Stage):
    """Generate character image via Z-Image or import from file."""

    name = "generate"
    tier_required = "t0"
    depends_on: list[str] = []

    def is_complete(self, state: dict) -> bool:
        return self.ctx.approved_image.exists()

    def run(self, state: dict) -> StageResult:
        ctx = self.ctx
        gen = ctx.character_data.get("generation", {})
        source = gen.get("source", "generate")

        if source == "import":
            return self._import_image(gen)
        else:
            return self._generate_image(gen)

    def _import_image(self, gen: dict) -> StageResult:
        """Import an existing image as the character reference."""
        ctx = self.ctx
        img_path = Path(gen.get("image", ""))
        if not img_path.is_absolute():
            yaml_dir = Path(ctx.yaml_path).parent
            img_path = yaml_dir / img_path

        if not img_path.exists():
            return StageResult(status="failed", error=f"Import image not found: {img_path}")

        ctx.design_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, ctx.approved_image)

        return StageResult(
            status="complete",
            artifacts=["design/approved.png"],
            meta={
                "source": "import",
                "input_image": str(img_path),
                "input_hash": file_hash(img_path),
            },
        )

    def _generate_image(self, gen: dict) -> StageResult:
        """Generate character image via Z-Image workflow."""
        ctx = self.ctx
        quality = gen.get("quality", "turbo")
        prompt = gen.get("prompt", "")
        seed = ctx.seed

        ctx.design_dir.mkdir(parents=True, exist_ok=True)

        patches = {
            "CLIPTextEncode": {"text": prompt},
            "KSampler": {"seed": seed},
            "SaveImage": {"filename_prefix": f"PoseDirector/{ctx.character_name}/character"},
        }
        if quality == "base":
            patches["UNETLoader"] = {"unet_name": "z_image_bf16.safetensors"}
            patches["KSampler"] = {
                "seed": seed, "steps": 50, "cfg": 4.0,
                "sampler_name": "euler", "scheduler": "normal",
            }

        images = self.executor.run_ui_workflow(
            workflow_path=ctx.workflows_dir / "00_z_image_character.json",
            patches=patches,
            output_node_type="SaveImage",
            timeout=900,  # Remote ComfyUI may need time for model loading
        )

        if not images:
            return StageResult(status="failed", error="Z-Image produced no output")

        ctx.approved_image.write_bytes(images[0])

        return StageResult(
            status="complete",
            artifacts=["design/approved.png"],
            meta={
                "source": "generate",
                "quality": quality,
                "prompt": prompt,
                "seed": seed,
            },
        )
