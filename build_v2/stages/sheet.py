"""Sheet stage — VNCCS Step 1.1 Clone to create character sheet from approved image."""

from __future__ import annotations

from build_v2.context import VNCCS_CHAR_NAME
from build_v2.stages.base import Stage, StageResult


class SheetStage(Stage):
    """Generate VNCCS character sheet from the approved character image.

    Face refinement strategy (packages.sheet.face_refinement):
      - "skip" (default): No post-processing on the sheet.
      - "facedetailer": Run Z-Image FaceDetailer pass on the sheet after generation.

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

    def is_complete(self, state: dict) -> bool:
        sheet_dir = self.ctx.sheet_dir
        return sheet_dir.exists() and bool(list(sheet_dir.glob("*.png")))

    # ── Strategy resolution ────────────────────────────────────────────────

    def _resolve_face_refinement(self) -> str:
        """Read face_refinement strategy from YAML, default to skip."""
        return self.ctx.pkg("sheet").get("face_refinement", "skip")

    # ── Run dispatch ───────────────────────────────────────────────────────

    def run(self, state: dict) -> StageResult:
        result = self._generate_sheet()

        # Optional post-processing: face refinement
        if result.status == "complete" and self._resolve_face_refinement() == "facedetailer":
            result = self._refine_faces(result)

        return result

    # ── Sheet generation ───────────────────────────────────────────────────

    def _generate_sheet(self) -> StageResult:
        ctx = self.ctx

        if not ctx.approved_image.exists():
            return StageResult(status="failed", error="Approved image not found")

        ctx.sheet_dir.mkdir(parents=True, exist_ok=True)

        comfyui_name = self.executor.upload_character(
            ctx.approved_image, f"{ctx.character_name}_source"
        )

        patches = {
            self._LOAD_IMAGE_NODE: {"image": comfyui_name},
            self._CHARACTER_CREATOR_NODE: {
                "existing_character": VNCCS_CHAR_NAME,
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
            )
        except Exception as e:
            return StageResult(status="failed", error=f"Sheet generation failed: {e}")

        if not images:
            return StageResult(status="failed", error="Sheet produced no output")

        artifacts = self._save_artifacts(images, ctx.sheet_dir, "sheet")

        return StageResult(
            status="complete",
            artifacts=artifacts,
            meta={"source": "vnccs_step11_clone", "face_refinement": "skip"},
        )

    # ── Face refinement ────────────────────────────────────────────────────

    def _refine_faces(self, sheet_result: StageResult) -> StageResult:
        """Run Z-Image FaceDetailer pass on generated sheet images.

        TODO: Implement FaceDetailer workflow wiring. Needs a workflow JSON
        that loads the sheet image, runs FaceDetailer with Z-Image Turbo
        model/clip/vae, and saves the refined output.
        """
        print("  [WARN] FaceDetailer not yet implemented, skipping face refinement")
        sheet_result.meta["face_refinement"] = "skipped_not_implemented"
        return sheet_result
