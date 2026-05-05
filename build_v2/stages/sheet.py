"""Sheet stage — VNCCS Step 1.1 Clone to create character sheet from approved image."""

from __future__ import annotations

from build_v2.context import DEPLOYMENT_CHECKPOINT, VNCCS_CHAR_NAME
from build_v2.stages.base import Stage, StageResult


class SheetStage(Stage):
    """Generate VNCCS character sheet from the approved character image.

    Face refinement strategy (packages.sheet.face_refinement):
      - "skip" (default): No post-processing on the sheet.
      - "facedetailer": Run impact pack FaceDetailer with SDXL checkpoint.
        Requires a face bbox model in models/ultralytics/ (e.g. face_yolov8n.pt).

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

    # FaceDetailer model names
    _BBOX_MODEL = "face_yolov8n.pt"

    def is_complete(self, state: dict) -> bool:
        sheet_dir = self.ctx.sheet_dir
        return sheet_dir.exists() and bool(list(sheet_dir.glob("*.png")))

    # ── Strategy resolution ────────────────────────────────────────────────

    def _resolve_face_refinement(self) -> str:
        return self.ctx.pkg("sheet").get("face_refinement", "skip")

    # ── Run dispatch ───────────────────────────────────────────────────────

    def run(self, state: dict) -> StageResult:
        result = self._generate_sheet()

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
                timeout=1800,
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

    def _check_bbox_model_available(self) -> bool:
        """Check if the face bbox detection model exists."""
        from pathlib import Path
        model_dirs = [
            Path("models/ultralytics"),
            Path("models/yolo"),
            Path("models/bbox"),
        ]
        for d in model_dirs:
            if (d / self._BBOX_MODEL).exists():
                return True
        return False

    def _refine_faces(self, sheet_result: StageResult) -> StageResult:
        """Run impact pack FaceDetailer with SDXL on generated sheet images.

        Constructs an API-format workflow:
            LoadImage → CheckpointLoaderSimple → CLIPTextEncode (pos/neg)
                      → FaceDetailer → SaveImage

        Requires face_yolov8n.pt in models/ultralytics/.
        """
        from pipeline.comfyui_client import submit_workflow, get_output_images, upload_image

        ctx = self.ctx

        if not self._check_bbox_model_available():
            return StageResult(
                status="failed",
                error=(
                    f"FaceDetailer requires {self._BBOX_MODEL} in models/ultralytics/. "
                    "Download it with: "
                    "wget https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt "
                    "-O models/ultralytics/face_yolov8n.pt"
                ),
            )

        # Find the sheet image to refine
        sheet_images = sorted(ctx.sheet_dir.glob("sheet_*.png"))
        if not sheet_images:
            sheet_result.meta["face_refinement"] = "skipped_no_images"
            return sheet_result

        refined_dir = ctx.sheet_dir / "refined"
        refined_dir.mkdir(parents=True, exist_ok=True)

        all_artifacts = list(sheet_result.artifacts)

        for sheet_path in sheet_images:
            # Upload sheet image
            upload_name = f"{ctx.character_name}_sheet_{sheet_path.stem}"
            upload_image(sheet_path.read_bytes(), f"{upload_name}.png", "PoseDirector", base_url=ctx.comfyui_url)
            comfyui_img = f"PoseDirector/{upload_name}.png"

            # Build FaceDetailer API workflow
            wf = {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": comfyui_img},
                },
                "2": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": DEPLOYMENT_CHECKPOINT},
                },
                "3": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "text": "detailed face, high quality, sharp focus, clear eyes",
                        "clip": ["2", 1],
                    },
                },
                "4": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "text": "blurry, low quality, deformed, artifacts",
                        "clip": ["2", 1],
                    },
                },
                "5": {
                    "class_type": "FaceDetailer",
                    "inputs": {
                        "image": ["1", 0],
                        "model": ["2", 0],
                        "clip": ["2", 1],
                        "vae": ["2", 2],
                        "guide_size": 512,
                        "guide_size_for": True,
                        "max_size": 1024,
                        "seed": ctx.seed,
                        "steps": 20,
                        "cfg": 8.0,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "positive": ["3", 0],
                        "negative": ["4", 0],
                        "denoise": 0.5,
                        "feather": 5,
                        "noise_mask": True,
                        "force_inpaint": True,
                        "bbox_threshold": 0.5,
                        "bbox_dilation": 10,
                        "bbox_crop_factor": 3.0,
                        "sam_detection_hint": "center-1",
                        "sam_dilation": 0,
                        "sam_threshold": 0.93,
                        "sam_bbox_expansion": 0,
                        "sam_mask_hint_threshold": 0.7,
                    },
                },
                "6": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["5", 0],
                        "filename_prefix": f"PoseDirector/{ctx.character_name}/sheet_refined",
                    },
                },
            }

            try:
                result = submit_workflow(wf, timeout=600, base_url=ctx.comfyui_url)
                images = get_output_images(result["outputs"], "6", base_url=ctx.comfyui_url)
            except Exception as e:
                print(f"  [WARN] FaceDetailer failed for {sheet_path.name}: {e}")
                continue

            if images:
                refined_artifacts = self._save_artifacts(images, refined_dir, sheet_path.stem)
                all_artifacts.extend(refined_artifacts)

        has_refined = (refined_dir / "refined").with_suffix(".png").exists() or bool(list(refined_dir.glob("*.png")))

        sheet_result.artifacts = all_artifacts
        sheet_result.meta["face_refinement"] = "complete" if has_refined else "failed"
        return sheet_result
