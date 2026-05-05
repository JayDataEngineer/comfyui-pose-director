"""Sprites Static stage — VNCCS Step 4 SpriteCreator for static sprite sheet."""

from __future__ import annotations

from build_v2.context import DEPLOYMENT_CHECKPOINT, DEPLOYMENT_RMBG_MODEL, VNCCS_CHAR_NAME
from build_v2.stages.base import Stage, StageResult

DIRECTION_ROTATIONS = {
    "front": 0.0,
    "right": 90.0,
    "back": 180.0,
    "left": 270.0,
}


class SpritesStaticStage(Stage):
    """Generate static sprite sheet from character via VNCCS SpriteCreator.

    Body reference strategy (packages.sprites.body_reference):
      - "anny" (default): Uses the existing VNCCS Step 4 workflow with Anny body mesh.
      - "trellis": Uses TRELLIS-generated 3D model for pose reference.
        Auto-falls back to "anny" if TRELLIS stage did not complete.

    Uses VNCCS Step 4 API workflow. Takes the VNCCS character name
    and produces a full sprite sheet (all directions in one image).
    """

    name = "sprites_static"
    tier_required = "t1"
    depends_on = ["sheet"]

    _SPRITE_NODE = "1"
    _SAVE_NODE = "5"

    def is_complete(self, state: dict) -> bool:
        static_dir = self.ctx.sprites_dir / "static"
        return static_dir.exists() and bool(list(static_dir.glob("*.png")))

    # ── Strategy resolution ────────────────────────────────────────────────

    def _resolve_body_reference(self) -> str:
        """Read body_reference strategy from YAML, default to anny."""
        return self.ctx.pkg("sprites").get("body_reference", "anny")

    # ── Run dispatch ───────────────────────────────────────────────────────

    def run(self, state: dict) -> StageResult:
        strategy = self._resolve_body_reference()

        # Auto-fallback: trellis → anny if trellis stage didn't complete
        if strategy == "trellis":
            trellis_state = state.get("stages", {}).get("trellis", {})
            if trellis_state.get("status") != "complete":
                print("  [WARN] TRELLIS not complete, falling back to Anny body mesh")
                strategy = "anny"

        if strategy == "trellis":
            return self._run_with_trellis(state)
        else:
            return self._run_with_anny(state)

    # ── Anny strategy (current behavior) ───────────────────────────────────

    def _run_with_anny(self, state: dict) -> StageResult:
        ctx = self.ctx
        outfit = ctx.pkg("sprites").get("outfit", "default")

        static_dir = ctx.sprites_dir / "static"
        static_dir.mkdir(parents=True, exist_ok=True)

        patches = {
            self._SPRITE_NODE: {
                "character": VNCCS_CHAR_NAME,
            },
            # filename_prefix linked to SpriteGenerator output (is_list=True → list)
            # causes SaveImage to crash with "expected str, got list"
            self._SAVE_NODE: {
                "filename_prefix": f"PoseDirector/{ctx.character_name}/sprites_static",
            },
        }

        try:
            images = self.executor.run_api_workflow(
                workflow_path=ctx.api_workflows_dir / "vnccs_step4_sprites.json",
                patches=patches,
                output_node_id=self._SAVE_NODE,
                timeout=1800,  # Can be slow on first run
                checkpoint_override=DEPLOYMENT_CHECKPOINT,
                rmbg_override=DEPLOYMENT_RMBG_MODEL,
            )
        except Exception as e:
            return StageResult(status="failed", error=f"Sprite generation failed: {e}")

        if not images:
            return StageResult(status="failed", error="Sprites produced no output")

        artifacts = self._save_artifacts(images, static_dir, f"sprite_{outfit}")

        return StageResult(
            status="complete",
            artifacts=artifacts,
            meta={"outfit": outfit, "body_reference": "anny", "source": "vnccs_step4_sprites"},
        )

    # ── TRELLIS strategy ────────────────────────────────────────────────

    def _run_with_trellis(self, state: dict) -> StageResult:
        """Render Trellis GLB from multiple angles, feed each to VNCCS sprite pipeline.

        Path: Trellis GLB → load_glb → render_mesh(rotation) → upload → VNCCS edit
        For each direction angle, renders the character's 3D model and feeds it
        as image1 to VNCCS, replacing the generic Anny body mesh with the
        character-specific geometry.

        The workflow (05_sprite_bodymesh_edit.json) has BodyMeshRenderer wired
        to VNCCS image1. We bypass this by loading the workflow, replacing
        the BodyMeshRenderer node with a second LoadImage node that loads the
        Trellis render, then submitting the modified workflow.
        """
        ctx = self.ctx
        import copy
        import io
        from PIL import Image
        from utils.mesh_renderer import render_glb
        from pipeline.comfyui_client import upload_image
        from pipeline.ui_to_api import load_ui_workflow, convert_ui_to_api
        from pipeline.comfyui_client import submit_workflow, get_output_images

        # Find the Trellis GLB from the trellis stage
        trellis_artifacts = state.get("stages", {}).get("trellis", {}).get("artifacts", [])
        glb_path = None
        for art in trellis_artifacts:
            if art.endswith(".glb"):
                glb_path = ctx.output_dir / art
                break
        if not glb_path or not glb_path.exists():
            return StageResult(status="failed", error="Trellis GLB not found")

        static_dir = ctx.sprites_dir / "static"
        static_dir.mkdir(parents=True, exist_ok=True)

        sprites = ctx.pkg("sprites")
        directions = sprites.get("directions", ["front"])
        outfit = sprites.get("outfit", "default")
        width = sprites.get("width", 1024)
        height = sprites.get("height", 1024)

        # Upload character image once (image2 in VNCCS)
        comfyui_char_name = self.executor.upload_character(
            ctx.approved_image, f"{ctx.character_name}_char"
        )

        all_artifacts = []

        for direction in directions:
            rotation_y = DIRECTION_ROTATIONS.get(direction, 0.0)

            # Render Trellis GLB from this angle
            mesh_image = render_glb(
                glb_path,
                width=width,
                height=height,
                model_rotation_y=rotation_y,
            )

            # Upload rendered mesh image to ComfyUI
            img = Image.fromarray(mesh_image)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            mesh_upload_name = f"{ctx.character_name}_trellis_{direction}"
            upload_image(buf.getvalue(), f"{mesh_upload_name}.png", "PoseDirector", base_url=ctx.comfyui_url)
            comfyui_mesh_name = f"PoseDirector/{mesh_upload_name}.png"

            # Load workflow and replace BodyMeshRenderer (node 2) with LoadImage
            wf = load_ui_workflow(str(ctx.workflows_dir / "05_sprite_bodymesh_edit.json"))
            body_node = next(n for n in wf["nodes"] if n["id"] == 2)

            # Replace BodyMeshRenderer with LoadImage for the Trellis render
            body_node["type"] = "LoadImage"
            body_node["widgets_values"] = [comfyui_mesh_name, "image"]
            # Remove old widget keys that don't apply to LoadImage
            body_node.pop("properties", None)

            # Set the character image on the original LoadImage (node 1)
            for n in wf["nodes"]:
                if n["type"] == "LoadImage" and n["id"] == 1:
                    n["widgets_values"] = [comfyui_char_name, "image"]

            # Set seed on KSampler
            for n in wf["nodes"]:
                if n["type"] == "KSampler":
                    if not n.get("widgets_values"):
                        n["widgets_values"] = [ctx.seed, "randomize", 20, 7.0, "euler", "simple", 1.0]
                    else:
                        n["widgets_values"][0] = ctx.seed

            # Convert and submit
            api = convert_ui_to_api(wf)
            try:
                result = submit_workflow(api, timeout=300, base_url=ctx.comfyui_url)
                save_node = next(n for n in wf["nodes"] if n["type"] == "SaveImage")
                images = get_output_images(result["outputs"], save_node["id"], base_url=ctx.comfyui_url)
            except Exception as e:
                print(f"  [WARN] Trellis sprite render failed for {direction}: {e}")
                continue

            if images:
                artifact = self._save_artifacts(images, static_dir, f"trellis_{outfit}_{direction}")
                all_artifacts.extend(artifact)

        return StageResult(
            status="complete" if all_artifacts else "failed",
            artifacts=all_artifacts,
            meta={"body_reference": "trellis", "directions": directions, "glb": str(glb_path)},
        )
