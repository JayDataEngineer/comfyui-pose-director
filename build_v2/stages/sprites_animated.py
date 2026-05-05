"""Sprites Animated stage — HY-Motion + Anny body mesh → VNCCS sprite frames."""

from __future__ import annotations

import json
from pathlib import Path

from build_v2.stages.base import Stage, StageResult

# Prompts matching the VNCCS Pose Studio reference pipeline
VNCCS_PROMPT = "Draw character from image2"
VNCCS_INSTRUCTION = (
    "Match the body pose shown in Picture 1 (3D body mesh). "
    "Picture 2 is the character to draw. Picture 3 shows the skeleton overlay. "
    "Replicate the exact pose, limb positions, and body orientation from Picture 1 "
    "while maintaining the character's identity, clothing, and appearance."
)

DIRECTION_ROTATIONS = {
    "front": 0.0,
    "right": 90.0,
    "back": 180.0,
    "left": 270.0,
}


class SpritesAnimatedStage(Stage):
    """Generate animated sprite frames using HY-Motion or LLM + Anny body mesh + VNCCS edit.

    Motion strategy (packages.sprites.motion_strategy):
      - "hymotion" (default): Uses HY-Motion trained DiT for keyframe generation.
        Falls back to "llm" if shared NPZ files are missing.
      - "llm": Uses LLM (MotionDirector) to generate joint rotations per keyframe.

    For each animation declared in `sprites.animated`:
    1. Resolve motion (HY-Motion NPZ or LLM-generated rotations)
    2. For each keyframe × direction: render Anny mesh → VNCCS edit → save frame
    """

    name = "sprites_animated"
    tier_required = "t1.5"
    depends_on = ["generate"]

    def is_complete(self, state: dict) -> bool:
        """Check if all animated sprite frames exist."""
        sprites = self.ctx.pkg("sprites")
        for anim in sprites.get("animated", []):
            anim_name = anim["name"]
            outfit = sprites.get("outfit", "default")
            for direction in sprites.get("directions", ["front"]):
                frame_dir = self.ctx.sprites_dir / outfit / "normal" / anim_name / direction
                if not frame_dir.exists() or not list(frame_dir.glob("frame_*.png")):
                    return False
        return True

    # ── Strategy resolution ────────────────────────────────────────────────

    def _resolve_motion_strategy(self) -> str:
        """Read motion_strategy from YAML, default to hymotion."""
        return self.ctx.pkg("sprites").get("motion_strategy", "hymotion")

    def _resolve_body_reference(self) -> str:
        """Read body_reference strategy from YAML, default to anny."""
        return self.ctx.pkg("sprites").get("body_reference", "anny")

    def _check_hymotion_available(self) -> bool:
        """Check if any shared HY-Motion NPZ files exist."""
        shared_dir = self.ctx.motions_dir / "shared"
        if not shared_dir.exists():
            return False
        return bool(list(shared_dir.glob("*.npz")))

    # ── Run dispatch ───────────────────────────────────────────────────────

    def run(self, state: dict) -> StageResult:
        motion = self._resolve_motion_strategy()
        body = self._resolve_body_reference()

        # Auto-fallback: hymotion → llm if no NPZ files available
        if motion == "hymotion" and not self._check_hymotion_available():
            print("  [WARN] HY-Motion NPZ not available, falling back to LLM pose guessing")
            motion = "llm"

        # Auto-fallback: trellis → anny if trellis stage didn't complete
        if body == "trellis":
            trellis_state = state.get("stages", {}).get("trellis", {})
            if trellis_state.get("status") != "complete":
                print("  [WARN] TRELLIS not complete, falling back to Anny body mesh")
                body = "anny"

        if motion == "llm":
            return self._run_llm_strategy(state, body)

        return self._run_hymotion_strategy(state, body)

    # ── HY-Motion strategy ─────────────────────────────────────────────────

    def _run_hymotion_strategy(self, state: dict, body: str = "anny") -> StageResult:
        from utils.hymotion_converter import extract_keyframes

        ctx = self.ctx
        sprites = ctx.pkg("sprites")
        outfit = sprites.get("outfit", "default")
        directions = sprites.get("directions", ["front"])

        if not ctx.approved_image.exists():
            return StageResult(status="failed", error="Character image not found")

        # Upload character once
        comfyui_name = self.executor.upload_character(
            ctx.approved_image, f"{ctx.character_name}_char"
        )

        # For Trellis body, preload the GLB
        glb_path = None
        if body == "trellis":
            glb_path = self._resolve_trellis_glb(state)

        all_artifacts = []
        total_frames = 0

        for anim in sprites.get("animated", []):
            anim_name = anim["name"]
            motion_source = anim.get("motion", "shared")
            motion_prompt = anim.get("prompt", "")

            # Resolve motion NPZ
            npz_path = self._resolve_motion(anim_name, motion_source, motion_prompt)
            if not npz_path:
                return StageResult(
                    status="failed",
                    error=f"No motion data for {anim_name}",
                )

            # Extract keyframes
            num_keyframes = 6
            keyframes = extract_keyframes(npz_path, num_keyframes)

            # Generate frames for each keyframe × direction
            for kf_idx, rotations in enumerate(keyframes):
                for direction in directions:
                    frame_seed = ctx.seed + kf_idx * len(directions) + directions.index(direction)
                    if body == "trellis" and glb_path:
                        frame_artifact = self._generate_frame_trellis(
                            comfyui_name=comfyui_name,
                            glb_path=glb_path,
                            outfit=outfit,
                            anim_name=anim_name,
                            direction=direction,
                            kf_idx=kf_idx,
                            seed=frame_seed,
                        )
                    else:
                        frame_artifact = self._generate_frame(
                            comfyui_name=comfyui_name,
                            outfit=outfit,
                            anim_name=anim_name,
                            direction=direction,
                            kf_idx=kf_idx,
                            rotations=rotations,
                            seed=frame_seed,
                        )
                    if frame_artifact:
                        all_artifacts.append(frame_artifact)
                        total_frames += 1

        return StageResult(
            status="complete" if total_frames > 0 else "failed",
            artifacts=all_artifacts,
            meta={
                "motion_strategy": "hymotion",
                "body_reference": body,
                "total_frames": total_frames,
                "animations": [a["name"] for a in sprites.get("animated", [])],
            },
        )

    # ── LLM strategy ───────────────────────────────────────────────────────

    def _run_llm_strategy(self, state: dict, body: str = "anny") -> StageResult:
        """Generate sprite frames using LLM (MotionDirector) for pose keyframes.

        Instead of HY-Motion NPZ extraction, calls the MotionDirector node
        via ComfyUI to generate joint rotations as JSON for each animation.
        """
        ctx = self.ctx
        sprites = ctx.pkg("sprites")
        outfit = sprites.get("outfit", "default")
        directions = sprites.get("directions", ["front"])

        if not ctx.approved_image.exists():
            return StageResult(status="failed", error="Character image not found")

        # Upload character once
        comfyui_name = self.executor.upload_character(
            ctx.approved_image, f"{ctx.character_name}_char"
        )

        all_artifacts = []
        total_frames = 0

        for anim in sprites.get("animated", []):
            anim_name = anim["name"]
            motion_prompt = anim.get("prompt", anim_name)
            num_keyframes = 6

            # Generate keyframes via LLM (MotionDirector)
            keyframes = self._generate_llm_keyframes(
                motion_prompt, num_keyframes, anim_name
            )
            if not keyframes:
                print(f"  [WARN] LLM keyframe generation failed for {anim_name}, skipping")
                continue

            # Generate frames for each keyframe × direction
            for kf_idx, rotations in enumerate(keyframes):
                for direction in directions:
                    frame_seed = ctx.seed + kf_idx * len(directions) + directions.index(direction)
                    frame_artifact = self._generate_frame(
                        comfyui_name=comfyui_name,
                        outfit=outfit,
                        anim_name=anim_name,
                        direction=direction,
                        kf_idx=kf_idx,
                        rotations=rotations,
                        seed=frame_seed,
                    )
                    if frame_artifact:
                        all_artifacts.append(frame_artifact)
                        total_frames += 1

        return StageResult(
            status="complete" if total_frames > 0 else "failed",
            artifacts=all_artifacts,
            meta={
                "motion_strategy": "llm",
                "total_frames": total_frames,
                "animations": [a["name"] for a in sprites.get("animated", [])],
            },
        )

    def _generate_llm_keyframes(
        self, motion_prompt: str, num_keyframes: int, anim_name: str
    ) -> list[dict]:
        """Generate keyframe rotations via MotionDirector (LLM).

        Submits the workflow and parses MotionDirector's STRING output
        to extract JSON rotations. Falls back to identity poses only
        if text parsing fails.
        """
        from pipeline.ui_to_api import load_ui_workflow, patch_ui_workflow, convert_ui_to_api
        from pipeline.comfyui_client import submit_workflow, get_output_text

        ctx = self.ctx

        patches = {
            "MotionDirector": {
                "motion_description": motion_prompt,
                "num_keyframes": num_keyframes,
                "total_frames": num_keyframes,
            },
        }

        try:
            wf = load_ui_workflow(str(ctx.workflows_dir / "03_motion_keyframes_qwen_edit_mesh.json"))
            patch_ui_workflow(wf, patches)

            # Find MotionDirector node ID to read its text output
            md_node = next((n for n in wf["nodes"] if n["type"] == "MotionDirector"), None)
            if not md_node:
                print("  [WARN] No MotionDirector node in workflow")
                return []

            api = convert_ui_to_api(wf)
            result = submit_workflow(api, timeout=300, base_url=ctx.comfyui_url)

            # Get STRING output from MotionDirector node
            md_id = str(md_node["id"])
            text_output = get_output_text(result["outputs"], md_id, base_url=ctx.comfyui_url)

            if not text_output:
                print("  [WARN] MotionDirector returned no text output")
                return []

            # Parse the JSON rotations
            return self._parse_rotations_json(text_output, num_keyframes)

        except Exception as e:
            print(f"  [WARN] LLM motion generation failed: {e}")
            return []

    def _parse_rotations_json(self, text: str, num_keyframes: int) -> list[dict]:
        """Parse rotation JSON from MotionDirector text output.

        Handles: direct JSON array, markdown-wrapped, trailing commas.
        """
        from utils.json_parser import parse_llm_json

        try:
            parsed = parse_llm_json(text)
        except Exception:
            print(f"  [WARN] Could not parse MotionDirector output as JSON")
            return []

        if isinstance(parsed, list):
            return parsed[:num_keyframes]
        if isinstance(parsed, dict):
            # Single keyframe dict
            return [parsed]
        if isinstance(parsed, str):
            # Double-encoded
            try:
                inner = json.loads(parsed)
                if isinstance(inner, list):
                    return inner[:num_keyframes]
            except Exception:
                pass

        return []

    # ── Motion resolution (HY-Motion) ──────────────────────────────────────

    def _resolve_motion(
        self, anim_name: str, motion_source: str, motion_prompt: str
    ) -> Path | None:
        """Resolve motion NPZ path from the specified source."""
        ctx = self.ctx

        if motion_source == "shared":
            npz_path = ctx.motions_dir / "shared" / f"{anim_name}.npz"
            if not npz_path.exists():
                fallback = ctx.motions_dir / "shared" / "idle.npz"
                if fallback.exists():
                    return fallback
            return npz_path if npz_path.exists() else None

        elif motion_source == "custom":
            npz_path = ctx.motions_dir / "custom" / f"{ctx.character_name}_{anim_name}.npz"
            if npz_path.exists():
                return npz_path
            # Generate via HY-Motion
            return self._run_hymotion(motion_prompt, anim_name)

        elif motion_source == "file":
            npz_path = Path(ctx.pkg("sprites")
                           .get("animated", [{}])[0].get("path", ""))
            return npz_path if npz_path.exists() else None

        elif motion_source == "manual":
            return self._build_manual_npz(anim_name)

        return None

    def _run_hymotion(self, prompt: str, anim_name: str) -> Path | None:
        """Run HY-Motion via ComfyUI workflow to generate motion NPZ."""
        ctx = self.ctx
        npz_target = ctx.motions_dir / "custom" / f"{ctx.character_name}_{anim_name}.npz"
        if npz_target.exists():
            return npz_target

        npz_target.parent.mkdir(parents=True, exist_ok=True)

        try:
            patches = {
                "HYMotionEncodeText": {"text": prompt},
                "HYMotionGenerate": {"seed": ctx.seed},
                "HYMotionSaveNPZ": {
                    "output_dir": "motions_custom",
                    "filename_prefix": f"{ctx.character_name}_{anim_name}",
                },
            }
            self.executor.run_ui_workflow(
                workflow_path=ctx.workflows_dir / "01_hymotion_keyframes.json",
                patches=patches,
                output_node_type="HYMotionSaveNPZ",
                timeout=600,
            )

            # Find the NPZ in ComfyUI output
            comfyui_output = ctx.project_root / "output"
            npz_files = sorted(
                comfyui_output.rglob(f"{ctx.character_name}_{anim_name}*.npz"),
                key=lambda p: p.stat().st_mtime,
            )
            if npz_files:
                import shutil
                shutil.copy2(npz_files[-1], npz_target)
                return npz_target
        except Exception as e:
            print(f"  [WARN] HY-Motion failed: {e}")

        return None

    def _build_manual_npz(self, anim_name: str) -> Path:
        """Build an NPZ from manual joint rotation data."""
        import numpy as np

        ctx = self.ctx
        npz_path = ctx.motions_dir / "custom" / f"{ctx.character_name}_{anim_name}_manual.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        # Get manual frames from YAML
        frames = []
        for anim in ctx.pkg("sprites").get("animated", []):
            if anim["name"] == anim_name:
                frames = anim.get("frames", [])
                break

        np.savez(npz_path, frames=frames)
        return npz_path

    # ── Frame generation (shared by both strategies) ────────────────────────

    def _generate_frame(
        self,
        comfyui_name: str,
        outfit: str,
        anim_name: str,
        direction: str,
        kf_idx: int,
        rotations: dict,
        seed: int,
    ) -> str | None:
        """Generate a single sprite frame. Returns artifact path or None."""
        ctx = self.ctx
        model_rotation_y = DIRECTION_ROTATIONS.get(direction, 0.0)

        patches = {
            "LoadImage": {"image": comfyui_name},
            "BodyMeshRenderer": {
                "rotations_json": json.dumps(rotations),
                "model_rotation_y": model_rotation_y,
            },
            "VNCCS_QWEN_Encoder": {
                "prompt": VNCCS_PROMPT,
                "instruction": VNCCS_INSTRUCTION,
            },
            "KSampler": {"seed": seed},
            "SaveImage": {"filename_prefix": f"PoseDirector/{ctx.character_name}/sprite"},
        }

        try:
            images = self.executor.run_ui_workflow(
                workflow_path=ctx.workflows_dir / "05_sprite_bodymesh_edit.json",
                patches=patches,
                output_node_type="SaveImage",
                timeout=300,
            )

            if images:
                frame_dir = ctx.sprites_dir / outfit / "normal" / anim_name / direction
                frame_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frame_dir / f"frame_{kf_idx + 1:03d}.png"
                frame_path.write_bytes(images[0])
                rel_path = frame_path.relative_to(ctx.output_dir)
                return str(rel_path)

        except Exception as e:
            print(f"  [WARN] Frame {kf_idx+1} {direction} failed: {e}")

        return None

    # ── Trellis body reference helpers ────────────────────────────────────

    def _resolve_trellis_glb(self, state: dict) -> Path | None:
        """Find the Trellis GLB from the trellis stage artifacts."""
        trellis_artifacts = state.get("stages", {}).get("trellis", {}).get("artifacts", [])
        for art in trellis_artifacts:
            if art.endswith(".glb"):
                glb_path = self.ctx.output_dir / art
                if glb_path.exists():
                    return glb_path
        return None

    def _generate_frame_trellis(
        self,
        comfyui_name: str,
        glb_path: Path,
        outfit: str,
        anim_name: str,
        direction: str,
        kf_idx: int,
        seed: int,
    ) -> str | None:
        """Generate a sprite frame using Trellis 3D model instead of Anny.

        Renders the Trellis GLB from the specified viewing angle, uploads it,
        and feeds it through the VNCCS sprite pipeline.
        """
        import io
        from PIL import Image
        from utils.mesh_renderer import render_glb
        from pipeline.comfyui_client import upload_image
        from pipeline.ui_to_api import load_ui_workflow, convert_ui_to_api
        from pipeline.comfyui_client import submit_workflow, get_output_images

        ctx = self.ctx
        rotation_y = DIRECTION_ROTATIONS.get(direction, 0.0)

        # Render Trellis GLB from this viewing angle
        mesh_image = render_glb(
            glb_path,
            width=1024,
            height=1024,
            model_rotation_y=rotation_y,
        )

        # Upload rendered mesh to ComfyUI
        img = Image.fromarray(mesh_image)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        mesh_name = f"{ctx.character_name}_trellis_{direction}_f{kf_idx}"
        upload_image(buf.getvalue(), f"{mesh_name}.png", "PoseDirector", base_url=ctx.comfyui_url)
        comfyui_mesh = f"PoseDirector/{mesh_name}.png"

        # Load workflow, replace BodyMeshRenderer (node 2) with LoadImage
        wf = load_ui_workflow(str(ctx.workflows_dir / "05_sprite_bodymesh_edit.json"))
        body_node = next(n for n in wf["nodes"] if n["id"] == 2)
        body_node["type"] = "LoadImage"
        body_node["widgets_values"] = [comfyui_mesh, "image"]
        body_node.pop("properties", None)

        # Set character image on original LoadImage (node 1)
        for n in wf["nodes"]:
            if n["type"] == "LoadImage" and n["id"] == 1:
                n["widgets_values"] = [comfyui_name, "image"]

        # Set seed on KSampler
        for n in wf["nodes"]:
            if n["type"] == "KSampler":
                if not n.get("widgets_values"):
                    n["widgets_values"] = [seed, "randomize", 20, 7.0, "euler", "simple", 1.0]
                else:
                    n["widgets_values"][0] = seed

        try:
            api = convert_ui_to_api(wf)
            result = submit_workflow(api, timeout=300, base_url=ctx.comfyui_url)
            save_node = next(n for n in wf["nodes"] if n["type"] == "SaveImage")
            images = get_output_images(result["outputs"], save_node["id"], base_url=ctx.comfyui_url)

            if images:
                frame_dir = ctx.sprites_dir / outfit / "normal" / anim_name / direction
                frame_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frame_dir / f"frame_{kf_idx + 1:03d}.png"
                frame_path.write_bytes(images[0])
                return str(frame_path.relative_to(ctx.output_dir))

        except Exception as e:
            print(f"  [WARN] Trellis frame {kf_idx+1} {direction} failed: {e}")

        return None
