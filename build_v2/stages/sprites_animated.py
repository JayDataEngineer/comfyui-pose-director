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
    """Generate animated sprite frames using HY-Motion + Anny body mesh + VNCCS edit.

    For each animation declared in `sprites.animated`:
    1. Resolve motion NPZ (shared, custom via HY-Motion, file, or manual)
    2. Extract keyframes via hymotion_converter
    3. For each keyframe × direction: render Anny mesh → VNCCS edit → save frame
    """

    name = "sprites_animated"
    tier_required = "t1.5"
    depends_on = ["generate"]

    def is_complete(self, state: dict) -> bool:
        """Check if all animated sprite frames exist."""
        sprites = self.ctx.character_data.get("sprites", {})
        for anim in sprites.get("animated", []):
            anim_name = anim["name"]
            outfit = sprites.get("outfit", "default")
            for direction in sprites.get("directions", ["front"]):
                frame_dir = self.ctx.sprites_dir / outfit / "normal" / anim_name / direction
                if not frame_dir.exists() or not list(frame_dir.glob("frame_*.png")):
                    return False
        return True

    def run(self, state: dict) -> StageResult:
        from utils.hymotion_converter import extract_keyframes

        ctx = self.ctx
        sprites = ctx.character_data.get("sprites", {})
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
                "total_frames": total_frames,
                "animations": [a["name"] for a in sprites.get("animated", [])],
            },
        )

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
            npz_path = Path(ctx.character_data.get("sprites", {})
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
        for anim in ctx.character_data.get("sprites", {}).get("animated", []):
            if anim["name"] == anim_name:
                frames = anim.get("frames", [])
                break

        np.savez(npz_path, frames=frames)
        return npz_path

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
