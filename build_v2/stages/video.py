"""Video stage — LTX Video assembly from sprite frames into animation clips.

Uses the 04_ltx_video_assembly.json workflow to combine sprite frames
into smooth video animations with keyframe guides.
"""

from __future__ import annotations

from pathlib import Path

from build_v2.stages.base import Stage, StageResult


class VideoStage(Stage):
    """Assemble sprite frames into video clips via LTX Video.

    Takes static sprite frames and combines them into animated video clips
    using the LTX Video 2.3 assembly workflow. Each sprite direction/pose
    becomes a short looping video clip.

    Uses the UI-format 04_ltx_video_assembly.json workflow which handles
    LTXVAddGuideAdvanced keyframe injection and 5D latent video generation.
    """

    name = "video"
    tier_required = "t1.5"
    depends_on = ["sprites_static"]

    def is_complete(self, state: dict) -> bool:
        video_dir = self.ctx.output_dir / "video"
        return video_dir.exists() and bool(list(video_dir.glob("*.mp4")))

    def _get_sprite_frames(self) -> list[Path]:
        """Collect sprite frame images from the sprites output."""
        sprites_dir = self.ctx.sprites_dir / "static"
        if not sprites_dir.exists():
            return []
        return sorted(sprites_dir.glob("*.png"))

    def run(self, state: dict) -> StageResult:
        ctx = self.ctx
        frames = self._get_sprite_frames()

        if not frames:
            return StageResult(status="skipped", error="No sprite frames found")

        video_dir = ctx.output_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)

        # Upload the first sprite frame as the reference image
        source_frame = frames[0]
        comfyui_name = self.executor.upload_character(
            source_frame, f"{ctx.character_name}_video_source"
        )

        # Build patches for the LTX Video assembly workflow
        # The workflow uses keyframe guides at specific frame indices
        # For a basic clip, we use the sprite frame as both start and end guide
        patches = {
            "LoadImage": {"image": comfyui_name},
            "SaveImage": {
                "filename_prefix": f"PoseDirector/{ctx.character_name}/video/clip",
            },
        }

        try:
            images = self.executor.run_ui_workflow(
                workflow_path=ctx.workflows_dir / "04_ltx_video_assembly.json",
                patches=patches,
                output_node_type="SaveImage",
                timeout=600,  # 10 min — LTX Video can be slow
            )
        except Exception as e:
            return StageResult(status="failed", error=f"Video assembly failed: {e}")

        if not images:
            return StageResult(status="failed", error="Video assembly produced no output")

        artifacts = []
        for i, data in enumerate(images):
            out_path = video_dir / f"clip_{i:03d}.png"
            out_path.write_bytes(data)
            artifacts.append(str(out_path.relative_to(ctx.output_dir)))

        return StageResult(
            status="complete",
            artifacts=artifacts,
            meta={"frame_count": len(frames), "source": "ltx_video"},
        )
