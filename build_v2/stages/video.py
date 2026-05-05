"""Video stage — LTX Video assembly from sprite frames into animation clips.

Uses the 04_ltx_video_assembly.json workflow to combine sprite frames
into smooth video animations with keyframe guides, then encodes to MP4.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from build_v2.stages.base import Stage, StageResult


class VideoStage(Stage):
    """Assemble sprite frames into video clips via LTX Video.

    Takes static sprite frames and combines them into animated video clips
    using the LTX Video 2.3 assembly workflow. Each sprite direction/pose
    becomes a short looping video clip.

    The LTX Video workflow outputs frame PNGs via SaveImage. This stage
    then encodes those frames into MP4 using ffmpeg.
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
                timeout=600,
            )
        except Exception as e:
            return StageResult(status="failed", error=f"Video assembly failed: {e}")

        if not images:
            return StageResult(status="failed", error="Video assembly produced no output")

        # Save individual frames
        frame_dir = video_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        saved_frames = []
        for i, data in enumerate(images):
            frame_path = frame_dir / f"frame_{i:04d}.png"
            frame_path.write_bytes(data)
            saved_frames.append(frame_path)

        # Encode frames to MP4
        mp4_path = video_dir / f"clip_{ctx.character_name}.mp4"
        artifacts = self._encode_mp4(saved_frames, mp4_path, ctx)

        return StageResult(
            status="complete" if artifacts else "failed",
            artifacts=artifacts,
            meta={"frame_count": len(frames), "source": "ltx_video"},
        )

    def _encode_mp4(self, frames: list[Path], output_path: Path, ctx) -> list[str]:
        """Encode a sequence of PNG frames into an MP4 video using ffmpeg.

        Returns list of artifact paths relative to output_dir.
        """
        if not frames:
            return []

        fps = ctx.pkg("video").get("fps", 12)

        # Use first frame to get resolution
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(frames[0].read_bytes()))
        w, h = img.size

        # ffmpeg: read numbered PNGs, encode to h264 MP4
        frame_pattern = str(frames[0].parent / "frame_%04d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={w}:{h}:flags=lanczos",
            "-preset", "medium",
            "-crf", "18",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                print(f"  [WARN] ffmpeg encoding failed: {result.stderr[:300]}")
                # Fall back to listing individual frames as artifacts
                return [str(f.relative_to(ctx.output_dir)) for f in frames]
        except FileNotFoundError:
            print("  [WARN] ffmpeg not found, saving frames without MP4 encoding")
            return [str(f.relative_to(ctx.output_dir)) for f in frames]
        except subprocess.TimeoutExpired:
            print("  [WARN] ffmpeg timed out")
            return [str(f.relative_to(ctx.output_dir)) for f in frames]

        return [str(output_path.relative_to(ctx.output_dir))]
