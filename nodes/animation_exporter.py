"""AnimationExporter — ComfyUI node for interpolating and exporting animation.

Takes keyframe rotations from MotionDirector, interpolates them using SLERP,
then exports as BVH (for game engines) and/or MP4/GIF (for preview).
"""

import json
import os

import numpy as np
import torch

from ..utils.interpolation import interpolate_keyframes
from ..utils.bvh_writer import write_bvh
from ..utils.animation_export import (
    frames_to_mp4,
    frames_to_gif,
    get_output_dir,
    render_animation_frames,
)


class AnimationExporterNode:
    """Interpolate keyframes and export animation as BVH + video preview."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rotations_json": ("STRING", {"default": "[]", "multiline": True}),
                "keyframes_json": ("STRING", {"default": "{}", "multiline": True}),
                "total_frames": ("INT", {"default": 48, "min": 8, "max": 480}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "export_bvh": ("BOOLEAN", {"default": True}),
                "export_video": ("BOOLEAN", {"default": True}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048}),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "animation"}),
                "video_format": (["mp4", "gif", "both"], {"default": "mp4"}),
                "backend": (["auto", "pyrender", "pil"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    OUTPUT_NAMES = ("preview_batch", "bvh_path", "video_path")
    FUNCTION = "export"
    CATEGORY = "pose_director"
    DESCRIPTION = "Interpolate keyframes and export animation as BVH + video preview."

    def export(
        self,
        rotations_json: str,
        keyframes_json: str,
        total_frames: int,
        fps: int,
        export_bvh: bool,
        export_video: bool,
        width: int,
        height: int,
        filename_prefix: str = "animation",
        video_format: str = "mp4",
        backend: str = "auto",
    ):
        # Parse inputs
        keyframe_rotations = json.loads(rotations_json) if isinstance(rotations_json, str) else rotations_json
        keyframes_data = json.loads(keyframes_json) if isinstance(keyframes_json, str) else keyframes_json

        # Extract frame numbers from keyframes_json
        frame_numbers = self._extract_frame_numbers(keyframes_data, len(keyframe_rotations), total_frames)

        # Interpolate all frames
        all_frames = interpolate_keyframes(keyframe_rotations, frame_numbers, total_frames)

        # Output directory
        output_dir = get_output_dir()
        bvh_path = ""
        video_path = ""

        # Export BVH
        if export_bvh:
            bvh_path = os.path.join(output_dir, f"{filename_prefix}.bvh")
            write_bvh(all_frames, fps=float(fps), output_path=bvh_path)
            print(f"[AnimationExporter] BVH written to {bvh_path}")

        # Export video
        if export_video:
            images = render_animation_frames(all_frames, width, height, backend=backend)

            if video_format in ("mp4", "both"):
                mp4_path = os.path.join(output_dir, f"{filename_prefix}.mp4")
                frames_to_mp4(images, mp4_path, fps=fps)
                video_path = mp4_path
                print(f"[AnimationExporter] MP4 written to {mp4_path}")

            if video_format in ("gif", "both"):
                gif_path = os.path.join(output_dir, f"{filename_prefix}.gif")
                frames_to_gif(images, gif_path, fps=min(fps, 15))
                if not video_path:
                    video_path = gif_path
                print(f"[AnimationExporter] GIF written to {gif_path}")

        # Build preview batch: sample up to 8 frames evenly
        preview = self._build_preview_batch(all_frames, width, height, backend)

        return (preview, bvh_path, video_path)

    def _extract_frame_numbers(self, keyframes_data, num_keyframes, total_frames) -> list[int]:
        """Extract frame numbers from keyframes_json, or distribute evenly."""
        # Try to get frame numbers from the keyframes data
        if isinstance(keyframes_data, dict) and "keyframes" in keyframes_data:
            kfs = keyframes_data["keyframes"]
            if kfs and "frame" in kfs[0]:
                return [kf["frame"] for kf in kfs]

        # Evenly distribute
        if num_keyframes <= 1:
            return [0]
        interval = total_frames // (num_keyframes - 1)
        frames = [i * interval for i in range(num_keyframes)]
        frames[-1] = total_frames - 1
        return frames

    def _build_preview_batch(self, all_frames, width, height, backend, max_previews=8):
        """Render a subset of frames as a preview batch tensor."""
        n = len(all_frames)
        step = max(1, n // max_previews)
        indices = list(range(0, n, step))[:max_previews]
        if indices[-1] != n - 1:
            indices.append(n - 1)

        from .body_mesh_renderer import BodyMeshRendererNode
        renderer = BodyMeshRendererNode()
        images = []
        for idx in indices:
            rots_json = json.dumps(all_frames[idx])
            (img_tensor,) = renderer.render(rots_json, width, height, backend=backend)
            images.append(img_tensor)

        return torch.cat(images, dim=0)
