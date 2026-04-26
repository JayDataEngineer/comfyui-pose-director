"""MotionMeshRenderer — renders HY-Motion 3D mesh keyframes.

Takes HYMOTION_DATA (from HYMotionGenerate) and renders the wooden boy mesh
at selected keyframe intervals using HY-Motion's bundled WoodenMesh model
and our existing pyrender pipeline. Outputs proper 3D body mesh images
instead of skeleton stick figures.
"""

import os

import numpy as np
import torch

from ..utils.mesh_renderer import render_mesh

# Lazy-loaded WoodenMesh instance
_mesh_model = None
_mesh_model_path = None


def _get_mesh_model():
    """Load and cache the WoodenMesh model from HY-Motion assets."""
    global _mesh_model, _mesh_model_path
    if _mesh_model is not None:
        return _mesh_model

    # Find the dump_wooden directory in ComfyUI-HY-Motion1 plugin
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..",
                     "ComfyUI-HY-Motion1", "assets", "dump_wooden"),
        os.path.join(os.path.dirname(__file__), "..", "..",
                     "comfyui-pose-director", "assets", "dump_wooden"),
        "/home/ubuntu/Documents/programs/img/comfyui/custom_nodes/ComfyUI-HY-Motion1/assets/dump_wooden",
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.isdir(path):
            _mesh_model_path = path
            break

    if _mesh_model_path is None:
        raise FileNotFoundError(
            "Cannot find dump_wooden/ mesh assets from ComfyUI-HY-Motion1"
        )

    # Import WoodenMesh from the HY-Motion plugin
    sys_path = os.path.normpath(os.path.join(_mesh_model_path, "..", ".."))
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    from hymotion.pipeline.body_model import WoodenMesh
    _mesh_model = WoodenMesh(_mesh_model_path)
    return _mesh_model


class MotionMeshRenderer:
    """Render 3D body mesh keyframes from HY-Motion motion data."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA",),
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 3}),
                "frame_step": ("INT", {"default": 5, "min": 1, "max": 30}),
                "num_keyframes": ("INT", {"default": 3, "min": 0, "max": 50,
                                          "tooltip": "Max keyframes to output. 0 = all frames."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NAMES = ("mesh_images",)
    FUNCTION = "render"
    CATEGORY = "pose_director"
    DESCRIPTION = "Render 3D body mesh keyframes from HY-Motion motion data."

    def render(self, motion_data, sample_index=0, frame_step=5,
               num_keyframes=3, width=1024, height=1024):
        model = _get_mesh_model()
        faces = model.faces.astype(np.int32)

        # Extract rot6d and translation for the selected sample
        rot6d = motion_data.output_dict["rot6d"]  # (batch, frames, 22, 6)
        transl = motion_data.output_dict["transl"]  # (batch, frames, 3)

        idx = min(sample_index, rot6d.shape[0] - 1)
        sample_rot6d = rot6d[idx]  # (frames, 22, 6)
        sample_transl = transl[idx]  # (frames, 3)

        if hasattr(sample_rot6d, "cpu"):
            sample_rot6d = sample_rot6d.cpu()
        if hasattr(sample_transl, "cpu"):
            sample_transl = sample_transl.cpu()

        total_frames = sample_rot6d.shape[0]

        # Pick frame indices
        frame_indices = list(range(0, total_frames, frame_step))
        if num_keyframes > 0:
            frame_indices = frame_indices[:num_keyframes]

        if not frame_indices:
            return (torch.zeros(1, height, width, 3),)

        rendered_frames = []
        for fi in frame_indices:
            # Forward pass: deform mesh for this frame
            params = {
                "rot6d": sample_rot6d[fi : fi + 1],  # (1, 22, 6)
                "trans": sample_transl[fi : fi + 1],   # (1, 3)
            }
            result = model.forward(params)
            vertices = result["vertices"][0].detach().cpu().numpy()  # (V, 3)

            # SMPL mesh is Y-up, our renderer expects Z-up (Anny convention).
            # Swap Y↔Z so Y becomes depth and Z becomes height.
            vertices = vertices[:, [0, 2, 1]]

            # Scale down to add headroom — prevents head clipping
            vertices = vertices * 0.75

            # Render to 2D image
            image = render_mesh(vertices, faces, width, height)
            rendered_frames.append(image)

        # Stack into ComfyUI tensor: (N, H, W, 3) float32 [0, 1]
        batch = np.stack(rendered_frames, axis=0)
        tensor = torch.from_numpy(batch.astype(np.float32) / 255.0)

        print(f"[MotionMeshRenderer] Rendered {len(rendered_frames)} keyframe meshes "
              f"({width}x{height})")

        return (tensor,)
