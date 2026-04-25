"""BodyMeshRenderer — ComfyUI node for deterministic 3D body mesh rendering.

Takes named joint rotations from the LLM (via MotionDirector) and renders
a 3D body mesh using Anny + pyrender. No diffusion — pure geometry.

This replaces OpenPose stick figures as the image2 input to VNCCS,
providing body volume and form that VNCCS can actually use.
"""

import json

import numpy as np
import torch

from ..utils.anny_compat import forward_anny, get_faces, rotations_to_pose_parameters
from ..utils.mesh_renderer import render_mesh


class BodyMeshRendererNode:
    """Render a 3D body mesh from named joint rotations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rotations_json": ("STRING", {"default": "{}", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048}),
            },
            "optional": {
                "backend": (["auto", "pyrender", "pil"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NAMES = ("mesh_image",)
    FUNCTION = "render"
    CATEGORY = "pose_director"
    DESCRIPTION = "Render a 3D body mesh from named joint rotations (Anny model)."

    def render(self, rotations_json, width, height, backend="auto"):
        rotations = json.loads(rotations_json) if isinstance(rotations_json, str) else rotations_json

        # Run Anny forward pass
        result = forward_anny(rotations)
        vertices = result["vertices"][0].detach().cpu().numpy()  # (13718, 3)
        faces = get_faces().cpu().numpy()  # (27420, 3)

        # Render to 2D image
        image = render_mesh(vertices, faces, width, height, backend=backend)

        # Convert to ComfyUI tensor format: (1, H, W, 3) float32 [0, 1]
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)

        return (tensor,)
