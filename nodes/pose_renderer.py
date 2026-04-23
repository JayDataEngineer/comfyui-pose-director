"""PoseRenderer — Converts OpenPose keypoint data into skeleton images for ControlNet."""

import json
import math

import torch
import numpy as np
from PIL import Image, ImageDraw

from ..utils.openpose import SKELETON_BONES, BONE_COLORS, KEYPOINT_COLORS


class PoseRendererNode:
    """Renders OpenPose keypoints into skeleton images suitable for ControlNet input.

    Matches the reference rendering from controlnet_aux: per-bone gradient colors,
    thick elliptical bones, and per-keypoint circles.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("STRING", {"default": "[]", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "line_width": ("INT", {"default": 4, "min": 1, "max": 20}),
                "point_radius": ("INT", {"default": 4, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "pose_director"
    DESCRIPTION = "Render OpenPose keypoints JSON into a skeleton image matching DWPose output style."

    def render(self, keypoints, width, height, line_width=4, point_radius=4):
        people = json.loads(keypoints)
        if isinstance(people, dict):
            people = [people]

        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        for person in people:
            pts = person.get("pose_keypoints_2d", person.get("keypoints", []))
            # Flatten any nested lists and convert to float
            flat_pts = []
            for v in pts:
                if isinstance(v, (list, tuple)):
                    for sub in v:
                        try:
                            flat_pts.append(float(sub))
                        except (ValueError, TypeError):
                            flat_pts.append(0.0)
                else:
                    try:
                        flat_pts.append(float(v))
                    except (ValueError, TypeError):
                        flat_pts.append(0.0)
            pts = flat_pts

            # Draw bones with per-bone gradient colors
            for bone_idx, (i, j) in enumerate(SKELETON_BONES):
                if i * 3 + 2 >= len(pts) or j * 3 + 2 >= len(pts):
                    continue
                c1 = pts[i * 3 + 2]
                c2 = pts[j * 3 + 2]
                if c1 <= 0.1 or c2 <= 0.1:
                    continue

                x1, y1 = pts[i * 3], pts[i * 3 + 1]
                x2, y2 = pts[j * 3], pts[j * 3 + 1]
                color = BONE_COLORS[bone_idx]

                # Draw thick line (PIL doesn't do elliptical bones, but thick lines work)
                draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)

            # Draw keypoints with per-keypoint gradient colors
            for idx in range(min(len(pts) // 3, len(KEYPOINT_COLORS))):
                conf = pts[idx * 3 + 2]
                if conf <= 0.1:
                    continue
                x, y = pts[idx * 3], pts[idx * 3 + 1]
                color = KEYPOINT_COLORS[idx]
                r = point_radius
                draw.ellipse(
                    [x - r, y - r, x + r, y + r],
                    fill=color,
                )

        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return (img_tensor,)
