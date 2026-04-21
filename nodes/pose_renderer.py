"""PoseRenderer — Converts OpenPose keypoint data into skeleton images for ControlNet."""

import torch
import numpy as np
from PIL import Image, ImageDraw

from ..utils.openpose import SKELETON_BONES, KEYPOINT_COLOR


class PoseRendererNode:
    """Renders OpenPose keypoints into skeleton images suitable for ControlNet input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("STRING", {"default": "[]", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "line_width": ("INT", {"default": 4, "min": 1, "max": 20}),
                "point_radius": ("INT", {"default": 6, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "pose_director"
    DESCRIPTION = "Render OpenPose keypoints JSON into a skeleton image."

    def render(self, keypoints, width, height, line_width, point_radius):
        import json

        people = json.loads(keypoints)
        if isinstance(people, dict):
            people = [people]

        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        for person in people:
            pts = person.get("pose_keypoints_2d", person.get("keypoints", []))

            # Draw bones
            for i, j in SKELETON_BONES:
                if i * 3 + 2 < len(pts) and j * 3 + 2 < len(pts):
                    c1 = pts[i * 3 + 2]
                    c2 = pts[j * 3 + 2]
                    if c1 > 0.1 and c2 > 0.1:
                        x1, y1 = pts[i * 3], pts[i * 3 + 1]
                        x2, y2 = pts[j * 3], pts[j * 3 + 1]
                        draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255), width=line_width)

            # Draw keypoints
            for idx in range(0, len(pts), 3):
                conf = pts[idx + 2]
                if conf > 0.1:
                    x, y = pts[idx], pts[idx + 1]
                    draw.ellipse(
                        [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                        fill=KEYPOINT_COLOR,
                    )

        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return (img_tensor,)
