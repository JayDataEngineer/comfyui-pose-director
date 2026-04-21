"""PoseManipulator — Takes an OpenPose skeleton + text prompt, uses LLM to recalculate joint positions."""

import json
import torch
import numpy as np
from PIL import Image

from ..utils.api_client import call_llm
from ..utils.openpose import KEYPOINT_NAMES, BODY_PARTS
from .pose_renderer import PoseRendererNode


SYSTEM_PROMPT = """You are a pose manipulation assistant. You receive OpenPose keypoint data for a person and a text instruction describing how to modify their pose.

You MUST respond with ONLY a JSON array of the same format you received. No explanations, no markdown, just the raw JSON array of keypoint coordinates.

The keypoints use COCO-18 format: [x, y, confidence] for each of 18 points:
0=Nose, 1=Neck, 2=RShoulder, 3=RElbow, 4=RWrist, 5=LShoulder, 6=LElbow, 7=LWrist,
8=RHip, 9=RKnee, 10=RAnkle, 11=LHip, 12=LKnee, 13=LAnkle, 14=REye, 15=LEye, 16=REar, 17=LEar

Rules:
- Keep the overall scale and position similar to the input
- Adjust confidence values if you moved a keypoint (0.9 for confident, 0.5 for uncertain)
- Maintain anatomical constraints (elbows bend one way, knees bend one way)
- Keep the person balanced and grounded
- Output ONLY the keypoints array, no wrapper object"""


class PoseManipulatorNode:
    """Manipulate an OpenPose skeleton using an LLM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("STRING", {"default": "[]", "multiline": True}),
                "instruction": ("STRING", {"default": "Raise both arms above head", "multiline": True}),
                "base_url": ("STRING", {"default": "http://localhost:11434/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gemma3:12b"}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    OUTPUT_NAMES = ("pose_image", "keypoints_json")
    FUNCTION = "manipulate"
    CATEGORY = "pose_director"
    DESCRIPTION = "Use an LLM to modify an OpenPose skeleton based on a text instruction."

    def manipulate(self, keypoints, instruction, base_url, api_key, model, width, height):
        kpts = json.loads(keypoints)

        user_msg = f"""Here are the current keypoints (flat array, [x, y, confidence] per point):
{json.dumps(kpts)}

Image dimensions: {width}x{height}

Instruction: {instruction}

Respond with ONLY the modified keypoints array."""

        response = call_llm(base_url, api_key, model, SYSTEM_PROMPT, user_msg)

        # Parse the LLM response — extract JSON from possible markdown wrapping
        response = response.strip()
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()

        new_kpts = json.loads(response)
        new_kpts_json = json.dumps(new_kpts)

        # Render the skeleton image
        renderer = PoseRendererNode()
        (img_tensor,) = renderer.render(new_kpts_json, width, height, 4, 6)

        return (img_tensor, new_kpts_json)
