"""PoseManipulator — Takes an OpenPose skeleton + text prompt, uses LLM to recalculate joint positions."""

import json

from ..utils.api_client import call_llm
from ..utils.json_parser import parse_llm_json, extract_keypoints_from_dwpose
from ..utils.skeleton_validator import validate_keypoints, sanitize_keypoints
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
                "base_url": ("STRING", {"default": "https://generativelanguage.googleapis.com/v1beta/openai/"}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "models/gemini-2.5-flash"}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
            },
            "optional": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "base_pose": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "POSE_KEYPOINT")
    OUTPUT_NAMES = ("pose_image", "keypoints_json", "pose_keypoint")
    FUNCTION = "manipulate"
    CATEGORY = "pose_director"
    DESCRIPTION = "Use an LLM to modify an OpenPose skeleton based on a text instruction."

    def manipulate(self, keypoints, instruction, base_url, api_key, model, width, height, pose_keypoint=None, base_pose=""):
        # If POSE_KEYPOINT provided, extract from it (takes priority over STRING input)
        if pose_keypoint is not None:
            kpts, canvas_h, canvas_w = extract_keypoints_from_dwpose(pose_keypoint)
            if canvas_h and canvas_w:
                height = canvas_h
                width = canvas_w
        else:
            kpts = json.loads(keypoints) if isinstance(keypoints, str) else keypoints

        user_msg = f"""Here are the current keypoints (flat array, [x, y, confidence] per point):
{json.dumps(kpts)}

Image dimensions: {width}x{height}

Instruction: {instruction}"""

        # If a base pose from the library is provided, include it as an anatomical anchor
        if base_pose:
            try:
                from ..utils.pose_library import load_pose
                pose_data = load_pose(base_pose)
                user_msg += f"\n\nAnatomical reference (base pose for proportion/scale anchor):\n{json.dumps(pose_data['keypoints'])}\nUse these as a reference for limb proportions and body scale."
            except (FileNotFoundError, KeyError):
                print(f"[PoseManipulator] Warning: base pose '{base_pose}' not found, skipping")

        user_msg += "\n\nRespond with ONLY the modified keypoints array."

        response = call_llm(base_url, api_key, model, SYSTEM_PROMPT, user_msg)

        new_kpts = parse_llm_json(response)
        new_kpts_json = json.dumps(new_kpts)

        # Sanitize overlapping joints before rendering
        new_kpts = sanitize_keypoints(new_kpts, width, height)

        # Soft validation: log anatomical issues but don't block
        is_valid, issues = validate_keypoints(new_kpts, width, height)
        if not is_valid or issues:
            for issue in issues:
                print(f"[PoseManipulator] Validation {issue.severity}: {issue.message}")

        # Build POSE_KEYPOINT compatible output for downstream ControlNet nodes
        pose_keypoint_out = {
            "people": [{"pose_keypoints_2d": new_kpts}],
            "canvas_height": height,
            "canvas_width": width,
        }

        # Render the skeleton image
        renderer = PoseRendererNode()
        render_input = json.dumps([{"pose_keypoints_2d": new_kpts}])
        (img_tensor,) = renderer.render(render_input, width, height, 4, 6)

        return (img_tensor, new_kpts_json, pose_keypoint_out)
