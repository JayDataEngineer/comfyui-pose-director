"""MotionDirector — Generates keyframe poses from a start pose + text using LLM."""

import json

import torch

from ..utils.api_client import call_llm
from ..utils.json_parser import parse_llm_json, extract_keypoints_from_dwpose
from .pose_renderer import PoseRendererNode


SYSTEM_PROMPT = """You are a motion keyframe generator. You receive an OpenPose keypoint array for a starting pose and a text description of a motion. You must generate keyframe poses at specific frame numbers.

The keypoints use COCO-18 format: flat array of [x, y, confidence] for each of 18 points:
0=Nose, 1=Neck, 2=RShoulder, 3=RElbow, 4=RWrist, 5=LShoulder, 6=LElbow, 7=LWrist,
8=RHip, 9=RKnee, 10=RAnkle, 11=LHip, 12=LKnee, 13=LAnkle, 14=REye, 15=LEye, 16=REar, 17=LEar

Rules:
- Keep the overall scale and position similar to the input
- Maintain anatomical constraints (elbows bend one way, knees bend one way)
- Keep the person balanced in each keyframe
- Smooth transitions between keyframes (no sudden jumps)
- Each keyframe should be a plausible pose at that point in the motion
- Use confidence 0.9 for keypoints you are sure about, 0.5 for uncertain

You MUST respond with ONLY a JSON object with this exact format:
{
  "keyframes": [
    {"frame": 0, "keypoints": [...]},
    {"frame": <n>, "keypoints": [...]},
    ...
  ]
}

No explanations, no markdown, just the raw JSON object."""


class MotionDirectorNode:
    """Generate keyframe poses from a start pose and motion description using an LLM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("STRING", {"default": "[]", "multiline": True}),
                "motion_description": ("STRING", {"default": "Person waving hello", "multiline": True}),
                "num_keyframes": ("INT", {"default": 4, "min": 2, "max": 16}),
                "total_frames": ("INT", {"default": 48, "min": 8, "max": 240}),
                "base_url": ("STRING", {"default": "https://generativelanguage.googleapis.com/v1beta/openai/"}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "models/gemini-2.5-flash"}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
            },
            "optional": {
                "pose_keypoint": ("POSE_KEYPOINT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "POSE_KEYPOINT")
    OUTPUT_NAMES = ("pose_batch", "keyframes_json", "pose_keypoints")
    FUNCTION = "generate"
    CATEGORY = "pose_director"
    DESCRIPTION = "Generate a sequence of keyframe poses from a start pose and motion description using an LLM."

    def generate(self, keypoints, motion_description, num_keyframes, total_frames,
                 base_url, api_key, model, width, height, pose_keypoint=None):
        # If POSE_KEYPOINT provided, extract from it
        if pose_keypoint is not None:
            kpts, canvas_h, canvas_w = extract_keypoints_from_dwpose(pose_keypoint)
            if canvas_h and canvas_w:
                height = canvas_h
                width = canvas_w
        else:
            kpts = json.loads(keypoints) if isinstance(keypoints, str) else keypoints

        # Calculate frame spacing for keyframes
        frame_interval = total_frames // (num_keyframes - 1) if num_keyframes > 1 else total_frames
        target_frames = [i * frame_interval for i in range(num_keyframes)]
        # Ensure last frame equals total_frames
        target_frames[-1] = total_frames - 1

        user_msg = f"""Starting pose keypoints (flat array, [x, y, confidence] per point):
{json.dumps(kpts)}

Image dimensions: {width}x{height}
Motion description: {motion_description}
Total frames: {total_frames}
Number of keyframes to generate: {num_keyframes}
Target frame numbers: {target_frames}

Generate keyframe poses for the motion described above. The first keyframe (frame 0) should be close to the starting pose. Generate smooth, anatomically plausible intermediate and ending poses.

Respond with ONLY the JSON keyframes object."""

        response = call_llm(base_url, api_key, model, SYSTEM_PROMPT, user_msg)

        keyframes_data = parse_llm_json(response)
        keyframes = keyframes_data["keyframes"]
        keyframes_json = json.dumps(keyframes_data)

        # Render each keyframe into a skeleton image
        renderer = PoseRendererNode()
        images = []
        for kf in keyframes:
            kf_kpts = kf["keypoints"]
            render_input = json.dumps([{"pose_keypoints_2d": kf_kpts}])
            (img_tensor,) = renderer.render(render_input, width, height, 4, 6)
            images.append(img_tensor)

        # Stack all images into a single batch tensor [N, H, W, 3]
        batch = torch.cat(images, dim=0)

        # Build POSE_KEYPOINT compatible output
        pose_keypoint_out = [
            {
                "people": [{"pose_keypoints_2d": kf["keypoints"]}],
                "canvas_height": height,
                "canvas_width": width,
                "frame": kf["frame"],
            }
            for kf in keyframes
        ]

        return (batch, keyframes_json, pose_keypoint_out)
