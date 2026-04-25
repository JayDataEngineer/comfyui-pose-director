"""MotionDirector — Generates keyframe poses from a start pose + text using LLM."""

import json

import torch

from ..utils.api_client import call_llm
from ..utils.json_parser import parse_llm_json, extract_keypoints_from_dwpose
from ..utils.skeleton_validator import validate_keypoints, validate_keyframe_consistency, sanitize_keypoints
from .pose_renderer import PoseRendererNode


SYSTEM_PROMPT_ROTATIONS = """You are a motion keyframe generator. You generate poses as joint rotation offsets from a T-pose (arms outstretched horizontally). For each joint, provide rotation in degrees around X (pitch), Y (yaw), Z (roll) axes.

Available joints:
- spine, neck, head
- right_shoulder, right_elbow, right_wrist
- left_shoulder, left_elbow, left_wrist
- right_hip, right_knee, right_ankle
- left_hip, left_knee, left_ankle

Rotation conventions:
- All rotations are in degrees relative to T-pose (0,0,0 = T-pose position)
- X axis: positive = forward tilt, negative = backward tilt
- Y axis: positive = rotate left, negative = rotate right
- Z axis: positive = outward rotation, negative = inward rotation
- Elbows only bend in one direction (negative Z for right, positive Z for left)
- Knees only bend backward (positive X)
- Keep rotations anatomically plausible (no 360° spins)
- Smooth transitions between keyframes

You MUST respond with ONLY a JSON object with this exact format:
{
  "keyframes": [
    {"frame": 0, "rotations": {"spine": [0, 0, 0], "neck": [0, 0, 0], "head": [0, 0, 0], "right_shoulder": [0, 0, 90], "right_elbow": [0, 0, 0], "right_wrist": [0, 0, 0], "left_shoulder": [0, 0, -90], "left_elbow": [0, 0, 0], "left_wrist": [0, 0, 0], "right_hip": [0, 0, 0], "right_knee": [0, 0, 0], "right_ankle": [0, 0, 0], "left_hip": [0, 0, 0], "left_knee": [0, 0, 0], "left_ankle": [0, 0, 0]}},
    {"frame": <n>, "rotations": {...}},
    ...
  ]
}

No explanations, no markdown, just the raw JSON object."""

SYSTEM_PROMPT_KEYPOINTS = """You are a motion keyframe generator. You receive an OpenPose keypoint array for a starting pose and a text description of a motion. You must generate keyframe poses at specific frame numbers.

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
                "base_url": ("STRING", {"default": "https://openrouter.ai/api/v1"}),
                "model": ("STRING", {"default": "deepseek/deepseek-v4-flash"}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
            },
            "optional": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "base_pose": ("STRING", {"default": ""}),
                "output_mode": (["keypoints", "rotations"], {"default": "keypoints"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "POSE_KEYPOINT", "STRING")
    OUTPUT_NAMES = ("pose_batch", "keyframes_json", "pose_keypoints", "rotations_json")
    FUNCTION = "generate"
    CATEGORY = "pose_director"
    DESCRIPTION = "Generate keyframe poses from a start pose and motion description using an LLM."

    def generate(self, keypoints, motion_description, num_keyframes, total_frames,
                 base_url, model, width, height,
                 pose_keypoint=None, base_pose="", output_mode="rotations"):
        # Calculate frame spacing for keyframes
        frame_interval = total_frames // (num_keyframes - 1) if num_keyframes > 1 else total_frames
        target_frames = [i * frame_interval for i in range(num_keyframes)]
        target_frames[-1] = total_frames - 1

        if output_mode == "rotations":
            return self._generate_rotations(
                motion_description, num_keyframes, total_frames, target_frames,
                base_url, model, width, height,
            )
        else:
            return self._generate_keypoints(
                keypoints, motion_description, num_keyframes, total_frames, target_frames,
                base_url, model, width, height, pose_keypoint, base_pose,
            )

    def _generate_rotations(self, motion_description, num_keyframes, total_frames,
                            target_frames, base_url, model, width, height):
        """Rotation-based mode: LLM outputs joint rotations, rendered as 3D meshes."""
        user_msg = f"""Motion description: {motion_description}
Total frames: {total_frames}
Number of keyframes to generate: {num_keyframes}
Target frame numbers: {target_frames}
Image dimensions: {width}x{height}

The first keyframe (frame 0) should be a neutral standing pose (all rotations near zero).
Generate smooth, anatomically plausible poses for each keyframe.

Respond with ONLY the JSON keyframes object."""

        response = call_llm(base_url, "", model, SYSTEM_PROMPT_ROTATIONS, user_msg)

        keyframes_data = parse_llm_json(response)
        if isinstance(keyframes_data, list):
            keyframes_data = keyframes_data[0] if keyframes_data else {}
        keyframes = keyframes_data["keyframes"]
        keyframes_json = json.dumps(keyframes_data)

        # Render each keyframe as a 3D body mesh
        from .body_mesh_renderer import BodyMeshRendererNode
        mesh_renderer = BodyMeshRendererNode()
        images = []
        all_rotations = []
        for kf in keyframes:
            rots = kf["rotations"]
            all_rotations.append(rots)
            rots_json = json.dumps(rots)
            (img_tensor,) = mesh_renderer.render(rots_json, width, height)
            images.append(img_tensor)

        batch = torch.cat(images, dim=0)

        # No POSE_KEYPOINT output in rotation mode — return empty list
        pose_keypoint_out = []

        # Rotations JSON: one rotation dict per keyframe
        rotations_json = json.dumps(all_rotations)

        return (batch, keyframes_json, pose_keypoint_out, rotations_json)

    def _generate_keypoints(self, keypoints, motion_description, num_keyframes, total_frames,
                            target_frames, base_url, model, width, height,
                            pose_keypoint=None, base_pose=""):
        # If POSE_KEYPOINT provided, extract from it
        if pose_keypoint is not None:
            kpts, canvas_h, canvas_w = extract_keypoints_from_dwpose(pose_keypoint)
            if canvas_h and canvas_w:
                height = canvas_h
                width = canvas_w
        else:
            kpts = json.loads(keypoints) if isinstance(keypoints, str) else keypoints

        user_msg = f"""Starting pose keypoints (flat array, [x, y, confidence] per point):
{json.dumps(kpts)}

Image dimensions: {width}x{height}
Motion description: {motion_description}
Total frames: {total_frames}
Number of keyframes to generate: {num_keyframes}
Target frame numbers: {target_frames}"""

        # If a base pose from the library is provided, include it as an anatomical anchor
        if base_pose:
            try:
                from ..utils.pose_library import load_pose
                pose_data = load_pose(base_pose)
                user_msg += f"\n\nAnatomical reference (base pose for proportion/scale anchor):\n{json.dumps(pose_data['keypoints'])}\nKeep limb lengths and body proportions consistent with this reference across all keyframes."
            except (FileNotFoundError, KeyError):
                print(f"[MotionDirector] Warning: base pose '{base_pose}' not found, skipping")

        user_msg += "\n\nGenerate keyframe poses for the motion described above. The first keyframe (frame 0) should be close to the starting pose. Generate smooth, anatomically plausible intermediate and ending poses.\n\nRespond with ONLY the JSON keyframes object."

        response = call_llm(base_url, "", model, SYSTEM_PROMPT_KEYPOINTS, user_msg)

        keyframes_data = parse_llm_json(response)
        print(f"[MotionDirector] Parsed: type={type(keyframes_data).__name__}, "
              f"keys={list(keyframes_data.keys()) if isinstance(keyframes_data, dict) else 'list len=' + str(len(keyframes_data))}")
        # LLM may return a list wrapping the dict — unwrap it
        if isinstance(keyframes_data, list):
            keyframes_data = keyframes_data[0] if keyframes_data else {}
        keyframes = keyframes_data["keyframes"]
        keyframes_json = json.dumps(keyframes_data)

        # Soft validation: log anatomical issues per keyframe
        all_kpts = []
        for i, kf in enumerate(keyframes):
            kf_kpts = kf["keypoints"]
            is_valid, issues = validate_keypoints(kf_kpts, width, height)
            if not is_valid or issues:
                for issue in issues:
                    print(f"[MotionDirector] Keyframe {i} validation {issue.severity}: {issue.message}")
            all_kpts.append(kf_kpts)

        # Cross-keyframe consistency check (shimmer detection)
        if len(all_kpts) >= 2:
            consistency = validate_keyframe_consistency(all_kpts)
            if not consistency["is_consistent"]:
                print(f"[MotionDirector] Keyframe consistency warning: {consistency['flagged_bones']}")

        # Render each keyframe into a skeleton image
        renderer = PoseRendererNode()
        images = []
        for kf in keyframes:
            kf_kpts = sanitize_keypoints(kf["keypoints"], width, height)
            kf["keypoints"] = kf_kpts  # update for downstream use
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

        return (batch, keyframes_json, pose_keypoint_out, "")
