"""MotionDirector — Generates keyframe poses from a motion description using LLM.

Rotation-only mode: LLM outputs joint rotation angles (degrees from T-pose),
validated and sanitized before rendering as 3D Anny body meshes.
"""

import json

import torch

from ..utils.api_client import call_llm
from ..utils.json_parser import parse_llm_json
from ..utils.rotation_validator import (
    REQUIRED_JOINTS,
    validate_rotations,
    sanitize_rotations,
    validate_rotation_consistency,
)

SYSTEM_PROMPT = """\
You are a motion keyframe generator for a 3D humanoid body model (Anny).
You receive a text description of a motion and output keyframe poses as joint rotations.

## Coordinate System
- All rotations are in **degrees** relative to T-pose (arms outstretched horizontally, palms down).
- Rotation order: **XYZ intrinsic** (equivalent to Z-Y-X extrinsic). Applied as: first X, then Y, then Z.
- **X-axis (Pitch):** Forward tilt = positive, backward tilt = negative. Think: nodding "yes".
- **Y-axis (Yaw):** Rotate left = positive, rotate right = negative. Think: shaking head "no".
- **Z-axis (Roll):** Move outward/away from body = positive for RIGHT limbs, negative for LEFT limbs. Think: tilting head to shoulder.

## Rotation Hierarchy (IMPORTANT)
Rotations cascade down the skeleton. Parent joint rotations affect child joints in world space.
- To turn the ENTIRE character, change only the **spine Y-rotation** — do NOT rotate every joint.
- Arm rotations are LOCAL to the shoulder. If the shoulder is rotated, the elbow/wrist follow.

## Mirroring (Left vs Right)
For lateral movements (Z-axis), left and right limbs are MIRRORED:
- Right shoulder Z=+90 means arm points DOWN. Left shoulder Z=-90 means arm points DOWN.
- To move BOTH arms out from body: right_shoulder Z=positive, left_shoulder Z=negative.

## Reference Poses (Anchors)
Use these to calibrate your spatial reasoning:
- **Arm straight down:** shoulder Z = +90 (right) or -90 (left)
- **Arm straight up:** shoulder Z = -90 (right) or +90 (left)
- **Elbow bent 90°:** elbow Z = -90 (right) or +90 (left)
- **Knee bent 90°:** knee X = +90
- **Character facing right:** spine Y = -90
- **Leaning forward 45°:** spine X = +45

## Available Joints (exactly 15)
spine, neck, head,
right_shoulder, right_elbow, right_wrist,
left_shoulder, left_elbow, left_wrist,
right_hip, right_knee, right_ankle,
left_hip, left_knee, left_ankle

## Rules
- Elbows only bend in ONE direction: negative Z for right, positive Z for left
- Knees only bend BACKWARD: positive X only (range 0 to 140)
- Keep rotations anatomically plausible (no 360° spins, no dislocated joints)
- Smooth transitions between keyframes (no teleportation)
- Frame 0 is always the neutral T-pose (all zeros)

## Output Format
You MUST respond with ONLY a JSON object — no markdown, no explanations:
{
  "keyframes": [
    {"frame": 0, "rotations": {"spine": [0,0,0], "neck": [0,0,0], "head": [0,0,0], "right_shoulder": [0,0,0], "right_elbow": [0,0,0], "right_wrist": [0,0,0], "left_shoulder": [0,0,0], "left_elbow": [0,0,0], "left_wrist": [0,0,0], "right_hip": [0,0,0], "right_knee": [0,0,0], "right_ankle": [0,0,0], "left_hip": [0,0,0], "left_knee": [0,0,0], "left_ankle": [0,0,0]}},
    {"frame": <n>, "rotations": {<ONLY include joints that CHANGE from previous keyframe>}}
  ]
}

## Example: "Raise right arm and wave"
{
  "keyframes": [
    {"frame": 0, "rotations": {"spine": [0,0,0], "neck": [0,0,0], "head": [0,0,0], "right_shoulder": [0,0,0], "right_elbow": [0,0,0], "right_wrist": [0,0,0], "left_shoulder": [0,0,0], "left_elbow": [0,0,0], "left_wrist": [0,0,0], "right_hip": [0,0,0], "right_knee": [0,0,0], "right_ankle": [0,0,0], "left_hip": [0,0,0], "left_knee": [0,0,0], "left_ankle": [0,0,0]}},
    {"frame": 16, "rotations": {"right_shoulder": [0,0,-90], "right_elbow": [0,0,-90], "right_wrist": [0,0,20]}},
    {"frame": 32, "rotations": {"right_shoulder": [0,0,-90], "right_elbow": [0,0,-90], "right_wrist": [0,0,-20]}},
    {"frame": 47, "rotations": {"right_shoulder": [0,0,-90], "right_elbow": [0,0,-90], "right_wrist": [0,0,20]}}
  ]
}

## Example: "Squat down"
{
  "keyframes": [
    {"frame": 0, "rotations": {"spine": [0,0,0], "neck": [0,0,0], "head": [0,0,0], "right_shoulder": [0,0,0], "right_elbow": [0,0,0], "right_wrist": [0,0,0], "left_shoulder": [0,0,0], "left_elbow": [0,0,0], "left_wrist": [0,0,0], "right_hip": [0,0,0], "right_knee": [0,0,0], "right_ankle": [0,0,0], "left_hip": [0,0,0], "left_knee": [0,0,0], "left_ankle": [0,0,0]}},
    {"frame": 16, "rotations": {"spine": [10,0,0], "right_hip": [60,0,0], "right_knee": [90,0,0], "left_hip": [60,0,0], "left_knee": [90,0,0], "right_shoulder": [0,0,45], "left_shoulder": [0,0,-45]}},
    {"frame": 32, "rotations": {"spine": [15,0,0], "right_hip": [80,0,0], "right_knee": [110,0,0], "left_hip": [80,0,0], "left_knee": [110,0,0], "right_shoulder": [0,0,60], "left_shoulder": [0,0,-60]}},
    {"frame": 47, "rotations": {"spine": [10,0,0], "right_hip": [60,0,0], "right_knee": [90,0,0], "left_hip": [60,0,0], "left_knee": [90,0,0], "right_shoulder": [0,0,45], "left_shoulder": [0,0,-45]}}
  ]
}"""


class MotionDirectorNode:
    """Generate keyframe poses from a motion description using an LLM.

    Outputs joint rotations rendered as 3D Anny body meshes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_description": ("STRING", {"default": "Person waving hello", "multiline": True}),
                "num_keyframes": ("INT", {"default": 4, "min": 2, "max": 16}),
                "total_frames": ("INT", {"default": 48, "min": 8, "max": 240}),
                "base_url": ("STRING", {"default": "https://openrouter.ai/api/v1"}),
                "model": ("STRING", {"default": "deepseek/deepseek-v4-flash"}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
            },
            "optional": {
                "max_retries": ("INT", {"default": 2, "min": 0, "max": 5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    OUTPUT_NAMES = ("pose_batch", "keyframes_json", "rotations_json")
    FUNCTION = "generate"
    CATEGORY = "pose_director"
    DESCRIPTION = "Generate keyframe poses from a motion description. Outputs 3D body mesh renders."

    def generate(self, motion_description, num_keyframes, total_frames,
                 base_url, model, width, height, max_retries=2):
        # Calculate frame spacing
        frame_interval = total_frames // (num_keyframes - 1) if num_keyframes > 1 else total_frames
        target_frames = [i * frame_interval for i in range(num_keyframes)]
        target_frames[-1] = total_frames - 1

        # Build user prompt
        user_msg = _build_user_prompt(motion_description, num_keyframes, total_frames, target_frames)

        # Generate with validation + retry
        keyframes = self._generate_with_retry(
            base_url, model, user_msg, max_retries,
        )

        # Sanitize: clamp to anatomical limits
        keyframes = sanitize_rotations(keyframes)

        # Fill in missing joints from previous keyframe (for partial rotations)
        keyframes = _fill_missing_joints(keyframes)

        # Temporal consistency check (log warnings only)
        consistency = validate_rotation_consistency(keyframes)
        if not consistency["is_consistent"]:
            for flag in consistency["flagged"]:
                print(f"[MotionDirector] Consistency: {flag['joint']} {flag['axis']}-axis "
                      f"jumped {flag['delta']:.1f}° between frames {flag['frame_pair']} "
                      f"(max allowed: {flag['max_allowed']:.1f}°)")

        # Build output data
        keyframes_data = {"keyframes": keyframes}
        keyframes_json = json.dumps(keyframes_data)
        all_rotations = [kf["rotations"] for kf in keyframes]
        rotations_json = json.dumps(all_rotations)

        # Render each keyframe as a 3D body mesh
        from .body_mesh_renderer import BodyMeshRendererNode
        mesh_renderer = BodyMeshRendererNode()
        images = []
        for kf in keyframes:
            rots_json = json.dumps(kf["rotations"])
            (img_tensor,) = mesh_renderer.render(rots_json, width, height)
            images.append(img_tensor)

        batch = torch.cat(images, dim=0)

        return (batch, keyframes_json, rotations_json)

    def _generate_with_retry(self, base_url, model, user_msg, max_retries):
        """Call LLM, validate output, retry with error feedback if needed."""
        last_error = ""
        for attempt in range(max_retries + 1):
            # Build prompt with error feedback from previous attempt
            prompt = user_msg
            if last_error:
                prompt += f"\n\nYour previous attempt had these errors:\n{last_error}\nPlease fix and try again."

            response = call_llm(base_url, "", model, SYSTEM_PROMPT, prompt)

            try:
                keyframes_data = parse_llm_json(response)
                if isinstance(keyframes_data, list):
                    keyframes_data = keyframes_data[0] if keyframes_data else {}
                keyframes = keyframes_data["keyframes"]
            except (KeyError, TypeError, IndexError, ValueError) as e:
                last_error = f"Could not parse keyframes from response: {e}"
                print(f"[MotionDirector] Attempt {attempt + 1} parse error: {e}")
                continue

            # Validate structure
            valid, issues = validate_rotations(keyframes)
            if valid:
                return keyframes

            # Build error feedback for retry
            errors = [i for i in issues if i.severity == "error"]
            last_error = "\n".join(i.message for i in errors)
            print(f"[MotionDirector] Attempt {attempt + 1} validation failed: {last_error}")

        # All retries exhausted — return last parsed keyframes (will be sanitized)
        print(f"[MotionDirector] All {max_retries + 1} attempts exhausted, using best effort")
        return keyframes


def _build_user_prompt(motion_description, num_keyframes, total_frames, target_frames):
    """Build the user-facing prompt for the LLM."""
    return f"""Generate keyframes for this motion:

Motion: {motion_description}
Total frames: {total_frames}
Keyframes needed: {num_keyframes}
Target frame numbers: {target_frames}

Frame 0 MUST be the neutral T-pose (all rotations zero).
For frames after 0, ONLY include joints that change from the previous keyframe.

Respond with ONLY the JSON keyframes object."""


def _fill_missing_joints(keyframes: list[dict]) -> list[dict]:
    """Fill in joints missing from later keyframes using values from previous keyframe.

    The LLM is instructed to only output changed joints. This fills in the rest
    so every keyframe has all 15 joints for BodyMeshRenderer.
    """
    if not keyframes:
        return keyframes

    # Ensure frame 0 has all joints
    for joint in REQUIRED_JOINTS:
        if joint not in keyframes[0]["rotations"]:
            keyframes[0]["rotations"][joint] = [0, 0, 0]

    # Cascade forward
    for i in range(1, len(keyframes)):
        prev_rots = keyframes[i - 1]["rotations"]
        curr_rots = keyframes[i]["rotations"]
        for joint in REQUIRED_JOINTS:
            if joint not in curr_rots:
                curr_rots[joint] = list(prev_rots[joint])

    return keyframes
