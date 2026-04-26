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
from ..utils.pose_evaluator import (
    evaluate_keyframe_renders,
    build_correction_prompt,
    is_evaluation_passing,
)

SYSTEM_PROMPT = """\
You are a motion keyframe generator for a 3D humanoid body model (Anny, Z-up convention).
You receive a text description of a motion and output keyframe poses as joint rotations.

## Coordinate System (Z-up)
- All rotations are in **degrees** relative to rest pose (arms angled ~50° below horizontal).
- Rotation order: **XYZ intrinsic** (equivalent to Z-Y-X extrinsic). Applied as: first X, then Y, then Z.
- **Z-axis:** Arm raise/lower for shoulders. Z = +90 raises RIGHT arm UP. Z = -90 raises LEFT arm UP.
- **Y-axis:** Horizontal rotation (yaw). Y = +90 rotates counter-clockwise seen from above.
- **X-axis:** Horizontal sweep / complex arm movement for shoulders.

## Per-Joint Axis Effects (CRITICAL — memorize these)
Shoulder: **Z** raises/lowers arm. right_shoulder Z=+90 = arm straight UP. left_shoulder Z=-90 = arm straight UP.
Elbow: **Z** bends forearm. Z=-90 (right) or Z=+90 (left) = forearm bends toward upper arm.
Knee: **X** bends lower leg backward. X=+90 = knee bent 90 degrees.
Spine: **X** leans forward (+) or backward (-). **Y** rotates torso.

## Rotation Hierarchy (IMPORTANT)
Rotations cascade down the skeleton. Parent joint rotations affect child joints.
- To turn the ENTIRE character, change only the **spine Y-rotation**.
- Arm rotations are LOCAL to the shoulder. If the shoulder is rotated, the elbow/wrist follow automatically.

## Mirroring (Left vs Right)
Left and right limbs use OPPOSITE Z-signs for the same motion:
- Right shoulder Z=+90 raises RIGHT arm up. Left shoulder Z=-90 raises LEFT arm up.
- Right elbow Z=-90 bends right elbow. Left elbow Z=+90 bends left elbow.
- To raise BOTH arms: right_shoulder [0,0,90], left_shoulder [0,0,-90].

## Reference Poses (Anchors — use these to calibrate your spatial reasoning)
- **Right arm straight UP:** right_shoulder Z = +90
- **Right arm pointing DOWN/forward:** right_shoulder X = +90
- **Left arm straight UP:** left_shoulder Z = -90
- **Right elbow bent 90°:** right_elbow Z = -90
- **Left elbow bent 90°:** left_elbow Z = +90
- **Right knee bent 90°:** right_knee X = +90
- **Character facing right:** spine Y = -90
- **Leaning forward 45°:** spine X = +45

## Available Joints (exactly 15)
spine, neck, head,
right_shoulder, right_elbow, right_wrist,
left_shoulder, left_elbow, left_wrist,
right_hip, right_knee, right_ankle,
left_hip, left_knee, left_ankle

## Rules
- Shoulders use **Z-axis** for vertical arm movement — this is the most important axis!
- Elbows only bend in ONE direction: negative Z for right, positive Z for left
- Knees only bend BACKWARD: positive X only (range 0 to 140)
- Keep rotations anatomically plausible (no 360° spins, no dislocated joints)
- Smooth transitions between keyframes (no teleportation)
- Frame 0 is always the neutral rest pose (all zeros)
- Generate DRAMATIC, visible poses — use large rotation values (60-120°), not tiny adjustments

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
    {"frame": 16, "rotations": {"right_shoulder": [0,0,90], "right_elbow": [0,0,-90]}},
    {"frame": 32, "rotations": {"right_shoulder": [0,0,90], "right_elbow": [0,0,-90], "right_wrist": [-30,0,0]}},
    {"frame": 47, "rotations": {"right_shoulder": [0,0,90], "right_elbow": [0,0,-90], "right_wrist": [30,0,0]}}
  ]
}

## Example: "Raise both arms up"
{
  "keyframes": [
    {"frame": 0, "rotations": {"spine": [0,0,0], "neck": [0,0,0], "head": [0,0,0], "right_shoulder": [0,0,0], "right_elbow": [0,0,0], "right_wrist": [0,0,0], "left_shoulder": [0,0,0], "left_elbow": [0,0,0], "left_wrist": [0,0,0], "right_hip": [0,0,0], "right_knee": [0,0,0], "right_ankle": [0,0,0], "left_hip": [0,0,0], "left_knee": [0,0,0], "left_ankle": [0,0,0]}},
    {"frame": 16, "rotations": {"right_shoulder": [0,0,90], "left_shoulder": [0,0,-90]}},
    {"frame": 31, "rotations": {"right_shoulder": [0,0,90], "left_shoulder": [0,0,-90]}}
  ]
}

## Example: "Squat down"
{
  "keyframes": [
    {"frame": 0, "rotations": {"spine": [0,0,0], "neck": [0,0,0], "head": [0,0,0], "right_shoulder": [0,0,0], "right_elbow": [0,0,0], "right_wrist": [0,0,0], "left_shoulder": [0,0,0], "left_elbow": [0,0,0], "left_wrist": [0,0,0], "right_hip": [0,0,0], "right_knee": [0,0,0], "right_ankle": [0,0,0], "left_hip": [0,0,0], "left_knee": [0,0,0], "left_ankle": [0,0,0]}},
    {"frame": 16, "rotations": {"spine": [10,0,0], "right_hip": [60,0,0], "right_knee": [90,0,0], "left_hip": [60,0,0], "left_knee": [90,0,0]}},
    {"frame": 32, "rotations": {"spine": [15,0,0], "right_hip": [80,0,0], "right_knee": [110,0,0], "left_hip": [80,0,0], "left_knee": [110,0,0]}},
    {"frame": 47, "rotations": {"spine": [10,0,0], "right_hip": [60,0,0], "right_knee": [90,0,0], "left_hip": [60,0,0], "left_knee": [90,0,0]}}
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
                "model": ("STRING", {"default": "deepseek/deepseek-v4-pro"}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
            },
            "optional": {
                "max_retries": ("INT", {"default": 2, "min": 0, "max": 5}),
                "vision_feedback_rounds": ("INT", {"default": 0, "min": 0, "max": 5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    OUTPUT_NAMES = ("pose_batch", "keyframes_json", "rotations_json")
    FUNCTION = "generate"
    CATEGORY = "pose_director"
    DESCRIPTION = "Generate keyframe poses from a motion description. Outputs 3D body mesh renders."

    def generate(self, motion_description, num_keyframes, total_frames,
                 base_url, model, width, height, max_retries=2,
                 vision_feedback_rounds=0):
        # Calculate frame spacing
        frame_interval = total_frames // (num_keyframes - 1) if num_keyframes > 1 else total_frames
        target_frames = [i * frame_interval for i in range(num_keyframes)]
        target_frames[-1] = total_frames - 1

        print(f"[MotionDirector] === GENERATE ===")
        print(f"[MotionDirector] Motion: {motion_description}")
        print(f"[MotionDirector] Model: {model} via {base_url}")
        print(f"[MotionDirector] Frames: {num_keyframes} keyframes over {total_frames} total")
        print(f"[MotionDirector] Target frames: {target_frames}")
        print(f"[MotionDirector] Vision feedback rounds: {vision_feedback_rounds}")

        # Build user prompt
        user_msg = _build_user_prompt(motion_description, num_keyframes, total_frames, target_frames)

        # Vision feedback loop
        vision_feedback = ""
        for round_idx in range(vision_feedback_rounds + 1):
            # Generate with validation + retry
            prompt = user_msg
            if vision_feedback:
                prompt += f"\n\n{vision_feedback}"

            keyframes = self._generate_with_retry(
                base_url, model, prompt, max_retries,
            )

            # DEBUG: Log pre-sanitize keyframes
            print(f"[MotionDirector] Pre-sanitize keyframes:")
            for kf in keyframes:
                changed = {j: a for j, a in kf["rotations"].items()
                           if any(abs(v) > 0.1 for v in a)}
                print(f"  Frame {kf['frame']}: {changed if changed else '(all zero)'}")

            # Sanitize: clamp to anatomical limits
            keyframes = sanitize_rotations(keyframes)

            # DEBUG: Log post-sanitize (check if clamping changed anything)
            print(f"[MotionDirector] Post-sanitize keyframes:")
            for kf in keyframes:
                changed = {j: a for j, a in kf["rotations"].items()
                           if any(abs(v) > 0.1 for v in a)}
                print(f"  Frame {kf['frame']}: {changed if changed else '(all zero)'}")

            # Fill in missing joints from previous keyframe (for partial rotations)
            keyframes = _fill_missing_joints(keyframes)

            # DEBUG: Log final filled keyframes
            print(f"[MotionDirector] Final keyframes (after fill):")
            for kf in keyframes:
                changed = {j: a for j, a in kf["rotations"].items()
                           if any(abs(v) > 0.1 for v in a)}
                print(f"  Frame {kf['frame']}: {changed if changed else '(all zero)'}")

            # Temporal consistency check (log warnings only)
            consistency = validate_rotation_consistency(keyframes)
            if not consistency["is_consistent"]:
                for flag in consistency["flagged"]:
                    print(f"[MotionDirector] Consistency: {flag['joint']} {flag['axis']}-axis "
                          f"jumped {flag['delta']:.1f}° between frames {flag['frame_pair']} "
                          f"(max allowed: {flag['max_allowed']:.1f}°)")

            # If no vision feedback requested, we're done
            if vision_feedback_rounds == 0:
                break

            # Render keyframes for vision evaluation
            from .body_mesh_renderer import BodyMeshRendererNode
            mesh_renderer = BodyMeshRendererNode()
            eval_images = []
            for kf in keyframes:
                rots_json = json.dumps(kf["rotations"])
                (img_tensor,) = mesh_renderer.render(rots_json, width, height)
                eval_images.append(img_tensor[0])  # Remove batch dim for vision API

            # Evaluate rendered poses — include rotation data for cross-checking
            print(f"[MotionDirector] Vision feedback round {round_idx + 1}/{vision_feedback_rounds}")
            frame_numbers = [kf["frame"] for kf in keyframes]
            evaluation = evaluate_keyframe_renders(
                images=eval_images,
                motion_description=motion_description,
                frame_numbers=frame_numbers,
                keyframes=keyframes,
            )

            print(f"[MotionDirector] Vision assessment: {evaluation.get('overall_assessment', 'unknown')}")
            problems = evaluation.get("key_problems", [])
            for p in problems:
                print(f"[MotionDirector]   Problem: {p}")

            if is_evaluation_passing(evaluation):
                print(f"[MotionDirector] Vision evaluation PASSED!")
                break

            # Build correction feedback for next round
            vision_feedback = build_correction_prompt(evaluation, keyframes)
            print(f"[MotionDirector] Applying corrections for next round...")

        # Build output data
        keyframes_data = {"keyframes": keyframes}
        keyframes_json = json.dumps(keyframes_data)
        all_rotations = [kf["rotations"] for kf in keyframes]
        rotations_json = json.dumps(all_rotations)

        # Render each keyframe as a 3D body mesh (final render)
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

            # DEBUG: Log raw LLM response
            print(f"[MotionDirector] Attempt {attempt + 1} raw response ({len(response)} chars):")
            print(f"[MotionDirector] {response[:800]}")

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
