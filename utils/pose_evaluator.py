"""Vision-based pose evaluation for self-correcting keyframe generation.

Evaluates rendered body mesh images against the intended motion description
and returns structured feedback for the LLM to correct its rotations.

Architecture:
    1. Render keyframe poses (BodyMeshRenderer)
    2. Send renders to vision model (Gemini Flash / Moondream-2 / Florence-2)
    3. Compare visual description to intended motion
    4. Return pass/fail + specific correction guidance
"""

import json
import base64
import io
from typing import Optional

from PIL import Image
import numpy as np

from .api_client import call_llm_vision, resolve_api_key

EVAL_SYSTEM_PROMPT = """\
You are a 3D pose evaluation expert. You analyze rendered body mesh images alongside
their rotation data and determine whether the pose matches the intended motion description.

The images show a grey 3D humanoid model (Anny) rendered from a front-facing camera.
The model starts in T-pose (arms outstretched horizontally). All rotations are
relative to this T-pose, in degrees, XYZ Euler order.

## Joint axis effects (for interpreting rotation values):
- Shoulder: Z raises/lowers arm. Z=+90 = RIGHT arm UP, Z=-90 = LEFT arm UP
- Elbow: Z bends forearm. Z=-90 (right) or Z=+90 (left) = 90° bend
- Knee: X bends backward. X=+90 = knee bent 90°
- Spine: X leans forward (+), Y rotates torso

For each image, you receive:
1. The rendered image
2. The rotation values used to generate it
3. The frame number and motion context

Your job:
1. Describe what pose the model is ACTUALLY in based on the image
2. Cross-check: do the rotation values match what you see? (e.g., right_shoulder Z=+90 should show arm raised)
3. Does this pose contribute to the intended motion?
4. If not, what specific rotation changes are needed?

Be brutally honest. If all rotations are near-zero and the model looks like T-pose, say "STILL IN T-POSE".
If the rotation values don't match what you see in the render, flag that too.

Respond in JSON format:
{
  "poses": [
    {
      "frame": <frame_number>,
      "description": "<what you see in the image>",
      "rotation_analysis": "<do the rotation values match the visual?>",
      "matches_intent": true/false,
      "issues": ["<specific problem 1>", ...],
      "corrections": {"<joint_name>": "<what to change, e.g. 'Set X to -90'>", ...}
    }
  ],
  "overall_assessment": "<pass/fail>",
  "key_problems": ["<summary of main issues>"]
}"""


def evaluate_keyframe_renders(
    images: list,  # List of ComfyUI tensors (H, W, 3) float32
    motion_description: str,
    frame_numbers: list[int],
    keyframes: list[dict] = None,  # Rotation data per keyframe
    base_url: str = "",
    api_key: str = "",
    model: str = "",
) -> dict:
    """Evaluate rendered keyframe images against the intended motion.

    Args:
        images: List of rendered body mesh images (ComfyUI tensor format).
        motion_description: The intended motion (e.g., "character waves right hand").
        frame_numbers: Frame number for each image.
        keyframes: Optional rotation dicts per keyframe, for cross-checking.
        base_url: Vision API base URL (defaults to Gemini).
        api_key: API key (auto-resolved if empty).
        model: Vision model name (defaults to Gemini Flash).

    Returns:
        Evaluation dict with per-frame assessments and corrections.
    """
    # Default to Gemini Flash (free, fast, zero VRAM)
    if not base_url:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    if not model:
        model = "models/gemini-2.5-flash"

    # Build rotation data section for the prompt
    rotation_section = ""
    if keyframes:
        rotation_lines = []
        for i, kf in enumerate(keyframes):
            if i >= len(images):
                break
            rots = kf.get("rotations", {})
            # Only show non-zero rotations for readability
            changed = {j: a for j, a in rots.items() if any(abs(v) > 0.1 for v in a)}
            frame = kf.get("frame", frame_numbers[i] if i < len(frame_numbers) else i)
            if changed:
                rotation_lines.append(f"  Frame {frame} rotations: {json.dumps(changed)}")
            else:
                rotation_lines.append(f"  Frame {frame} rotations: (all zero / T-pose)")
        rotation_section = "\n## Rotation data used for each frame:\n" + "\n".join(rotation_lines) + "\n"

    # Build pose summary for context
    pose_summary = _summarize_intended_poses(motion_description, frame_numbers)

    prompt = f"""Analyze these {len(images)} rendered keyframe poses.

## Intended motion: "{motion_description}"

{pose_summary}

Frame numbers: {frame_numbers}
{rotation_section}
Frame 0 should be T-pose (neutral starting position).
Later frames should show progressive motion matching the description.

For each frame, describe what you see in the image AND cross-check against the rotation data.
If the rotation values don't match what you see visually (e.g., rotation says Z=+90 but arm is horizontal),
that indicates a rendering or interpretation issue.

Focus on: arm positions (raised? lowered? bent?), leg positions, spine rotation, head orientation.

If a frame shows no visible change from T-pose (arms still horizontal, legs straight),
mark it as "STILL IN T-POSE" and describe what SHOULD be different.

Respond with the JSON evaluation."""

    try:
        response = call_llm_vision(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=EVAL_SYSTEM_PROMPT,
            text_prompt=prompt,
            images=images,
            temperature=0.3,
        )

        # Parse the JSON response
        from .json_parser import parse_llm_json
        result = parse_llm_json(response)
        if isinstance(result, list):
            result = result[0] if result else {}
        return result

    except Exception as e:
        print(f"[PoseEvaluator] Vision evaluation failed: {e}")
        return {
            "poses": [],
            "overall_assessment": "error",
            "key_problems": [f"Evaluation failed: {e}"],
        }


def _summarize_intended_poses(motion_description: str, frame_numbers: list[int]) -> str:
    """Generate a brief summary of what each frame SHOULD show for the given motion."""
    n = len(frame_numbers)
    if n <= 1:
        return "Frame 0: T-pose starting position."

    lines = ["Expected pose progression:"]
    lines.append(f"  Frame {frame_numbers[0]}: T-pose (neutral start)")

    if n == 2:
        lines.append(f"  Frame {frame_numbers[1]}: Full pose for '{motion_description}'")
    elif n == 3:
        lines.append(f"  Frame {frame_numbers[1]}: Midpoint of '{motion_description}' (partial motion)")
        lines.append(f"  Frame {frame_numbers[2]}: Full pose / end of '{motion_description}'")
    else:
        lines.append(f"  Frame {frame_numbers[1]}: Beginning of motion (e.g., arm starting to raise)")
        for i in range(2, n - 1):
            lines.append(f"  Frame {frame_numbers[i]}: Motion in progress (mid-action)")
        lines.append(f"  Frame {frame_numbers[-1]}: Final pose / end of '{motion_description}'")

    return "\n".join(lines)


def build_correction_prompt(evaluation: dict, original_keyframes: list[dict]) -> str:
    """Build a correction prompt from the evaluation feedback.

    This gets appended to the user message for the next LLM attempt.
    """
    problems = evaluation.get("key_problems", [])
    if not problems:
        return ""

    lines = [
        "Your previous keyframes were evaluated by a vision model. PROBLEMS FOUND:",
        "",
    ]

    for problem in problems:
        lines.append(f"- {problem}")

    # Add per-frame corrections
    for pose_eval in evaluation.get("poses", []):
        if not pose_eval.get("matches_intent", True):
            frame = pose_eval.get("frame", "?")
            issues = pose_eval.get("issues", [])
            corrections = pose_eval.get("corrections", {})

            if issues:
                lines.append(f"\nFrame {frame} issues:")
                for issue in issues:
                    lines.append(f"  - {issue}")

            if corrections:
                lines.append(f"\nFrame {frame} corrections needed:")
                for joint, fix in corrections.items():
                    lines.append(f"  - {joint}: {fix}")

    lines.extend([
        "",
        "CRITICAL: Use LARGE rotation values (60-180 degrees). Small values produce invisible changes.",
        "Remember: right_shoulder Z=+90 raises right arm UP, left_shoulder Z=-90 raises left arm UP.",
        "Fix these issues and generate new keyframes.",
    ])

    return "\n".join(lines)


def is_evaluation_passing(evaluation: dict) -> bool:
    """Check if the evaluation indicates the poses match the intent."""
    assessment = evaluation.get("overall_assessment", "").lower()
    if assessment == "pass":
        return True
    if assessment == "fail":
        return False

    # Check individual frames
    poses = evaluation.get("poses", [])
    if not poses:
        return True  # No evaluation data, assume pass

    return all(p.get("matches_intent", True) for p in poses)
