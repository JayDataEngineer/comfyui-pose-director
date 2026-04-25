"""Rotation validation for anatomical plausibility and temporal consistency.

Validates LLM-generated joint rotation keyframes before they reach BodyMeshRenderer.
Mirrors the validation pattern from skeleton_validator.py but for Euler angles
instead of 2D keypoint coordinates.
"""

import copy
import math
from typing import NamedTuple


class ValidationIssue(NamedTuple):
    severity: str       # "warning" or "error"
    category: str       # "structure", "anatomy", "consistency"
    message: str
    joint: str = ""


# The 15 canonical joints the LLM must provide per keyframe.
REQUIRED_JOINTS = [
    "spine", "neck", "head",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
]

# Anatomical angle limits per joint per axis: (min_degrees, max_degrees).
# Based on human biomechanics — joints cannot bend beyond these ranges.
# Convention: XYZ Euler from T-pose (arms outstretched horizontally).
#   X: forward/backward tilt
#   Y: left/right rotation
#   Z: inward/outward rotation
JOINT_LIMITS: dict[str, dict[str, tuple[float, float]]] = {
    # Spine: limited flexibility in all axes
    "spine":       {"x": (-45, 45), "y": (-45, 45), "z": (-30, 30)},
    # Neck: moderate range
    "neck":        {"x": (-40, 40), "y": (-60, 60), "z": (-30, 30)},
    # Head: similar to neck but independent nod/tilt
    "head":        {"x": (-40, 40), "y": (-60, 60), "z": (-30, 30)},
    # Shoulders: wide range for arm movement
    "right_shoulder": {"x": (-180, 180), "y": (-180, 180), "z": (-30, 180)},
    "left_shoulder":  {"x": (-180, 180), "y": (-180, 180), "z": (-180, 30)},
    # Elbows: hinge joints — only bend one way
    # Right elbow: bends via negative Z (toward body)
    "right_elbow": {"x": (-10, 10), "y": (-10, 10), "z": (-145, 10)},
    # Left elbow: bends via positive Z (toward body)
    "left_elbow":  {"x": (-10, 10), "y": (-10, 10), "z": (-10, 145)},
    # Wrists: moderate flexibility
    "right_wrist": {"x": (-70, 70), "y": (-30, 30), "z": (-30, 30)},
    "left_wrist":  {"x": (-70, 70), "y": (-30, 30), "z": (-30, 30)},
    # Hips: moderate flexion/extension, limited twist
    "right_hip":   {"x": (-30, 90), "y": (-45, 45), "z": (-30, 30)},
    "left_hip":    {"x": (-30, 90), "y": (-45, 45), "z": (-30, 30)},
    # Knees: hinge joints — only bend backward (positive X)
    "right_knee":  {"x": (-5, 140), "y": (-10, 10), "z": (-10, 10)},
    "left_knee":   {"x": (-5, 140), "y": (-10, 10), "z": (-10, 10)},
    # Ankles: moderate range
    "right_ankle": {"x": (-40, 40), "y": (-20, 20), "z": (-15, 15)},
    "left_ankle":  {"x": (-40, 40), "y": (-20, 20), "z": (-15, 15)},
}

# Maximum angular velocity (degrees per frame) for temporal consistency.
# Prevents impossibly fast limb teleportation between consecutive keyframes.
MAX_ANGLE_PER_FRAME = 15.0


def validate_rotations(keyframes: list[dict]) -> tuple[bool, list[ValidationIssue]]:
    """Validate rotation keyframe structure and joint presence.

    Args:
        keyframes: List of {"frame": int, "rotations": {joint: [x, y, z]}} dicts.

    Returns:
        (is_valid, issues) where is_valid is True if no "error" severity issues.
    """
    issues: list[ValidationIssue] = []
    required_set = set(REQUIRED_JOINTS)

    for i, kf in enumerate(keyframes):
        if "rotations" not in kf:
            issues.append(ValidationIssue("error", "structure",
                f"Keyframe {i} (frame {kf.get('frame', '?')}) missing 'rotations' key"))
            continue

        rotations = kf["rotations"]
        present = set(rotations.keys())

        # Missing joints
        missing = required_set - present
        if missing:
            if i == 0:
                # Frame 0 must have all joints (it's the anchor)
                issues.append(ValidationIssue("error", "structure",
                    f"Keyframe 0 missing joints: {sorted(missing)}"))
            else:
                # Later keyframes can be partial (filled from previous frame)
                issues.append(ValidationIssue("warning", "structure",
                    f"Keyframe {i} missing joints (will inherit from prev): {sorted(missing)}"))

        # Hallucinated joints
        extra = present - required_set
        if extra:
            issues.append(ValidationIssue("warning", "structure",
                f"Keyframe {i} has unknown joints (will be ignored): {sorted(extra)}"))

        # Validate angle values
        for joint in REQUIRED_JOINTS:
            if joint not in rotations:
                continue
            angles = rotations[joint]
            if not isinstance(angles, (list, tuple)):
                issues.append(ValidationIssue("error", "structure",
                    f"Keyframe {i} '{joint}' angles must be a list, got {type(angles).__name__}",
                    joint))
                continue
            if len(angles) != 3:
                issues.append(ValidationIssue("error", "structure",
                    f"Keyframe {i} '{joint}' must have exactly 3 angles, got {len(angles)}",
                    joint))
                continue
            for axis_idx, val in enumerate(angles):
                if not isinstance(val, (int, float)):
                    issues.append(ValidationIssue("error", "structure",
                        f"Keyframe {i} '{joint}' axis {axis_idx} must be numeric, got {type(val).__name__}",
                        joint))

    is_valid = not any(i.severity == "error" for i in issues)
    return is_valid, issues


def sanitize_rotations(keyframes: list[dict]) -> list[dict]:
    """Clamp joint angles to anatomical limits.

    Returns a new list with clamped values. Does not mutate input.
    """
    result = []
    for kf in keyframes:
        new_kf = {"frame": kf["frame"], "rotations": {}}
        for joint, angles in kf["rotations"].items():
            if joint not in JOINT_LIMITS:
                continue
            limits = JOINT_LIMITS[joint]
            clamped = []
            for axis_name in ("x", "y", "z"):
                idx = {"x": 0, "y": 1, "z": 2}[axis_name]
                lo, hi = limits[axis_name]
                val = angles[idx] if idx < len(angles) else 0.0
                clamped.append(max(lo, min(hi, float(val))))
            new_kf["rotations"][joint] = clamped
        result.append(new_kf)
    return result


def validate_rotation_consistency(keyframes: list[dict]) -> dict:
    """Check temporal consistency between consecutive keyframes.

    Flags joints that change too fast (angular velocity exceeds limit).

    Args:
        keyframes: Sanitized rotation keyframes.

    Returns:
        {
            "is_consistent": bool,
            "flagged": [{"frame_pair": (a, b), "joint": str, "axis": str, "delta": float}],
            "max_delta": float,
        }
    """
    if len(keyframes) < 2:
        return {"is_consistent": True, "flagged": [], "max_delta": 0.0}

    flagged = []
    max_delta = 0.0

    for i in range(1, len(keyframes)):
        prev = keyframes[i - 1]
        curr = keyframes[i]
        frame_gap = max(curr["frame"] - prev["frame"], 1)
        max_allowed = MAX_ANGLE_PER_FRAME * frame_gap

        for joint in REQUIRED_JOINTS:
            if joint not in prev["rotations"] or joint not in curr["rotations"]:
                continue
            for axis_idx, axis_name in enumerate(("x", "y", "z")):
                prev_val = prev["rotations"][joint][axis_idx]
                curr_val = curr["rotations"][joint][axis_idx]
                delta = abs(curr_val - prev_val)
                max_delta = max(max_delta, delta)
                if delta > max_allowed:
                    flagged.append({
                        "frame_pair": (prev["frame"], curr["frame"]),
                        "joint": joint,
                        "axis": axis_name,
                        "delta": delta,
                        "max_allowed": max_allowed,
                    })

    return {
        "is_consistent": len(flagged) == 0,
        "flagged": flagged,
        "max_delta": max_delta,
    }
