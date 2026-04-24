"""Skeleton validation for anatomical plausibility and keyframe consistency."""

import math
from typing import NamedTuple

from .openpose import SKELETON_BONES, KEYPOINT_NAMES


class ValidationIssue(NamedTuple):
    severity: str       # "warning" or "error"
    category: str       # "bounds", "proportion", "angle", "balance", "confidence"
    message: str
    keypoint_indices: list[int]


# Named bone pairs for proportional checks (joint_a, joint_b)
LIMB_PAIRS = {
    "right_upper_arm": (2, 3),   # RShoulder -> RElbow
    "right_forearm": (3, 4),     # RElbow -> RWrist
    "left_upper_arm": (5, 6),    # LShoulder -> LElbow
    "left_forearm": (6, 7),      # LElbow -> LWrist
    "right_thigh": (8, 9),       # RHip -> RKnee
    "right_shin": (9, 10),       # RKnee -> RAnkle
    "left_thigh": (11, 12),      # LHip -> LKnee
    "left_shin": (12, 13),       # LKnee -> LAnkle
}

# Joint angle checks: (vertex, point_a, point_b, max_angle_degrees)
# The angle is at the vertex joint, formed by rays to point_a and point_b
JOINT_ANGLE_LIMITS = {
    "right_elbow": (3, 2, 4, 170),    # RElbow, RShoulder, RWrist
    "left_elbow": (6, 5, 7, 170),     # LElbow, LShoulder, LWrist
    "right_knee": (9, 8, 10, 175),    # RKnee, RHip, RAnkle
    "left_knee": (12, 11, 13, 175),   # LKnee, LHip, LAnkle
}

# Proportion constraints: (bone_a_name, bone_b_name, max_ratio)
# bone_a length should not exceed bone_b length * max_ratio
PROPORTION_CONSTRAINTS = [
    ("right_shin", "right_thigh", 1.3),      # Shin shouldn't be much longer than thigh
    ("left_shin", "left_thigh", 1.3),
    ("right_forearm", "right_upper_arm", 1.4),
    ("left_forearm", "left_upper_arm", 1.4),
]


def validate_keypoints(keypoints: list, width: int, height: int) -> tuple[bool, list[ValidationIssue]]:
    """Validate a single set of COCO-18 keypoints for anatomical plausibility.

    Args:
        keypoints: Flat array of 54 floats [x, y, conf, x, y, conf, ...].
        width: Canvas width in pixels.
        height: Canvas height in pixels.
    Returns:
        (is_valid, issues_list) where is_valid=True if no "error" severity issues.
    """
    issues = []
    issues.extend(_check_bounds(keypoints, width, height))
    issues.extend(_check_limb_proportions(keypoints))
    issues.extend(_check_joint_angles(keypoints))
    issues.extend(_check_balance(keypoints, width))
    issues.extend(_check_confidence_gaps(keypoints))
    issues.extend(_check_joint_overlap(keypoints, min(width, height)))

    is_valid = not any(i.severity == "error" for i in issues)
    return is_valid, issues


def validate_keyframe_consistency(keyframes_list: list[list]) -> dict:
    """Check limb lengths stay consistent across keyframes (shimmer detection).

    Args:
        keyframes_list: List of keypoint arrays, one per keyframe.
    Returns:
        {
            "is_consistent": bool,
            "bone_variance": {bone_name: {"variance": float, "lengths": [float]}},
            "flagged_bones": [str],
            "max_variance": float,
        }
    """
    if len(keyframes_list) < 2:
        return {"is_consistent": True, "bone_variance": {}, "flagged_bones": [], "max_variance": 0.0}

    bone_data = {}
    for name, (idx_a, idx_b) in LIMB_PAIRS.items():
        lengths = []
        for kf in keyframes_list:
            if len(kf) < max(idx_a, idx_b) * 3 + 3:
                continue
            length = _bone_length(kf, idx_a, idx_b)
            if length > 0:
                lengths.append(length)

        if len(lengths) >= 2:
            mean = sum(lengths) / len(lengths)
            if mean > 0:
                variance = max(abs(l - mean) / mean for l in lengths)
            else:
                variance = 0.0
            bone_data[name] = {"variance": variance, "lengths": lengths}

    flagged = [name for name, data in bone_data.items() if data["variance"] > 0.3]
    max_var = max((d["variance"] for d in bone_data.values()), default=0.0)

    return {
        "is_consistent": len(flagged) == 0,
        "bone_variance": bone_data,
        "flagged_bones": flagged,
        "max_variance": max_var,
    }


def _bone_length(keypoints: list, idx_a: int, idx_b: int) -> float:
    """Euclidean distance between two keypoints."""
    ax, ay = keypoints[idx_a * 3], keypoints[idx_a * 3 + 1]
    bx, by = keypoints[idx_b * 3], keypoints[idx_b * 3 + 1]
    # Skip if either point has zero confidence
    if keypoints[idx_a * 3 + 2] < 0.1 or keypoints[idx_b * 3 + 2] < 0.1:
        return 0.0
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _get_point(keypoints: list, idx: int) -> tuple[float, float, float]:
    """Get (x, y, confidence) for a keypoint index."""
    base = idx * 3
    return keypoints[base], keypoints[base + 1], keypoints[base + 2]


def _check_bounds(keypoints: list, width: int, height: int) -> list[ValidationIssue]:
    """Check all coordinates are within canvas bounds (with 10% margin)."""
    issues = []
    margin_x, margin_y = width * 0.1, height * 0.1
    for i in range(0, len(keypoints), 3):
        x, y, conf = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if conf < 0.1:
            continue
        idx = i // 3
        if x < -margin_x or x > width + margin_x:
            issues.append(ValidationIssue("warning", "bounds",
                f"{KEYPOINT_NAMES.get(idx, idx)} x={x:.0f} outside canvas [{0}, {width}]",
                [idx]))
        if y < -margin_y or y > height + margin_y:
            issues.append(ValidationIssue("warning", "bounds",
                f"{KEYPOINT_NAMES.get(idx, idx)} y={y:.0f} outside canvas [{0}, {height}]",
                [idx]))
    return issues


def _check_limb_proportions(keypoints: list) -> list[ValidationIssue]:
    """Check limb length ratios are anatomically plausible."""
    issues = []
    lengths = {}
    for name, (idx_a, idx_b) in LIMB_PAIRS.items():
        lengths[name] = _bone_length(keypoints, idx_a, idx_b)

    for bone_a, bone_b, max_ratio in PROPORTION_CONSTRAINTS:
        la, lb = lengths.get(bone_a, 0), lengths.get(bone_b, 0)
        if la == 0 or lb == 0:
            continue
        ratio = la / lb
        if ratio > max_ratio:
            issues.append(ValidationIssue("warning", "proportion",
                f"{bone_a} ({la:.0f}px) is {ratio:.1f}x {bone_b} ({lb:.0f}px), exceeds {max_ratio}",
                []))

    # Check for zero-length limbs next to non-zero limbs
    for name, length in lengths.items():
        if length == 0:
            idx_a, idx_b = LIMB_PAIRS[name]
            if keypoints[idx_a * 3 + 2] > 0.5 and keypoints[idx_b * 3 + 2] > 0.5:
                issues.append(ValidationIssue("error", "proportion",
                    f"{name} is zero-length but both joints have high confidence",
                    [idx_a, idx_b]))
    return issues


def _check_joint_angles(keypoints: list) -> list[ValidationIssue]:
    """Check joint angles are within human range using dot product."""
    issues = []
    for name, (vertex, pt_a, pt_b, max_angle) in JOINT_ANGLE_LIMITS.items():
        vx, vy, vc = _get_point(keypoints, vertex)
        ax, ay, ac = _get_point(keypoints, pt_a)
        bx, by, bc = _get_point(keypoints, pt_b)
        if vc < 0.1 or ac < 0.1 or bc < 0.1:
            continue
        # Vectors from vertex to each point
        va_x, va_y = ax - vx, ay - vy
        vb_x, vb_y = bx - vx, by - vy
        len_a = math.sqrt(va_x ** 2 + va_y ** 2)
        len_b = math.sqrt(vb_x ** 2 + vb_y ** 2)
        if len_a < 1e-6 or len_b < 1e-6:
            continue
        cos_angle = (va_x * vb_x + va_y * vb_y) / (len_a * len_b)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for floating point
        angle_deg = math.degrees(math.acos(cos_angle))
        if angle_deg > max_angle:
            issues.append(ValidationIssue("warning", "angle",
                f"{name} angle is {angle_deg:.0f}°, exceeds max {max_angle}°",
                [vertex, pt_a, pt_b]))
    return issues


def _check_balance(keypoints: list, width: int) -> list[ValidationIssue]:
    """Check body center of gravity is roughly between the feet."""
    # Use average of shoulders and hips as torso center
    lsho_x, _, lsho_c = _get_point(keypoints, 5)  # LShoulder
    rsho_x, _, rsho_c = _get_point(keypoints, 2)  # RShoulder
    lhip_x, _, lhip_c = _get_point(keypoints, 11)  # LHip
    rhip_x, _, rhip_c = _get_point(keypoints, 8)   # RHip
    lankle_x, _, lankle_c = _get_point(keypoints, 13)  # LAnkle
    rankle_x, _, rankle_c = _get_point(keypoints, 10)   # RAnkle

    points = [(lsho_x, lsho_c), (rsho_x, rsho_c), (lhip_x, lhip_c), (rhip_x, rhip_c)]
    valid = [(x, c) for x, c in points if c > 0.1]
    if len(valid) < 2:
        return []
    torso_cx = sum(x for x, _ in valid) / len(valid)

    # Check if CoG is between ankles (with generous margin)
    if lankle_c > 0.1 and rankle_c > 0.1:
        foot_min = min(lankle_x, rankle_x)
        foot_max = max(lankle_x, rankle_x)
        stance = foot_max - foot_min
        margin = max(stance * 1.5, width * 0.15)  # generous margin
        if torso_cx < foot_min - margin or torso_cx > foot_max + margin:
            return [ValidationIssue("warning", "balance",
                f"Center of gravity ({torso_cx:.0f}) far outside foot stance [{foot_min:.0f}, {foot_max:.0f}]",
                [5, 2, 11, 8, 13, 10])]
    return []


def _check_confidence_gaps(keypoints: list) -> list[ValidationIssue]:
    """Flag zero-confidence joints between high-confidence joints on the same bone."""
    issues = []
    for idx_a, idx_b in SKELETON_BONES:
        ca = keypoints[idx_a * 3 + 2]
        cb = keypoints[idx_b * 3 + 2]
        if ca > 0.5 and cb < 0.1:
            issues.append(ValidationIssue("warning", "confidence",
                f"{KEYPOINT_NAMES.get(idx_b, idx_b)} has low confidence ({cb:.2f}) "
                f"but connected {KEYPOINT_NAMES.get(idx_a, idx_a)} is confident ({ca:.2f})",
                [idx_a, idx_b]))
        elif cb > 0.5 and ca < 0.1:
            issues.append(ValidationIssue("warning", "confidence",
                f"{KEYPOINT_NAMES.get(idx_a, idx_a)} has low confidence ({ca:.2f}) "
                f"but connected {KEYPOINT_NAMES.get(idx_b, idx_b)} is confident ({cb:.2f})",
                [idx_a, idx_b]))
    return issues


# Minimum pixel distance between connected joints (fraction of canvas min dimension)
_OVERLAP_THRESHOLD = 0.02  # 2% of canvas min dimension

# Bones that are naturally short — skip sanitization for these
_SHORT_BONES = frozenset([
    (0, 1),    # Nose → Neck
    (0, 14),   # Nose → REye
    (14, 16),  # REye → REar
    (0, 15),   # Nose → LEye
    (15, 17),  # LEye → LEar
])


def _check_joint_overlap(keypoints: list, canvas_min_dim: int) -> list[ValidationIssue]:
    """Flag bone-connected joints that are too close together (overlap/ghosting).

    This catches the "double imagery" artifact where the LLM places two
    connected joints at nearly the same pixel, causing VNCCS to produce
    ghosting/distortion. Only checks limb bones (skips short head/face bones).
    """
    min_dist = canvas_min_dim * _OVERLAP_THRESHOLD
    issues = []
    for idx_a, idx_b in SKELETON_BONES:
        if (idx_a, idx_b) in _SHORT_BONES:
            continue
        ca = keypoints[idx_a * 3 + 2]
        cb = keypoints[idx_b * 3 + 2]
        if ca < 0.1 or cb < 0.1:
            continue
        ax, ay = keypoints[idx_a * 3], keypoints[idx_a * 3 + 1]
        bx, by = keypoints[idx_b * 3], keypoints[idx_b * 3 + 1]
        dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        if dist < min_dist:
            name_a = KEYPOINT_NAMES.get(idx_a, idx_a)
            name_b = KEYPOINT_NAMES.get(idx_b, idx_b)
            issues.append(ValidationIssue("warning", "overlap",
                f"{name_a} and {name_b} overlap (dist={dist:.1f}px < min={min_dist:.1f}px)",
                [idx_a, idx_b]))
    return issues


def sanitize_keypoints(keypoints: list, width: int, height: int) -> list:
    """Push overlapping connected joints apart to prevent VNCCS ghosting.

    For each bone in SKELETON_BONES, if the two endpoints are closer than
    min_bone_length, pushes them apart along the bone axis. Preserves the
    midpoint and total extent — only redistributes the endpoints.

    Returns a new keypoint array (does not mutate input).
    """
    min_bone_length = min(width, height) * _OVERLAP_THRESHOLD * 2
    result = list(keypoints)

    for idx_a, idx_b in SKELETON_BONES:
        if (idx_a, idx_b) in _SHORT_BONES:
            continue
        base_a, base_b = idx_a * 3, idx_b * 3
        ca, cb = result[base_a + 2], result[base_b + 2]
        if ca < 0.1 or cb < 0.1:
            continue

        ax, ay = result[base_a], result[base_a + 1]
        bx, by = result[base_b], result[base_b + 1]
        dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

        if dist < min_bone_length:
            # Direction vector (or default downward if coincident)
            if dist > 0:
                dx, dy = (bx - ax) / dist, (by - ay) / dist
            else:
                dx, dy = 0, 1  # push vertically
            # Push apart symmetrically around midpoint
            push = min_bone_length / 2
            mid_x, mid_y = (ax + bx) / 2, (ay + by) / 2
            result[base_a] = mid_x - dx * push
            result[base_a + 1] = mid_y - dy * push
            result[base_b] = mid_x + dx * push
            result[base_b + 1] = mid_y + dy * push

    return result
