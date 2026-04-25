"""SLERP keyframe interpolation for joint rotation animation.

Takes sparse keyframes (4-16) from MotionDirector and produces dense
per-frame rotation data via quaternion spherical linear interpolation.
Each of the 15 joints is interpolated independently.
"""

from scipy.spatial.transform import Rotation, Slerp


# The 15 controllable joints
JOINT_NAMES = [
    "spine", "neck", "head",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
]

# Identity rotation (T-pose) in Euler XYZ degrees
_IDENTITY = [0.0, 0.0, 0.0]


def interpolate_keyframes(
    keyframes: list[dict[str, list[float]]],
    frame_numbers: list[int],
    total_frames: int,
) -> list[dict[str, list[float]]]:
    """Interpolate between keyframe rotations using quaternion SLERP.

    Args:
        keyframes: List of rotation dicts, one per keyframe.
                   Each: {"spine": [x,y,z], "neck": [x,y,z], ...}
                   Missing joints default to T-pose (identity).
        frame_numbers: Frame index for each keyframe (e.g. [0, 12, 24, 36]).
        total_frames: Total number of output frames to generate.

    Returns:
        List of rotation dicts, one per frame (length = total_frames).
        Frames before first keyframe use first keyframe pose.
        Frames after last keyframe use last keyframe pose.
    """
    if not keyframes:
        return [_empty_frame() for _ in range(total_frames)]

    if len(keyframes) == 1:
        return [dict(keyframes[0]) for _ in range(total_frames)]

    if len(keyframes) != len(frame_numbers):
        raise ValueError(
            f"keyframes ({len(keyframes)}) and frame_numbers ({len(frame_numbers)}) "
            f"must have the same length"
        )

    # Fill missing joints with identity in each keyframe
    filled = [_fill_frame(kf) for kf in keyframes]

    # Collect all output frame indices
    all_frame_indices = list(range(total_frames))

    result = []
    for frame_idx in all_frame_indices:
        frame_rotations = {}
        for joint in JOINT_NAMES:
            euler_values = _interpolate_joint(
                joint, filled, frame_numbers, frame_idx
            )
            frame_rotations[joint] = euler_values
        result.append(frame_rotations)

    return result


def _empty_frame() -> dict[str, list[float]]:
    """Return a T-pose frame (all zeros)."""
    return {joint: list(_IDENTITY) for joint in JOINT_NAMES}


def _fill_frame(kf: dict[str, list[float]]) -> dict[str, list[float]]:
    """Fill missing joints with identity rotation."""
    filled = {}
    for joint in JOINT_NAMES:
        filled[joint] = list(kf.get(joint, _IDENTITY))
    return filled


def _interpolate_joint(
    joint: str,
    keyframes: list[dict[str, list[float]]],
    frame_numbers: list[int],
    target_frame: int,
) -> list[float]:
    """SLERP interpolate a single joint's rotation at a target frame.

    Args:
        joint: Joint name (e.g. "right_elbow").
        keyframes: Pre-filled rotation dicts (all joints present).
        frame_numbers: Frame index for each keyframe.
        target_frame: The frame to interpolate at.

    Returns:
        [x_deg, y_deg, z_deg] rotation in our Euler XYZ convention.
    """
    # Clamp to keyframe range
    if target_frame <= frame_numbers[0]:
        return keyframes[0][joint]
    if target_frame >= frame_numbers[-1]:
        return keyframes[-1][joint]

    # Find bracketing keyframes
    # keyframes are sorted by frame_number
    for i in range(len(frame_numbers) - 1):
        f0 = frame_numbers[i]
        f1 = frame_numbers[i + 1]
        if f0 <= target_frame <= f1:
            if f0 == f1:
                return keyframes[i][joint]

            # Build rotations for SLERP
            r0 = Rotation.from_euler("XYZ", keyframes[i][joint], degrees=True)
            r1 = Rotation.from_euler("XYZ", keyframes[i + 1][joint], degrees=True)

            # Handle case where rotations are identical (avoid SLERP issues)
            quat0 = r0.as_quat()
            quat1 = r1.as_quat()
            if (quat0 == quat1).all() or (quat0 == -quat1).all():
                return keyframes[i][joint]

            # SLERP between the two keyframes
            slerp = Slerp([f0, f1], Rotation.concatenate([r0, r1]))
            interp = slerp([target_frame])[0]
            euler = interp.as_euler("XYZ", degrees=True)
            return euler.tolist()

    # Fallback (should not reach here)
    return keyframes[-1][joint]
