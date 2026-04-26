"""HY-Motion → Anny bone rotation converter.

Converts HY-Motion (Tencent Hunyuan) trained motion DiT output to Anny-compatible
bone rotations for the sprite pipeline.

HY-Motion outputs 22 joints with 6D rotation representation in Y-up convention.
Anny uses Z-up convention with Euler angle rotations.

Joint mapping:
  HY-Motion (22 joints)  →  Anny bone indices
  Pelvis(0)              →  pelvis(0)
  Spine1(3)              →  spine(44)
  Spine2(6)              →  spine(44)  (combined with Spine1)
  Spine3(9)              →  spine03(47)
  Neck(12)               →  neck(102)
  Head(15)               →  head(103)
  L_Shoulder(16)          →  left_shoulder(77)
  R_Shoulder(17)          →  right_shoulder(51)
  L_Elbow(18)             →  left_elbow(78)
  R_Elbow(19)             →  right_elbow(52)
  L_Wrist(20)             →  left_wrist(79)
  R_Wrist(21)             →  right_wrist(53)
  L_Hip(1)               →  left_hip(21)
  R_Hip(2)               →  right_hip(1)
  L_Knee(4)              →  left_knee(24)
  R_Knee(5)              →  right_knee(4)
  L_Ankle(7)             →  left_ankle(25)
  R_Ankle(8)             →  right_ankle(5)
"""

import math
from pathlib import Path

import numpy as np

# HY-Motion joint indices → Anny bone names
HYMOTION_TO_ANNY: dict[int, str] = {
    0: "pelvis",
    3: "spine",
    6: "spine",    # Spine2 → combined with Spine1
    9: "spine_03",
    12: "neck",
    15: "head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
    1: "left_hip",
    2: "right_hip",
    4: "left_knee",
    5: "right_knee",
    7: "left_ankle",
    8: "right_ankle",
}

# For spine combination: average Spine1(3) and Spine2(6) into one "spine" rotation
SPINE_INDICES = [3, 6]


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation to 3x3 rotation matrix (Zhou et al., CVPR 2019).

    Args:
        rot6d: (..., 6) array of 6D rotation representations.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    x = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]

    # Gram-Schmidt
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to XYZ Euler angles in degrees.

    Uses R = Rz * Ry * Rx convention (same as Anny's euler_to_rotation_matrix).
    """
    sy = -R[..., 2, 0]
    cy = np.sqrt(R[..., 0, 0]**2 + R[..., 1, 0]**2 + 1e-10)

    x = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    y = np.arctan2(sy, cy)
    z = np.arctan2(R[..., 1, 0], R[..., 0, 0])

    return np.degrees(np.stack([x, y, z], axis=-1))


def convert_yup_to_zup(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert Euler angles from Y-up to Z-up convention.

    Y-up: X=right, Y=up, Z=forward
    Z-up: X=right, Y=forward, Z=up

    For Euler XYZ in R=Rz*Ry*Rx convention, the axis swap Y↔Z
    transforms the rotation semantics appropriately for body joints.
    """
    x, y, z = euler_xyz[..., 0], euler_xyz[..., 1], euler_xyz[..., 2]
    # Swap Y and Z components
    return np.stack([x, z, y], axis=-1)


def extract_keyframes(
    npz_path: str | Path,
    num_keyframes: int = 4,
) -> list[dict[str, list[float]]]:
    """Extract keyframe rotations from HY-Motion NPZ output.

    Args:
        npz_path: Path to HY-Motion NPZ file.
        num_keyframes: Number of keyframes to extract (evenly spaced).

    Returns:
        List of dicts: [{"right_shoulder": [x,y,z], ...}, ...]
    """
    data = np.load(str(npz_path), allow_pickle=True)
    rot6d = data["rot6d"]  # (frames, 22, 6)
    total_frames = rot6d.shape[0]

    # Sample keyframes evenly
    if num_keyframes >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_keyframes, dtype=int).tolist()

    keyframes = []
    for fi in indices:
        frame_rot6d = rot6d[fi]  # (22, 6)
        frame_matrices = rot6d_to_matrix(frame_rot6d)  # (22, 3, 3)

        # Convert to Euler angles and map joints
        rotations: dict[str, list[float]] = {}

        for hym_idx, anny_name in HYMOTION_TO_ANNY.items():
            R = frame_matrices[hym_idx]  # (3, 3)
            euler = matrix_to_euler_xyz(R)  # (3,)
            euler_zup = convert_yup_to_zup(euler)

            if anny_name in rotations:
                # Average with existing (for spine combination)
                existing = np.array(rotations[anny_name])
                rotations[anny_name] = ((existing + euler_zup) / 2).tolist()
            else:
                rotations[anny_name] = euler_zup.tolist()

        keyframes.append(rotations)

    return keyframes


def npz_to_sprite_poses(
    npz_path: str | Path,
    num_keyframes: int = 4,
) -> list[dict]:
    """Convert HY-Motion NPZ to sprite pipeline pose format.

    Returns list of pose dicts compatible with run_sprite_pipeline.py:
        [{"bones": {"right_shoulder": [x,y,z], ...}}, ...]
    """
    keyframes = extract_keyframes(npz_path, num_keyframes)
    return [{"bones": kf} for kf in keyframes]
