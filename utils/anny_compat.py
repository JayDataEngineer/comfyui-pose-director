"""Anny body model compatibility layer — named rotations to Anny pose parameters.

Anny (Naver Labs, Apache 2.0) replaces SMPL for parametric body mesh generation.
163 bones, 13,718 vertices, 27,420 faces.

The LLM outputs named joint rotations (degrees from T-pose). This module maps
those to Anny's 4x4 pose parameter tensor.
"""

import math
from typing import Optional

import torch

# Mapping from LLM-friendly joint names to Anny bone indices.
# Anny has 163 bones but we only manipulate the major limb/spine bones.
# Unlisted bones default to identity (T-pose).
JOINT_NAME_TO_ANNY: dict[str, int] = {
    "spine": 3,
    "spine01": 3,
    "spine02": 4,
    "spine03": 5,
    "spine04": 6,
    "spine05": 7,
    "neck": 12,
    "neck01": 10,
    "neck02": 11,
    "neck03": 12,
    "head": 15,
    # Right arm
    "right_shoulder": 16,
    "right_upper_arm": 18,
    "right_upperarm": 18,
    "right_elbow": 20,
    "right_forearm": 20,
    "right_wrist": 22,
    # Left arm
    "left_shoulder": 17,
    "left_upper_arm": 19,
    "left_upperarm": 19,
    "left_elbow": 21,
    "left_forearm": 21,
    "left_wrist": 23,
    # Right leg
    "right_hip": 1,
    "right_thigh": 25,
    "right_upper_leg": 25,
    "right_upperleg": 25,
    "right_knee": 28,
    "right_shin": 28,
    "right_ankle": 31,
    # Left leg
    "left_hip": 2,
    "left_thigh": 26,
    "left_upper_leg": 26,
    "left_upperleg": 26,
    "left_knee": 29,
    "left_shin": 29,
    "left_ankle": 32,
}

# Default phenotype for a generic body
DEFAULT_PHENOTYPE = {
    "height": 1.7,
    "weight": 70.0,
    "age": 25,
    "gender": 0,
    "muscle": 0.5,
}

# Cached model singleton
_model = None


def _get_model():
    """Lazy-load and cache the Anny full body model."""
    global _model
    if _model is None:
        import anny
        _model = anny.create_fullbody_model().float()
    return _model


def euler_to_rotation_matrix(x_deg: float, y_deg: float, z_deg: float) -> torch.Tensor:
    """Convert Euler angles (degrees, XYZ convention) to a 3x3 rotation matrix.

    Uses the R = Rz * Ry * Rx convention (extrinsic rotations).
    """
    x = math.radians(x_deg)
    y = math.radians(y_deg)
    z = math.radians(z_deg)

    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)

    # Rz * Ry * Rx
    r00 = cz * cy
    r01 = cz * sy * sx - sz * cx
    r02 = cz * sy * cx + sz * sx
    r10 = sz * cy
    r11 = sz * sy * sx + cz * cx
    r12 = sz * sy * cx - cz * sx
    r20 = -sy
    r21 = cy * sx
    r22 = cy * cx

    return torch.tensor([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22],
    ], dtype=torch.float32)


def rotations_to_pose_parameters(
    rotations_dict: dict[str, list[float]],
    bone_count: int = 163,
) -> torch.Tensor:
    """Convert named rotation dict to Anny (1, bone_count, 4, 4) pose tensor.

    rotations_dict: {"right_shoulder": [x_deg, y_deg, z_deg], ...}
    Unlisted bones default to identity (T-pose).

    Returns:
        (1, bone_count, 4, 4) float32 tensor — identity matrices with
        rotation overrides applied to the top-left 3x3 block.
    """
    # Start with identity pose (T-pose)
    identity = torch.eye(4, dtype=torch.float32)
    pose = identity.unsqueeze(0).expand(1, bone_count, -1, -1).clone()

    for joint_name, angles in rotations_dict.items():
        idx = JOINT_NAME_TO_ANNY.get(joint_name)
        if idx is None:
            print(f"[anny_compat] Warning: unknown joint '{joint_name}', skipping")
            continue
        if len(angles) != 3:
            print(f"[anny_compat] Warning: expected 3 angles for '{joint_name}', got {len(angles)}")
            continue

        rot = euler_to_rotation_matrix(angles[0], angles[1], angles[2])
        pose[0, idx, :3, :3] = rot

    return pose


def forward_anny(
    rotations_dict: dict[str, list[float]],
    phenotype_kwargs: Optional[dict] = None,
) -> dict:
    """Run Anny forward pass from named rotations.

    Args:
        rotations_dict: {"joint_name": [x_deg, y_deg, z_deg], ...}
        phenotype_kwargs: Body shape parameters (height, weight, age, gender, muscle).

    Returns:
        dict with 'vertices' (1, 13718, 3), 'bone_poses' (1, 163, 4, 4), etc.
    """
    model = _get_model()
    if phenotype_kwargs is None:
        phenotype_kwargs = dict(DEFAULT_PHENOTYPE)

    pose = rotations_to_pose_parameters(rotations_dict, model.bone_count)
    result = model.forward(
        pose_parameters=pose,
        phenotype_kwargs=phenotype_kwargs,
    )
    return result


def get_faces() -> torch.Tensor:
    """Get Anny triangular face indices (27420, 3)."""
    return _get_model().get_triangular_faces()
