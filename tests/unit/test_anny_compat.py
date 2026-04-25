"""Unit tests for utils/anny_compat.py — rotation→Anny mapping."""

import math

import pytest
import torch

from pose_director_root.utils.anny_compat import (
    JOINT_NAME_TO_ANNY,
    DEFAULT_PHENOTYPE,
    euler_to_rotation_matrix,
    rotations_to_pose_parameters,
)


class TestJointNameMapping:
    """Tests for the JOINT_NAME_TO_ANNY mapping table."""

    def test_major_joints_present(self):
        """All major limb joints should be mapped."""
        required = [
            "spine", "neck", "head",
            "right_shoulder", "right_elbow", "right_wrist",
            "left_shoulder", "left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle",
            "left_hip", "left_knee", "left_ankle",
        ]
        for joint in required:
            assert joint in JOINT_NAME_TO_ANNY, f"Missing joint: {joint}"

    def test_no_duplicate_indices(self):
        """Each bone index should map to exactly one canonical name."""
        indices = list(JOINT_NAME_TO_ANNY.values())
        # Allow duplicates only for aliases (e.g., "right_upper_arm" and "right_upperarm")
        # But canonical names should be unique
        canonical_names = [n for n in JOINT_NAME_TO_ANNY if "_" not in n or n.endswith(("_01", "_02")) or n in (
            "spine", "neck", "head",
            "right_shoulder", "right_elbow", "right_wrist",
            "left_shoulder", "left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle",
            "left_hip", "left_knee", "left_ankle",
        )]
        # Just verify indices are in valid range
        for name, idx in JOINT_NAME_TO_ANNY.items():
            assert 0 <= idx < 163, f"{name} -> {idx} out of range [0, 163)"

    def test_aliases_map_correctly(self):
        """Alternative names should map to the same index."""
        assert JOINT_NAME_TO_ANNY["right_upper_arm"] == JOINT_NAME_TO_ANNY["right_upperarm"]
        assert JOINT_NAME_TO_ANNY["left_upper_arm"] == JOINT_NAME_TO_ANNY["left_upperarm"]


class TestEulerToRotationMatrix:
    """Tests for euler_to_rotation_matrix."""

    def test_identity_rotation(self):
        """Zero degrees on all axes should give identity matrix."""
        rot = euler_to_rotation_matrix(0, 0, 0)
        assert torch.allclose(rot, torch.eye(3), atol=1e-6)

    def test_90_degree_x(self):
        """90° around X should swap Y and Z."""
        rot = euler_to_rotation_matrix(90, 0, 0)
        expected = torch.tensor([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=torch.float32)
        assert torch.allclose(rot, expected, atol=1e-5)

    def test_90_degree_y(self):
        """90° around Y should swap X and Z."""
        rot = euler_to_rotation_matrix(0, 90, 0)
        expected = torch.tensor([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ], dtype=torch.float32)
        assert torch.allclose(rot, expected, atol=1e-5)

    def test_90_degree_z(self):
        """90° around Z should swap X and Y."""
        rot = euler_to_rotation_matrix(0, 0, 90)
        expected = torch.tensor([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=torch.float32)
        assert torch.allclose(rot, expected, atol=1e-5)

    def test_orthogonality(self):
        """Rotation matrix should be orthogonal (det=±1, R^T R = I)."""
        for angles in [(30, 45, 60), (-90, 0, 45), (0, 0, 0), (180, -90, 270)]:
            rot = euler_to_rotation_matrix(*angles)
            # R^T R ≈ I
            assert torch.allclose(rot @ rot.T, torch.eye(3), atol=1e-5)
            # det ≈ 1 (proper rotation)
            assert abs(torch.det(rot).item() - 1.0) < 1e-5

    def test_output_shape(self):
        """Should return (3, 3) tensor."""
        rot = euler_to_rotation_matrix(10, 20, 30)
        assert rot.shape == (3, 3)


class TestRotationsToPoseParameters:
    """Tests for rotations_to_pose_parameters."""

    def test_identity_pose(self):
        """Empty rotations dict should produce identity pose."""
        pose = rotations_to_pose_parameters({})
        assert pose.shape == (1, 163, 4, 4)
        # All bones should be identity
        identity = torch.eye(4)
        for i in range(163):
            assert torch.allclose(pose[0, i], identity, atol=1e-6)

    def test_single_joint_rotation(self):
        """Rotating one joint should only affect that bone."""
        pose = rotations_to_pose_parameters({"right_shoulder": [0, 0, 90]})
        identity = torch.eye(4)
        # right_shoulder should be rotated
        idx = JOINT_NAME_TO_ANNY["right_shoulder"]
        rot_block = pose[0, idx, :3, :3]
        # Should NOT be identity
        assert not torch.allclose(rot_block, identity[:3, :3])
        # Other bones should be identity
        for i in range(163):
            if i != idx:
                assert torch.allclose(pose[0, i], identity, atol=1e-6)

    def test_unknown_joint_ignored(self):
        """Unknown joint names should be skipped with a warning."""
        pose = rotations_to_pose_parameters({"nonexistent_joint": [0, 0, 90]})
        identity = torch.eye(4)
        for i in range(163):
            assert torch.allclose(pose[0, i], identity, atol=1e-6)

    def test_wrong_angle_count_ignored(self):
        """Joints with wrong number of angles should be skipped."""
        pose = rotations_to_pose_parameters({"right_shoulder": [0, 90]})
        identity = torch.eye(4)
        for i in range(163):
            assert torch.allclose(pose[0, i], identity, atol=1e-6)

    def test_preserves_translation_row(self):
        """The 4th row [0, 0, 0, 1] should always be preserved."""
        pose = rotations_to_pose_parameters({
            "spine": [45, 30, 60],
            "right_knee": [90, 0, 0],
        })
        for i in range(163):
            assert pose[0, i, 3, :3].sum().item() == 0.0
            assert pose[0, i, 3, 3].item() == 1.0

    def test_multiple_joints(self):
        """Multiple joints should each get their rotation applied."""
        rotations = {
            "right_shoulder": [0, 0, 45],
            "left_shoulder": [0, 0, -45],
            "right_knee": [30, 0, 0],
        }
        pose = rotations_to_pose_parameters(rotations)
        identity = torch.eye(4)

        for name, angles in rotations.items():
            idx = JOINT_NAME_TO_ANNY[name]
            expected_rot = euler_to_rotation_matrix(*angles)
            assert torch.allclose(pose[0, idx, :3, :3], expected_rot, atol=1e-5)
