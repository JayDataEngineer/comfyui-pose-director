"""Unit tests for HY-Motion → Anny bone rotation converter."""

import tempfile

import numpy as np
import pytest

from utils.hymotion_converter import (
    HYMOTION_TO_ANNY,
    convert_yup_to_zup,
    extract_keyframes,
    matrix_to_euler_xyz,
    npz_to_sprite_poses,
    rot6d_to_matrix,
)


def _make_test_npz(num_frames: int = 30, num_joints: int = 22) -> str:
    """Create a temporary NPZ file with synthetic HY-Motion data."""
    # Identity 6D rotation: [1, 0, 0, 0, 1, 0] → identity matrix
    rot6d = np.zeros((num_frames, num_joints, 6), dtype=np.float32)
    rot6d[..., 0] = 1.0  # a1_x
    rot6d[..., 4] = 1.0  # a2_y
    # Add some variation to right shoulder (joint 17) for testing
    for f in range(num_frames):
        angle = (f / num_frames) * np.pi / 2
        rot6d[f, 17, 0] = np.cos(angle)
        rot6d[f, 17, 1] = np.sin(angle)
        rot6d[f, 17, 4] = 1.0

    tmpfile = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(tmpfile.name, rot6d=rot6d)
    tmpfile.close()
    return tmpfile.name


class TestRot6dToMatrix:
    def test_identity_rotation(self):
        """Identity 6D representation should produce identity matrix.

        6D format is column-interleaved: [a1[0], a2[0], a1[1], a2[1], a1[2], a2[2]]
        Identity: a1 = [1,0,0], a2 = [0,1,0] → rot6d = [1,0, 0,1, 0,0]
        """
        rot6d = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)
        R = rot6d_to_matrix(rot6d)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-5)

    def test_90_degree_rotation_around_z(self):
        """A 90-degree rotation around Z axis.

        First column = [cos90, sin90, 0] = [0, 1, 0]
        Second column orthogonal = [-sin90, cos90, 0] = [-1, 0, 0]
        Interleaved: [0, -1, 1, 0, 0, 0]
        """
        rot6d = np.array([0, -1, 1, 0, 0, 0], dtype=np.float32)
        R = rot6d_to_matrix(rot6d)
        expected = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=np.float32)
        np.testing.assert_allclose(R, expected, atol=1e-5)

    def test_batch_input(self):
        """Should handle batched inputs."""
        rot6d = np.random.randn(5, 22, 6).astype(np.float32)
        R = rot6d_to_matrix(rot6d)
        assert R.shape == (5, 22, 3, 3)
        # Each should be a valid rotation matrix (det = 1)
        for i in range(5):
            for j in range(22):
                det = np.linalg.det(R[i, j])
                np.testing.assert_allclose(det, 1.0, atol=1e-4)


class TestMatrixToEuler:
    def test_identity(self):
        R = np.eye(3).reshape(1, 3, 3)
        euler = matrix_to_euler_xyz(R)
        np.testing.assert_allclose(euler[0], [0, 0, 0], atol=1e-5)

    def test_roundtrip(self):
        """Euler → matrix → Euler should be identity."""
        # Start with known Euler angles
        angles = np.array([[[30, 45, 60]]], dtype=np.float32)
        x, y, z = np.radians(angles[0, 0])

        Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        euler = matrix_to_euler_xyz(R.reshape(1, 3, 3))
        np.testing.assert_allclose(euler[0], [30, 45, 60], atol=1e-4)


class TestConvertYupToZup:
    def test_swap(self):
        euler = np.array([[10, 20, 30]], dtype=np.float32)
        result = convert_yup_to_zup(euler)
        np.testing.assert_allclose(result[0], [10, 30, 20])

    def test_zero(self):
        euler = np.zeros((1, 3))
        result = convert_yup_to_zup(euler)
        np.testing.assert_allclose(result[0], [0, 0, 0])


class TestExtractKeyframes:
    def test_basic_extraction(self):
        path = _make_test_npz(num_frames=30, num_joints=22)
        keyframes = extract_keyframes(path, num_keyframes=4)

        assert len(keyframes) == 4
        for kf in keyframes:
            assert isinstance(kf, dict)
            # Should have mapped bone names (17 unique: spine is combined from 2)
            assert len(kf) == len(set(HYMOTION_TO_ANNY.values()))
            for name, angles in kf.items():
                assert len(angles) == 3

    def test_single_keyframe(self):
        path = _make_test_npz(num_frames=10)
        keyframes = extract_keyframes(path, num_keyframes=1)
        assert len(keyframes) == 1

    def test_more_keyframes_than_frames(self):
        path = _make_test_npz(num_frames=3)
        keyframes = extract_keyframes(path, num_keyframes=10)
        assert len(keyframes) == 3  # capped at actual frame count

    def test_spine_combination(self):
        """Spine1(3) and Spine2(6) should be averaged into single 'spine' entry."""
        path = _make_test_npz(num_frames=10)
        keyframes = extract_keyframes(path, num_keyframes=1)

        # Should NOT have separate spine1/spine2 entries
        assert "spine" in keyframes[0]
        # Total unique names should exclude the duplicate "spine" mapping
        unique_anny_names = set(HYMOTION_TO_ANNY.values())
        assert len(keyframes[0]) == len(unique_anny_names)

    def test_right_shoulder_varies(self):
        """Right shoulder should vary across frames in test data."""
        path = _make_test_npz(num_frames=30)
        keyframes = extract_keyframes(path, num_keyframes=4)

        rs_angles = [kf["right_shoulder"][0] for kf in keyframes]
        # Not all identical (test data varies joint 17)
        assert len(set(round(a, 2) for a in rs_angles)) > 1


class TestNpzToSpritePoses:
    def test_format(self):
        path = _make_test_npz(num_frames=10)
        poses = npz_to_sprite_poses(path, num_keyframes=2)

        assert len(poses) == 2
        for pose in poses:
            assert "bones" in pose
            assert isinstance(pose["bones"], dict)
            for name, angles in pose["bones"].items():
                assert isinstance(name, str)
                assert len(angles) == 3
