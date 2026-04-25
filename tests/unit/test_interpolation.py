"""Tests for SLERP keyframe interpolation."""

import math

import pytest

from utils.interpolation import interpolate_keyframes, JOINT_NAMES


def _tpose():
    """T-pose frame: all joints at identity."""
    return {j: [0.0, 0.0, 0.0] for j in JOINT_NAMES}


def _frame_with(**overrides):
    """T-pose frame with specific joint overrides."""
    f = _tpose()
    f.update(overrides)
    return f


class TestInterpolateKeyframes:
    def test_identity_all_frames(self):
        """All keyframes are T-pose → all interpolated frames should be T-pose."""
        kfs = [_tpose(), _tpose(), _tpose()]
        frames = [0, 24, 47]
        result = interpolate_keyframes(kfs, frames, 48)
        assert len(result) == 48
        for i, frame in enumerate(result):
            for joint, angles in frame.items():
                for a in angles:
                    assert abs(a) < 1e-6, f"Frame {i}, joint {joint}: {angles}"

    def test_output_count(self):
        """Output length matches total_frames."""
        kfs = [_tpose(), _tpose()]
        for total in [8, 30, 48, 120, 240]:
            result = interpolate_keyframes(kfs, [0, total - 1], total)
            assert len(result) == total, f"Expected {total} frames, got {len(result)}"

    def test_single_keyframe_constant(self):
        """Single keyframe should produce constant pose for all frames."""
        kf = _frame_with(right_elbow=[0, 0, -45])
        result = interpolate_keyframes([kf], [0], 30)
        assert len(result) == 30
        for frame in result:
            assert frame["right_elbow"] == pytest.approx([0, 0, -45], abs=1e-6)

    def test_two_keyframes_midpoint(self):
        """Midpoint between two keyframes should have ~half the rotation."""
        kf0 = _frame_with(spine=[0, 0, 0])
        kf1 = _frame_with(spine=[0, 0, 90])
        result = interpolate_keyframes([kf0, kf1], [0, 10], 11)
        # Frame 5 should be roughly [0, 0, 45]
        mid = result[5]["spine"]
        assert abs(mid[2] - 45.0) < 2.0, f"Expected ~45, got {mid[2]}"

    def test_frame_clamping_before_first(self):
        """Frames before first keyframe use first keyframe pose."""
        kf0 = _frame_with(head=[10, 20, 30])
        kf1 = _frame_with(head=[40, 50, 60])
        result = interpolate_keyframes([kf0, kf1], [5, 20], 25)
        # Frame 0 is before first keyframe at 5
        for j, a in zip(JOINT_NAMES, result[0].values()):
            if j == "head":
                continue
            assert a == pytest.approx([0, 0, 0], abs=1e-6)
        assert result[0]["head"] == pytest.approx([10, 20, 30], abs=1e-6)

    def test_frame_clamping_after_last(self):
        """Frames after last keyframe use last keyframe pose."""
        kf0 = _tpose()
        kf1 = _frame_with(neck=[15, 0, 0])
        result = interpolate_keyframes([kf0, kf1], [0, 10], 20)
        # Frame 15 is after last keyframe at 10
        assert result[15]["neck"] == pytest.approx([15, 0, 0], abs=1e-6)

    def test_multi_joint_independent(self):
        """Each joint should interpolate independently."""
        kf0 = _frame_with(spine=[0, 0, 0], head=[0, 0, 0])
        kf1 = _frame_with(spine=[30, 0, 0], head=[0, 60, 0])
        result = interpolate_keyframes([kf0, kf1], [0, 10], 11)
        mid = result[5]
        # Spine should be ~15 on X
        assert abs(mid["spine"][0] - 15.0) < 2.0
        # Head should be ~30 on Y
        assert abs(mid["head"][1] - 30.0) < 2.0

    def test_rotation_continuity(self):
        """Consecutive interpolated frames should differ by small amounts."""
        kf0 = _frame_with(right_shoulder=[0, 0, 0])
        kf1 = _frame_with(right_shoulder=[0, 0, 90])
        result = interpolate_keyframes([kf0, kf1], [0, 30], 31)
        for i in range(1, len(result)):
            prev = result[i - 1]["right_shoulder"][2]
            curr = result[i]["right_shoulder"][2]
            delta = abs(curr - prev)
            assert delta < 10.0, f"Frame {i}: jump from {prev} to {curr} (delta={delta})"

    def test_empty_keyframes_returns_tpose(self):
        """Empty keyframe list returns all T-pose frames."""
        result = interpolate_keyframes([], [], 10)
        assert len(result) == 10
        for frame in result:
            for joint, angles in frame.items():
                assert angles == pytest.approx([0, 0, 0], abs=1e-6)

    def test_mismatched_lengths_raises(self):
        """Mismatched keyframes/frame_numbers should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            interpolate_keyframes([_tpose(), _tpose()], [0], 10)

    def test_all_joints_present(self):
        """Every output frame should contain all 15 joints."""
        kfs = [_tpose(), _frame_with(spine=[10, 0, 0])]
        result = interpolate_keyframes(kfs, [0, 10], 11)
        for i, frame in enumerate(result):
            assert set(frame.keys()) == set(JOINT_NAMES), f"Frame {i} missing joints"

    def test_missing_joints_filled_with_identity(self):
        """Joints not in keyframes default to T-pose."""
        kf0 = {"spine": [10, 0, 0]}  # Missing all other joints
        kf1 = {"spine": [20, 0, 0]}
        result = interpolate_keyframes([kf0, kf1], [0, 10], 11)
        for frame in result:
            for joint in JOINT_NAMES:
                if joint == "spine":
                    continue
                assert frame[joint] == pytest.approx([0, 0, 0], abs=1e-6)
