"""Unit tests for rotation validation module."""

import pytest

from pose_director_root.utils.rotation_validator import (
    REQUIRED_JOINTS,
    JOINT_LIMITS,
    MAX_ANGLE_PER_FRAME,
    validate_rotations,
    sanitize_rotations,
    validate_rotation_consistency,
)


def _make_valid_keyframe(frame=0, overrides=None):
    """Create a valid keyframe dict with all 15 joints at zero."""
    rotations = {joint: [0, 0, 0] for joint in REQUIRED_JOINTS}
    if overrides:
        rotations.update(overrides)
    return {"frame": frame, "rotations": rotations}


class TestValidateRotations:
    def test_valid_all_joints_present(self):
        kfs = [_make_valid_keyframe(0), _make_valid_keyframe(16)]
        valid, issues = validate_rotations(kfs)
        assert valid
        assert len(issues) == 0

    def test_missing_rotations_key(self):
        kfs = [{"frame": 0}]
        valid, issues = validate_rotations(kfs)
        assert not valid
        assert any("missing 'rotations'" in i.message for i in issues)

    def test_missing_joints(self):
        kfs = [{"frame": 0, "rotations": {"spine": [0, 0, 0]}}]
        valid, issues = validate_rotations(kfs)
        assert not valid
        missing_msgs = [i for i in issues if "missing joints" in i.message]
        assert len(missing_msgs) == 1

    def test_extra_joints_warning(self):
        rots = {j: [0, 0, 0] for j in REQUIRED_JOINTS}
        rots["tail"] = [0, 0, 0]
        kfs = [{"frame": 0, "rotations": rots}]
        valid, issues = validate_rotations(kfs)
        assert valid  # extra joints are warnings, not errors
        assert any("unknown joints" in i.message for i in issues)

    def test_bad_angle_type(self):
        rots = {j: [0, 0, 0] for j in REQUIRED_JOINTS}
        rots["spine"] = "not a list"
        kfs = [{"frame": 0, "rotations": rots}]
        valid, issues = validate_rotations(kfs)
        assert not valid
        assert any("must be a list" in i.message for i in issues)

    def test_wrong_angle_count(self):
        rots = {j: [0, 0, 0] for j in REQUIRED_JOINTS}
        rots["spine"] = [0, 0]
        kfs = [{"frame": 0, "rotations": rots}]
        valid, issues = validate_rotations(kfs)
        assert not valid
        assert any("exactly 3 angles" in i.message for i in issues)

    def test_non_numeric_angle(self):
        rots = {j: [0, 0, 0] for j in REQUIRED_JOINTS}
        rots["spine"] = [0, "hello", 0]
        kfs = [{"frame": 0, "rotations": rots}]
        valid, issues = validate_rotations(kfs)
        assert not valid
        assert any("must be numeric" in i.message for i in issues)

    def test_multiple_keyframes_first_bad(self):
        kfs = [
            {"frame": 0, "rotations": {"spine": [0, 0, 0]}},  # missing joints
            _make_valid_keyframe(16),
        ]
        valid, issues = validate_rotations(kfs)
        assert not valid


class TestSanitizeRotations:
    def test_valid_values_unchanged(self):
        kfs = [_make_valid_keyframe(0, {"spine": [10, 20, -10]})]
        result = sanitize_rotations(kfs)
        assert result[0]["rotations"]["spine"] == [10, 20, -10]

    def test_elbow_clamp_right(self):
        # Right elbow Z should be clamped to (-145, 10)
        kfs = [_make_valid_keyframe(0, {"right_elbow": [0, 0, 200]})]
        result = sanitize_rotations(kfs)
        assert result[0]["rotations"]["right_elbow"][2] == 10

    def test_elbow_clamp_left(self):
        # Left elbow Z should be clamped to (-10, 145)
        kfs = [_make_valid_keyframe(0, {"left_elbow": [0, 0, -200]})]
        result = sanitize_rotations(kfs)
        assert result[0]["rotations"]["left_elbow"][2] == -10

    def test_knee_clamp_backward(self):
        # Knee X must not go negative (no backward bend)
        kfs = [_make_valid_keyframe(0, {"right_knee": [-30, 0, 0]})]
        result = sanitize_rotations(kfs)
        assert result[0]["rotations"]["right_knee"][0] == -5  # min is -5

    def test_knee_clamp_max(self):
        kfs = [_make_valid_keyframe(0, {"left_knee": [200, 0, 0]})]
        result = sanitize_rotations(kfs)
        assert result[0]["rotations"]["left_knee"][0] == 140

    def test_does_not_mutate_input(self):
        original = [_make_valid_keyframe(0, {"spine": [100, 100, 100]})]
        original_copy = {"frame": 0, "rotations": dict(original[0]["rotations"])}
        sanitize_rotations(original)
        assert original[0]["rotations"]["spine"] == original_copy["rotations"]["spine"]

    def test_unknown_joints_dropped(self):
        rots = {j: [0, 0, 0] for j in REQUIRED_JOINTS}
        rots["phantom_joint"] = [99, 99, 99]
        kfs = [{"frame": 0, "rotations": rots}]
        result = sanitize_rotations(kfs)
        assert "phantom_joint" not in result[0]["rotations"]

    def test_all_joints_have_limits(self):
        """Every required joint must have defined limits."""
        for joint in REQUIRED_JOINTS:
            assert joint in JOINT_LIMITS, f"Missing limits for {joint}"


class TestValidateRotationConsistency:
    def test_single_keyframe_consistent(self):
        kfs = [_make_valid_keyframe(0)]
        result = validate_rotation_consistency(kfs)
        assert result["is_consistent"]

    def test_identical_keyframes_consistent(self):
        kfs = [_make_valid_keyframe(0), _make_valid_keyframe(16)]
        result = validate_rotation_consistency(kfs)
        assert result["is_consistent"]

    def test_smooth_transition_consistent(self):
        # 10° change over 16 frames = 0.625°/frame — well under limit
        kf0 = _make_valid_keyframe(0)
        kf1 = _make_valid_keyframe(16, {"right_shoulder": [10, 0, 0]})
        result = validate_rotation_consistency([kf0, kf1])
        assert result["is_consistent"]

    def test_wild_swing_flagged(self):
        # 300° change over 16 frames = 18.75°/frame — over 15°/frame limit
        kf0 = _make_valid_keyframe(0)
        kf1 = _make_valid_keyframe(16, {"right_shoulder": [300, 0, 0]})
        result = validate_rotation_consistency([kf0, kf1])
        assert not result["is_consistent"]
        assert any(f["joint"] == "right_shoulder" for f in result["flagged"])

    def test_max_delta_tracked(self):
        kf0 = _make_valid_keyframe(0)
        kf1 = _make_valid_keyframe(16, {"spine": [30, 30, 30]})
        result = validate_rotation_consistency([kf0, kf1])
        assert result["max_delta"] == 30.0

    def test_frame_gap_scales_limit(self):
        # 200° over 40 frames = 5°/frame — should pass
        kf0 = _make_valid_keyframe(0)
        kf1 = _make_valid_keyframe(40, {"right_shoulder": [200, 0, 0]})
        result = validate_rotation_consistency([kf0, kf1])
        assert result["is_consistent"]
