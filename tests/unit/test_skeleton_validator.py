"""Unit tests for skeleton validator utility."""

import pytest

from pose_director_root.utils.skeleton_validator import (
    validate_keypoints,
    validate_keyframe_consistency,
    sanitize_keypoints,
    ValidationIssue,
)


def make_valid_keypoints(canvas_w=1024, canvas_h=1024):
    """Generate a valid COCO-18 keypoint array for a standing person on a larger canvas."""
    cx, cy = canvas_w / 2, canvas_h / 2
    return [
        cx, cy - 360, 0.9,       # 0 Nose
        cx, cy - 320, 0.9,       # 1 Neck
        cx - 100, cy - 280, 0.9, # 2 RShoulder
        cx - 140, cy - 160, 0.9, # 3 RElbow
        cx - 120, cy - 40, 0.9,  # 4 RWrist
        cx + 100, cy - 280, 0.9, # 5 LShoulder
        cx + 140, cy - 160, 0.9, # 6 LElbow
        cx + 120, cy - 40, 0.9,  # 7 LWrist
        cx - 60, cy - 120, 0.9,  # 8 RHip
        cx - 70, cy + 60, 0.9,   # 9 RKnee
        cx - 80, cy + 240, 0.9,  # 10 RAnkle
        cx + 60, cy - 120, 0.9,  # 11 LHip
        cx + 70, cy + 60, 0.9,   # 12 LKnee
        cx + 80, cy + 240, 0.9,  # 13 LAnkle
        cx - 10, cy - 380, 0.9,  # 14 REye
        cx + 10, cy - 380, 0.9,  # 15 LEye
        cx - 30, cy - 370, 0.9,  # 16 REar
        cx + 30, cy - 370, 0.9,  # 17 LEar
    ]


class TestValidateKeypoints:
    """Tests for single keypoint set validation."""

    def test_valid_standing_pose(self):
        """A valid standing pose should pass with no error issues."""
        kpts = make_valid_keypoints()
        is_valid, issues = validate_keypoints(kpts, 1024, 1024)
        error_issues = [i for i in issues if i.severity == "error"]
        assert len(error_issues) == 0, f"Unexpected errors: {error_issues}"

    def test_out_of_bounds_detection(self):
        """Keypoints far outside canvas should be flagged."""
        kpts = make_valid_keypoints()
        kpts[0] = -500  # Nose x way out of bounds
        kpts[7 * 3 + 1] = 2000  # LWrist y out of bounds
        _, issues = validate_keypoints(kpts, 1024, 1024)
        bounds_issues = [i for i in issues if i.category == "bounds"]
        assert len(bounds_issues) >= 2

    def test_zero_length_limb_with_confidence(self):
        """Coincident high-confidence joints should be flagged as error."""
        kpts = make_valid_keypoints()
        # Make RShoulder and RElbow the same point
        kpts[3 * 3] = kpts[2 * 3]      # RElbow x = RShoulder x
        kpts[3 * 3 + 1] = kpts[2 * 3 + 1]  # RElbow y = RShoulder y
        _, issues = validate_keypoints(kpts, 1024, 1024)
        error_issues = [i for i in issues if i.severity == "error" and i.category == "proportion"]
        assert len(error_issues) >= 1

    def test_extreme_limb_ratio(self):
        """A shin much longer than thigh should be flagged."""
        kpts = make_valid_keypoints()
        # Move RAnkle way down to make shin very long
        kpts[10 * 3 + 1] = kpts[9 * 3 + 1] + 800  # RAnkle very far below RKnee
        _, issues = validate_keypoints(kpts, 1024, 1024)
        proportion_issues = [i for i in issues if i.category == "proportion"]
        assert len(proportion_issues) >= 1

    def test_impossible_elbow_angle(self):
        """Elbow bending backwards should be flagged."""
        kpts = make_valid_keypoints()
        # Make RElbow bend outward past 180 degrees
        # RShoulder is at (cx-100, cy-280), RElbow at (cx-140, cy-160)
        # Move RWrist to same side as RShoulder (bending wrong way)
        kpts[4 * 3] = kpts[2 * 3] - 20     # RWrist x past RShoulder
        kpts[4 * 3 + 1] = kpts[2 * 3 + 1]  # RWrist y at RShoulder height
        _, issues = validate_keypoints(kpts, 1024, 1024)
        angle_issues = [i for i in issues if i.category == "angle"]
        assert len(angle_issues) >= 1

    def test_unbalanced_pose(self):
        """CoG far outside foot stance should be flagged."""
        kpts = make_valid_keypoints()
        # Shift entire upper body + hips far to one side
        for i in [0, 1, 2, 5, 8, 11]:  # Nose, Neck, RShoulder, LShoulder, RHip, LHip
            kpts[i * 3] += 500  # shift x by 500px
        _, issues = validate_keypoints(kpts, 1024, 1024)
        balance_issues = [i for i in issues if i.category == "balance"]
        assert len(balance_issues) >= 1

    def test_low_confidence_joints_skipped(self):
        """Joints with zero confidence should not trigger angle/proportion checks."""
        kpts = make_valid_keypoints()
        # Zero out right arm confidence
        for i in [2, 3, 4]:  # RShoulder, RElbow, RWrist
            kpts[i * 3 + 2] = 0.0
        _, issues = validate_keypoints(kpts, 1024, 1024)
        # Should not have proportion errors for zero-confidence limbs
        error_issues = [i for i in issues if i.severity == "error" and i.category == "proportion"
                        and any(idx in [2, 3, 4] for idx in i.keypoint_indices)]
        assert len(error_issues) == 0

    def test_confidence_gap_detection(self):
        """A low-confidence joint between high-confidence joints should be flagged."""
        kpts = make_valid_keypoints()
        kpts[3 * 3 + 2] = 0.05  # RElbow very low confidence
        _, issues = validate_keypoints(kpts, 1024, 1024)
        conf_issues = [i for i in issues if i.category == "confidence"]
        assert any(3 in i.keypoint_indices for i in conf_issues)


class TestValidateKeyframeConsistency:
    """Tests for cross-keyframe shimmer detection."""

    def test_consistent_keyframes_pass(self):
        """Same pose across keyframes should show near-zero variance."""
        kpts = make_valid_keypoints()
        result = validate_keyframe_consistency([kpts, kpts, kpts])
        assert result["is_consistent"]
        assert result["max_variance"] < 0.01

    def test_shimmer_detection(self):
        """Limb length changing 50% between frames should be flagged."""
        kpts_a = make_valid_keypoints()
        kpts_b = make_valid_keypoints()
        # Double the right shin length in keyframe B
        kpts_b[10 * 3 + 1] = kpts_b[9 * 3 + 1] + 400  # RAnkle much further down
        result = validate_keyframe_consistency([kpts_a, kpts_b])
        assert not result["is_consistent"]
        assert "right_shin" in result["flagged_bones"]

    def test_single_keyframe_handled(self):
        """Single keyframe should return consistent=True."""
        kpts = make_valid_keypoints()
        result = validate_keyframe_consistency([kpts])
        assert result["is_consistent"]
        assert result["max_variance"] == 0.0


class TestJointOverlap:
    """Tests for joint overlap detection and sanitization."""

    def test_overlapping_joints_detected(self):
        """Two connected joints at the same position should be flagged."""
        kpts = make_valid_keypoints()
        # Place RElbow (3) and RWrist (4) at the same position
        kpts[4 * 3] = kpts[3 * 3]       # RWrist x = RElbow x
        kpts[4 * 3 + 1] = kpts[3 * 3 + 1]  # RWrist y = RElbow y
        _, issues = validate_keypoints(kpts, 1024, 1024)
        overlap_issues = [i for i in issues if i.category == "overlap"]
        assert len(overlap_issues) >= 1

    def test_no_overlap_for_normal_pose(self):
        """Normal pose should have no overlap warnings."""
        kpts = make_valid_keypoints()
        _, issues = validate_keypoints(kpts, 1024, 1024)
        overlap_issues = [i for i in issues if i.category == "overlap"]
        assert len(overlap_issues) == 0

    def test_sanitize_pushes_overlapping_apart(self):
        """sanitize_keypoints should push overlapping joints apart."""
        kpts = make_valid_keypoints()
        # Collapse RElbow and RWrist to same point
        kpts[4 * 3] = kpts[3 * 3]
        kpts[4 * 3 + 1] = kpts[3 * 3 + 1]

        result = sanitize_keypoints(kpts, 1024, 1024)
        # After sanitization, the distance should be >= min_bone_length
        ax, ay = result[3 * 3], result[3 * 3 + 1]
        bx, by = result[4 * 3], result[4 * 3 + 1]
        dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        assert dist > 0  # no longer overlapping

    def test_sanitize_preserves_non_overlapping(self):
        """Non-overlapping joints should not be changed."""
        kpts = make_valid_keypoints()
        result = sanitize_keypoints(kpts, 1024, 1024)
        # Should be identical (no overlaps to fix)
        for i in range(len(kpts)):
            assert abs(result[i] - kpts[i]) < 0.01

    def test_sanitize_does_not_mutate_input(self):
        """Should return a new list, not modify the input."""
        kpts = make_valid_keypoints()
        original = list(kpts)
        # Collapse a pair
        kpts[4 * 3] = kpts[3 * 3]
        kpts[4 * 3 + 1] = kpts[3 * 3 + 1]
        original_collapsed = list(kpts)

        sanitize_keypoints(kpts, 1024, 1024)
        # Input should still be collapsed
        assert kpts == original_collapsed
