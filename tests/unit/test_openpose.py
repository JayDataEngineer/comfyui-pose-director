"""Unit tests for OpenPose constants and data structures."""

from pose_director_root.utils.openpose import (
    KEYPOINT_NAMES,
    NUM_BODY_KEYPOINTS,
    SKELETON_BONES,
    BONE_COLORS,
    KEYPOINT_COLORS,
    BODY_PARTS,
    KEYPOINT_LABELS,
)


class TestOpenPoseConstants:
    """Tests for OpenPose data structure consistency."""

    def test_num_keypoints_matches_names(self):
        assert len(KEYPOINT_NAMES) == NUM_BODY_KEYPOINTS

    def test_skeleton_bones_count(self):
        """Should have 17 bones for COCO-18."""
        assert len(SKELETON_BONES) == 17

    def test_bone_colors_match_bones(self):
        """Each bone should have a corresponding color."""
        assert len(BONE_COLORS) == len(SKELETON_BONES)

    def test_keypoint_colors_match_keypoints(self):
        """Each keypoint should have a corresponding color."""
        assert len(KEYPOINT_COLORS) == NUM_BODY_KEYPOINTS

    def test_keypoint_labels_match_names(self):
        """KEYPOINT_LABELS should be derived from KEYPOINT_NAMES."""
        assert len(KEYPOINT_LABELS) == len(KEYPOINT_NAMES)
        for label in KEYPOINT_LABELS:
            parts = label.split(":", 1)
            assert len(parts) == 2
            idx = int(parts[0])
            assert idx in KEYPOINT_NAMES
            assert KEYPOINT_NAMES[idx] == parts[1]

    def test_body_parts_are_valid_indices(self):
        """All indices in BODY_PARTS should be within 0-17."""
        for part_name, indices in BODY_PARTS.items():
            for idx in indices:
                assert 0 <= idx < NUM_BODY_KEYPOINTS
                assert idx in KEYPOINT_NAMES

    def test_bone_colors_are_rgb(self):
        """Each bone color should be a 3-tuple of ints 0-255."""
        for color in BONE_COLORS:
            assert len(color) == 3
            for channel in color:
                assert isinstance(channel, int)
                assert 0 <= channel <= 255

    def test_keypoint_colors_are_rgb(self):
        """Each keypoint color should be a 3-tuple of ints 0-255."""
        for color in KEYPOINT_COLORS:
            assert len(color) == 3
            for channel in color:
                assert isinstance(channel, int)
                assert 0 <= channel <= 255

    def test_bones_connect_valid_keypoints(self):
        """All bone endpoints should be valid keypoint indices."""
        for a, b in SKELETON_BONES:
            assert a in KEYPOINT_NAMES
            assert b in KEYPOINT_NAMES
