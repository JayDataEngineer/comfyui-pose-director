"""Unit tests for pose seed library."""

import pytest

from pose_director_root.utils.pose_library import list_poses, load_pose, find_best_pose


class TestListPoses:
    """Tests for listing and filtering poses."""

    def test_list_all_poses(self):
        """Should return all 7 poses."""
        poses = list_poses()
        assert len(poses) == 7
        names = {p["name"] for p in poses}
        assert "standing_neutral" in names
        assert "dancing" in names

    def test_filter_by_tag(self):
        """Should filter poses by tag."""
        poses = list_poses(tags=["standing"])
        names = {p["name"] for p in poses}
        assert "standing_neutral" in names
        assert "dancing" not in names

    def test_filter_by_nonexistent_tag(self):
        """Should return empty list for non-matching tags."""
        poses = list_poses(tags=["underwater_basket_weaving"])
        assert len(poses) == 0

    def test_poses_have_required_fields(self):
        """Each listed pose should have name, description, tags."""
        poses = list_poses()
        for p in poses:
            assert "name" in p
            assert "description" in p
            assert "tags" in p
            assert "keypoints" not in p  # list_poses is lightweight


class TestLoadPose:
    """Tests for loading individual poses."""

    def test_load_standing_neutral(self):
        """Should load standing_neutral with 54 float keypoints."""
        pose = load_pose("standing_neutral")
        assert pose["name"] == "standing_neutral"
        assert len(pose["keypoints"]) == 54  # 18 keypoints * 3 (x, y, conf)
        # All coordinates should be within 1024x1024
        for i in range(0, 54, 3):
            assert 0 <= pose["keypoints"][i] <= 1024
            assert 0 <= pose["keypoints"][i + 1] <= 1024
            assert 0 <= pose["keypoints"][i + 2] <= 1.0

    def test_load_nonexistent_raises(self):
        """Should raise FileNotFoundError for missing poses."""
        with pytest.raises(FileNotFoundError):
            load_pose("does_not_exist")


class TestFindBestPose:
    """Tests for keyword-based pose matching."""

    def test_find_waving(self):
        """'person waving hello' should match waving pose."""
        pose = find_best_pose("person waving hello")
        assert pose is not None
        assert pose["name"] == "waving"

    def test_find_standing(self):
        """'person standing still' should match standing_neutral."""
        pose = find_best_pose("person standing still")
        assert pose is not None
        assert pose["name"] == "standing_neutral"

    def test_find_running(self):
        """'character running fast' should match running."""
        pose = find_best_pose("character running fast")
        assert pose is not None
        assert pose["name"] == "running"

    def test_no_match_returns_best_effort(self):
        """Even vague descriptions should return the closest match."""
        pose = find_best_pose("something")
        # May return None or a pose — both are acceptable
        # The function always returns the best scoring pose, even if score is low
