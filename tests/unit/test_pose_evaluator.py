"""Unit tests for pose evaluator module."""

import pytest

from pose_director_root.utils.pose_evaluator import (
    build_correction_prompt,
    is_evaluation_passing,
)


class TestIsEvaluationPassing:
    def test_explicit_pass(self):
        assert is_evaluation_passing({"overall_assessment": "pass"})

    def test_explicit_fail(self):
        assert not is_evaluation_passing({"overall_assessment": "fail"})

    def test_all_frames_match(self):
        ev = {"poses": [
            {"frame": 0, "matches_intent": True},
            {"frame": 16, "matches_intent": True},
        ]}
        assert is_evaluation_passing(ev)

    def test_some_frames_fail(self):
        ev = {"poses": [
            {"frame": 0, "matches_intent": True},
            {"frame": 16, "matches_intent": False},
        ]}
        assert not is_evaluation_passing(ev)

    def test_empty_evaluation_passes(self):
        assert is_evaluation_passing({})

    def test_case_insensitive(self):
        assert is_evaluation_passing({"overall_assessment": "PASS"})
        assert not is_evaluation_passing({"overall_assessment": "FAIL"})


class TestBuildCorrectionPrompt:
    def test_no_problems_returns_empty(self):
        assert build_correction_prompt({"key_problems": []}, []) == ""

    def test_includes_key_problems(self):
        ev = {"key_problems": ["Arms still at sides", "No visible wave"]}
        result = build_correction_prompt(ev, [])
        assert "Arms still at sides" in result
        assert "No visible wave" in result
        assert "PROBLEMS FOUND" in result

    def test_includes_per_frame_corrections(self):
        ev = {
            "key_problems": ["T-pose unchanged"],
            "poses": [
                {"frame": 16, "matches_intent": False,
                 "issues": ["Right arm not raised"],
                 "corrections": {"right_shoulder": "Set Z to +90 to raise arm"}},
            ],
        }
        result = build_correction_prompt(ev, [])
        assert "Frame 16 issues" in result
        assert "Right arm not raised" in result
        assert "right_shoulder" in result
        assert "Set Z to +90" in result

    def test_ignores_passing_frames(self):
        ev = {
            "key_problems": [],
            "poses": [
                {"frame": 0, "matches_intent": True, "issues": [], "corrections": {}},
                {"frame": 16, "matches_intent": True, "issues": [], "corrections": {}},
            ],
        }
        result = build_correction_prompt(ev, [])
        assert result == ""

    def test_includes_axis_reminders(self):
        ev = {"key_problems": ["test"]}
        result = build_correction_prompt(ev, [])
        assert "right_shoulder Z=+90" in result
        assert "LARGE rotation values" in result
