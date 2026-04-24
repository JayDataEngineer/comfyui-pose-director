"""Unit tests for KeyframeReviewer node with mocked vision LLM calls."""

import json
from unittest.mock import patch

import pytest
import torch

from pose_director_root.nodes.keyframe_reviewer import KeyframeReviewerNode


class TestKeyframeReviewerReview:
    """Tests for KeyframeReviewerNode.review() with mocked vision LLM."""

    @patch("pose_director_root.nodes.keyframe_reviewer.call_llm_vision")
    def test_review_passed(self, mock_vision):
        """Score >= threshold should pass."""
        mock_vision.return_value = json.dumps({
            "passed": True,
            "score": 8,
            "issues": [],
            "summary": "Good consistency",
        })

        node = KeyframeReviewerNode()
        images = torch.rand(3, 64, 64, 3)  # 3 keyframes
        review, passed, score = node.review(
            images=images,
            review_prompt="Review these keyframes",
            pass_threshold=7,
            base_url="http://test/v1",
            api_key="k",
            model="m",
        )

        assert passed is True
        assert score == 8
        assert "PASSED" in review
        assert "Good consistency" in review

    @patch("pose_director_root.nodes.keyframe_reviewer.call_llm_vision")
    def test_review_failed(self, mock_vision):
        """Score < threshold should fail."""
        mock_vision.return_value = json.dumps({
            "passed": True,
            "score": 4,
            "issues": ["Inconsistent hair color", "Jerky motion"],
            "summary": "Needs regeneration",
        })

        node = KeyframeReviewerNode()
        images = torch.rand(2, 64, 64, 3)
        review, passed, score = node.review(
            images=images,
            review_prompt="Review these keyframes",
            pass_threshold=7,
            base_url="http://test/v1",
            api_key="k",
            model="m",
        )

        assert passed is False
        assert score == 4
        assert "FAILED" in review
        assert "Inconsistent hair color" in review

    @patch("pose_director_root.nodes.keyframe_reviewer.call_llm_vision")
    def test_review_invalid_json_fallback(self, mock_vision):
        """If JSON parsing fails, return raw response with failed=False and score=0."""
        raw = "This is a plain text review without JSON"
        mock_vision.return_value = raw

        node = KeyframeReviewerNode()
        images = torch.rand(1, 64, 64, 3)
        review, passed, score = node.review(
            images=images,
            review_prompt="Review",
            pass_threshold=7,
            base_url="http://test/v1",
            api_key="k",
            model="m",
        )

        assert review == raw
        assert passed is False
        assert score == 0

    @patch("pose_director_root.nodes.keyframe_reviewer.call_llm_vision")
    def test_review_single_image(self, mock_vision):
        """Should handle a single keyframe."""
        mock_vision.return_value = json.dumps({
            "passed": True,
            "score": 9,
            "issues": [],
            "summary": "Excellent",
        })

        node = KeyframeReviewerNode()
        images = torch.rand(1, 32, 32, 3)
        review, passed, score = node.review(
            images=images,
            review_prompt="Review",
            pass_threshold=7,
            base_url="http://test/v1",
            api_key="k",
            model="m",
        )

        assert passed is True
        assert score == 9

    @patch("pose_director_root.nodes.keyframe_reviewer.call_llm_vision")
    def test_review_with_issues_listed(self, mock_vision):
        """Issues should be formatted as a bullet list in the review text."""
        mock_vision.return_value = json.dumps({
            "passed": False,
            "score": 5,
            "issues": ["Arm deformation", "Face mismatch"],
            "summary": "Mixed quality",
        })

        node = KeyframeReviewerNode()
        images = torch.rand(4, 64, 64, 3)
        review, passed, score = node.review(
            images=images,
            review_prompt="Review",
            pass_threshold=7,
            base_url="http://test/v1",
            api_key="k",
            model="m",
        )

        assert "Arm deformation" in review
        assert "Face mismatch" in review
        assert passed is False
