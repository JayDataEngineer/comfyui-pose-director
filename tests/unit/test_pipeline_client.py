"""Unit tests for pipeline/comfyui_client.py."""

import json
from unittest.mock import patch, MagicMock

import pytest

from pipeline.comfyui_client import (
    submit_workflow,
    get_output_images,
    get_output_video,
    get_output_text,
    upload_image,
    COMFYUI_URL,
)


class TestSubmitWorkflow:
    """Tests for submit_workflow()."""

    @patch("pipeline.comfyui_client.requests.post")
    @patch("pipeline.comfyui_client.requests.get")
    def test_success(self, mock_get, mock_post):
        """Successful workflow execution."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"prompt_id": "abc123"},
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"abc123": {"outputs": {"10": {"images": []}}}},
        )
        result = submit_workflow({"1": {"class_type": "Test", "inputs": {}}}, timeout=30)
        assert "outputs" in result

    @patch("pipeline.comfyui_client.requests.post")
    def test_submit_validation_error(self, mock_post):
        """Validation error at submit time should raise."""
        mock_post.return_value = MagicMock(
            status_code=400,
            text="Bad request",
        )
        with pytest.raises(RuntimeError, match="Submit failed"):
            submit_workflow({"1": {"class_type": "BadNode", "inputs": {}}})

    @patch("pipeline.comfyui_client.requests.post")
    def test_node_errors(self, mock_post):
        """Node errors in response should raise."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "prompt_id": "p1",
                "node_errors": {"1": "Unknown node type"},
            },
        )
        with pytest.raises(RuntimeError, match="Node errors"):
            submit_workflow({"1": {"class_type": "Bad", "inputs": {}}})

    @patch("pipeline.comfyui_client.requests.post")
    @patch("pipeline.comfyui_client.requests.get")
    def test_execution_error(self, mock_get, mock_post):
        """Execution failure should raise."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"prompt_id": "p2"},
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"p2": {"status": {"status_str": "error", "messages": [["execution_error", "OOM"]]}}},
        )
        with pytest.raises(RuntimeError, match="execution failed"):
            submit_workflow({"1": {"class_type": "Test", "inputs": {}}}, timeout=30)

    @patch("pipeline.comfyui_client.time.sleep", side_effect=lambda x: None)
    @patch("pipeline.comfyui_client.requests.post")
    @patch("pipeline.comfyui_client.requests.get")
    def test_timeout(self, mock_get, mock_post, mock_sleep):
        """Timeout should raise TimeoutError."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"prompt_id": "p3"},
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {},
        )
        with pytest.raises(TimeoutError, match="timed out"):
            submit_workflow({"1": {"class_type": "Test", "inputs": {}}}, timeout=2)

    @patch("pipeline.comfyui_client.time.sleep", side_effect=lambda x: None)
    @patch("pipeline.comfyui_client.requests.post")
    @patch("pipeline.comfyui_client.requests.get")
    def test_retries_on_429(self, mock_get, mock_post, mock_sleep):
        """Should retry on rate-limit errors."""
        mock_post.side_effect = [
            MagicMock(status_code=200, json=lambda: {"prompt_id": "p4"}),
        ]
        # First call: execution error with 429
        mock_get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"p4": {
                "status": {"status_str": "error", "messages": [["429", "rate limit"]]},
            }}),
            MagicMock(status_code=200, json=lambda: {"p4": {"outputs": {"1": {}}}}),
        ]
        # The retry will submit again
        mock_post.side_effect = [
            MagicMock(status_code=200, json=lambda: {"prompt_id": "p4"}),
            MagicMock(status_code=200, json=lambda: {"prompt_id": "p4_retry"}),
        ]
        mock_get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"p4": {
                "status": {"status_str": "error", "messages": [["429", "rate limit"]]},
            }}),
            MagicMock(status_code=200, json=lambda: {"p4_retry": {"outputs": {"1": {}}}}),
        ]
        result = submit_workflow({"1": {"class_type": "Test", "inputs": {}}}, timeout=30, retries=1)
        assert "outputs" in result


class TestGetOutputImages:
    """Tests for get_output_images()."""

    @patch("pipeline.comfyui_client.requests.get")
    def test_extracts_images(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, content=b"pngdata")
        outputs = {"14": {"images": [{"filename": "test.png", "subfolder": "", "type": "output"}]}}
        images = get_output_images(outputs, "14")
        assert len(images) == 1
        assert images[0] == b"pngdata"

    @patch("pipeline.comfyui_client.requests.get")
    def test_empty_node(self, mock_get):
        outputs = {"99": {}}
        images = get_output_images(outputs, "99")
        assert images == []

    @patch("pipeline.comfyui_client.requests.get")
    def test_missing_node(self, mock_get):
        images = get_output_images({}, "14")
        assert images == []


class TestGetOutputVideo:
    """Tests for get_output_video()."""

    @patch("pipeline.comfyui_client.requests.get")
    def test_extracts_video(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, content=b"mp4data")
        outputs = {"16": {"gifs": [{"filename": "test.mp4", "subfolder": "", "type": "output"}]}}
        video = get_output_video(outputs, "16")
        assert video == b"mp4data"

    def test_no_video_returns_none(self):
        outputs = {"16": {}}
        video = get_output_video(outputs, "16")
        assert video is None


class TestGetOutputText:
    """Tests for get_output_text()."""

    def test_extracts_text(self):
        outputs = {"3": {"text": ["hello world"]}}
        assert get_output_text(outputs, "3") == "hello world"

    def test_empty_returns_none(self):
        assert get_output_text({}, "3") is None

    def test_no_text_key_returns_none(self):
        outputs = {"3": {}}
        assert get_output_text(outputs, "3") is None


class TestUploadImage:
    """Tests for upload_image()."""

    @patch("pipeline.comfyui_client.requests.post")
    def test_upload_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        upload_image(b"pngdata", "test.png", "sub")
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["data"]["subfolder"] == "sub"

    @patch("pipeline.comfyui_client.requests.post")
    def test_upload_failure_raises(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500, text="Server error")
        with pytest.raises(AssertionError):
            upload_image(b"pngdata", "test.png")
