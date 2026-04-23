"""Unit tests for API client with mocked OpenAI calls."""

from unittest.mock import patch, MagicMock

import pytest

from pose_director_root.utils.api_client import (
    call_llm,
    call_llm_vision,
    _tensor_to_base64,
)


def _mock_response(content="test response"):
    """Create a mock OpenAI API response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestCallLlm:
    """Tests for call_llm() with mocked OpenAI client."""

    @patch("pose_director_root.utils.api_client.OpenAI")
    def test_call_llm_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("Hello!")

        result = call_llm(
            base_url="http://localhost:11434/v1",
            api_key="test-key",
            model="test-model",
            system_prompt="You are helpful.",
            user_message="Hi",
        )
        assert result == "Hello!"

        # Verify client was created with correct params
        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="test-key",
            timeout=300,
            max_retries=0,
        )

    @patch("pose_director_root.utils.api_client.OpenAI")
    def test_call_llm_empty_key_uses_placeholder(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("ok")

        call_llm(
            base_url="http://localhost:11434/v1",
            api_key="",
            model="test",
            system_prompt="sys",
            user_message="hi",
        )
        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="not-needed",
            timeout=300,
            max_retries=0,
        )

    @patch("pose_director_root.utils.api_client.time")
    @patch("pose_director_root.utils.api_client.OpenAI")
    def test_call_llm_retries_on_timeout(self, mock_openai_cls, mock_time):
        from openai import APITimeoutError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_time.sleep = MagicMock()
        mock_time.time.side_effect = [0, 0, 0, 0]  # Prevent infinite loop

        # Fail twice, succeed third time
        mock_client.chat.completions.create.side_effect = [
            APITimeoutError("timeout"),
            APITimeoutError("timeout"),
            _mock_response("recovered"),
        ]

        result = call_llm(
            base_url="http://test/v1",
            api_key="k",
            model="m",
            system_prompt="",
            user_message="",
            max_retries=3,
        )
        assert result == "recovered"
        assert mock_client.chat.completions.create.call_count == 3

    @patch("pose_director_root.utils.api_client.time")
    @patch("pose_director_root.utils.api_client.OpenAI")
    def test_call_llm_raises_after_max_retries(self, mock_openai_cls, mock_time):
        from openai import APITimeoutError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_time.sleep = MagicMock()
        mock_time.time.side_effect = [0] * 10

        mock_client.chat.completions.create.side_effect = APITimeoutError("timeout")

        with pytest.raises(ConnectionError, match="LLM request failed after 2 attempts"):
            call_llm(
                base_url="http://test/v1",
                api_key="k",
                model="m",
                system_prompt="",
                user_message="",
                max_retries=2,
            )

    @patch("pose_director_root.utils.api_client.OpenAI")
    def test_call_llm_raises_on_non_retryable_error(self, mock_openai_cls):
        from openai import APIStatusError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        error = APIStatusError(
            message="Bad request",
            response=MagicMock(status_code=400),
            body=None,
        )
        mock_client.chat.completions.create.side_effect = error

        with pytest.raises(APIStatusError):
            call_llm(
                base_url="http://test/v1",
                api_key="k",
                model="m",
                system_prompt="",
                user_message="",
                max_retries=3,
            )


class TestCallLlmVision:
    """Tests for call_llm_vision() with mocked OpenAI client."""

    @patch("pose_director_root.utils.api_client.OpenAI")
    def test_vision_sends_text_and_images(self, mock_openai_cls):
        import torch

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("I see images")

        img_tensor = torch.rand(64, 64, 3)  # Small test image
        result = call_llm_vision(
            base_url="http://test/v1",
            api_key="k",
            model="vision-model",
            system_prompt="Describe images.",
            text_prompt="What do you see?",
            images=[img_tensor],
        )
        assert result == "I see images"

        # Verify the message content includes image
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert len(user_content) == 2  # text + 1 image
        assert user_content[0]["type"] == "text"
        assert user_content[1]["type"] == "image_url"
        assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    @patch("pose_director_root.utils.api_client.OpenAI")
    def test_vision_handles_batch_dimension(self, mock_openai_cls):
        import torch

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("ok")

        img_tensor = torch.rand(1, 64, 64, 3)  # Batch dim present
        call_llm_vision(
            base_url="http://test/v1",
            api_key="k",
            model="m",
            system_prompt="",
            text_prompt="",
            images=[img_tensor],
        )

        # Should not raise — batch dim stripped internally


class TestTensorToBase64:
    """Tests for _tensor_to_base64() helper."""

    def test_produces_valid_base64(self):
        import base64
        import torch

        tensor = torch.rand(64, 64, 3)
        result = _tensor_to_base64(tensor)

        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

        # Should be a PNG (starts with PNG magic bytes)
        assert decoded[:4] == b'\x89PNG'

    def test_handles_zero_tensor(self):
        import torch

        tensor = torch.zeros(32, 32, 3)
        result = _tensor_to_base64(tensor)
        assert len(result) > 0
