"""Unit tests for API client with mocked OpenAI calls."""

import os
from unittest.mock import patch, MagicMock

import pytest

from pose_director_root.utils.api_client import (
    call_llm,
    call_llm_vision,
    resolve_api_key,
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
    def test_call_llm_empty_key_localhost_resolves(self, mock_openai_cls):
        """Empty api_key with localhost URL resolves to 'not-needed'."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("ok")

        # Clear env vars so the resolver falls through to localhost fallback
        with patch.dict(os.environ, {}, clear=True):
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


class TestRetryLoop:
    """Tests for _retry_loop edge cases."""

    @patch("pose_director_root.utils.api_client.OpenAI")
    def test_empty_content_with_reasoning_fallback(self, mock_openai_cls):
        """If content is empty but reasoning_content exists, return reasoning."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        msg = MagicMock()
        msg.content = ""
        msg.reasoning_content = '{"result": "from reasoning"}'
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]

        mock_client.chat.completions.create.return_value = resp
        result = call_llm(
            base_url="http://test/v1",
            api_key="k",
            model="m",
            system_prompt="",
            user_message="",
        )
        assert result == '{"result": "from reasoning"}'

    @patch("pose_director_root.utils.api_client.time")
    @patch("pose_director_root.utils.api_client.OpenAI")
    def test_api_connection_error_retries(self, mock_openai_cls, mock_time):
        """APIConnectionError should be retried."""
        from openai import APIConnectionError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_time.sleep = MagicMock()
        mock_time.time.side_effect = [0] * 10

        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(message="connection refused", request=MagicMock()),
            _mock_response("recovered"),
        ]

        result = call_llm(
            base_url="http://test/v1",
            api_key="k",
            model="m",
            system_prompt="",
            user_message="",
            max_retries=2,
        )
        assert result == "recovered"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("pose_director_root.utils.api_client.time")
    @patch("pose_director_root.utils.api_client.OpenAI")
    def test_api_status_error_429_retries_with_backoff(self, mock_openai_cls, mock_time):
        """HTTP 429 should trigger exponential backoff and retry."""
        from openai import APIStatusError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_time.sleep = MagicMock()
        mock_time.time.side_effect = [0] * 10

        error_429 = APIStatusError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_client.chat.completions.create.side_effect = [
            error_429,
            error_429,
            _mock_response("ok after rate limit"),
        ]

        result = call_llm(
            base_url="http://test/v1",
            api_key="k",
            model="m",
            system_prompt="",
            user_message="",
            max_retries=3,
        )
        assert result == "ok after rate limit"
        assert mock_client.chat.completions.create.call_count == 3
        # Should have slept at least twice for backoff
        assert mock_time.sleep.call_count >= 2

    @patch("pose_director_root.utils.api_client.time")
    @patch("pose_director_root.utils.api_client.OpenAI")
    def test_api_status_error_503_retries(self, mock_openai_cls, mock_time):
        """HTTP 503 should also trigger retry with backoff."""
        from openai import APIStatusError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_time.sleep = MagicMock()
        mock_time.time.side_effect = [0] * 10

        error_503 = APIStatusError(
            message="service unavailable",
            response=MagicMock(status_code=503),
            body=None,
        )
        mock_client.chat.completions.create.side_effect = [
            error_503,
            _mock_response("ok"),
        ]

        result = call_llm(
            base_url="http://test/v1",
            api_key="k",
            model="m",
            system_prompt="",
            user_message="",
            max_retries=2,
        )
        assert result == "ok"
        assert mock_client.chat.completions.create.call_count == 2


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


class TestResolveApiKey:
    """Tests for resolve_api_key() env var fallback."""

    def test_non_empty_widget_passes_through(self):
        assert resolve_api_key("my-key", "https://api.openai.com/v1") == "my-key"

    def test_whitespace_widget_passes_through(self):
        assert resolve_api_key("  my-key  ", "https://api.openai.com/v1") == "my-key"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key-123"}, clear=False)
    def test_openrouter_env_fallback(self):
        result = resolve_api_key("", "https://openrouter.ai/api/v1")
        assert result == "or-key-123"

    @patch.dict(os.environ, {"GEMINI_API_KEY": "gem-key-456"}, clear=False)
    def test_gemini_env_fallback(self):
        result = resolve_api_key("", "https://generativelanguage.googleapis.com/v1beta/openai/")
        assert result == "gem-key-456"

    @patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "google-key"}, clear=False)
    def test_gemini_secondary_env(self):
        result = resolve_api_key("", "https://generativelanguage.googleapis.com/v1beta/openai/")
        assert result == "google-key"

    @patch.dict(os.environ, {"ZAI_API_KEY": "zai-key"}, clear=False)
    def test_zai_env_fallback(self):
        result = resolve_api_key("", "https://zai.api.com/v1")
        assert result == "zai-key"

    def test_localhost_returns_not_needed(self):
        # Clear any LLM_API_KEY that might be set
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_API_KEY", None)
            os.environ.pop("POSE_DIRECTOR_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            result = resolve_api_key("", "http://localhost:11434/v1")
            assert result == "not-needed"

    def test_localhost_ip_returns_not_needed(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_API_KEY", None)
            os.environ.pop("POSE_DIRECTOR_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            result = resolve_api_key("", "http://127.0.0.1:8001/v1")
            assert result == "not-needed"

    @patch.dict(os.environ, {"POSE_DIRECTOR_API_KEY": "pd-key"}, clear=False)
    def test_generic_fallback_pose_director(self):
        result = resolve_api_key("", "https://unknown-provider.com/v1")
        assert result == "pd-key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "oai-key"}, clear=False)
    def test_generic_fallback_openai(self):
        # Clear more specific keys
        os.environ.pop("POSE_DIRECTOR_API_KEY", None)
        result = resolve_api_key("", "https://unknown-provider.com/v1")
        assert result == "oai-key"

    def test_empty_everything_returns_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_api_key("", "https://unknown-provider.com/v1")
            assert result == ""

    def test_widget_overrides_env(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}, clear=False):
            result = resolve_api_key("widget-key", "https://openrouter.ai/api/v1")
            assert result == "widget-key"
