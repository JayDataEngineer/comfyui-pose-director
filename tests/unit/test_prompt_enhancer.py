"""Unit tests for PromptEnhancer node."""

from unittest.mock import patch, MagicMock

from pose_director_root.nodes.prompt_enhancer import PromptEnhancerNode, _VARIANT_RULES


class TestPromptEnhancerMetadata:
    """Test node metadata and registration."""

    def test_return_types(self):
        assert PromptEnhancerNode.RETURN_TYPES == ("STRING",)

    def test_output_names(self):
        assert PromptEnhancerNode.OUTPUT_NAMES == ("enhanced_prompt",)

    def test_function(self):
        assert PromptEnhancerNode.FUNCTION == "enhance"

    def test_category(self):
        assert PromptEnhancerNode.CATEGORY == "pose_director"

    def test_input_types_callable(self):
        inputs = PromptEnhancerNode.INPUT_TYPES()
        assert "required" in inputs
        required = inputs["required"]
        assert "prompt" in required
        assert "variant" in required
        assert "base_url" in required
        assert "api_key" in required
        assert "model" in required

    def test_variant_options(self):
        inputs = PromptEnhancerNode.INPUT_TYPES()
        variant_def = inputs["required"]["variant"]
        assert isinstance(variant_def, tuple)
        options = variant_def[0]
        assert "turbo" in options
        assert "base" in options


class TestPromptEnhancerEnhance:
    """Test the enhance method with mocked LLM calls."""

    @patch("pose_director_root.nodes.prompt_enhancer.call_llm")
    def test_enhance_turbo_returns_string(self, mock_call_llm):
        mock_call_llm.return_value = "A fierce warrior woman clad in weathered leather armor"

        node = PromptEnhancerNode()
        (result,) = node.enhance(
            prompt="warrior woman in forest",
            variant="turbo",
            base_url="https://openrouter.ai/api/v1",
            api_key="",
            model="deepseek/deepseek-v4-flash",
        )

        assert result == "A fierce warrior woman clad in weathered leather armor"
        mock_call_llm.assert_called_once()
        call_kwargs = mock_call_llm.call_args
        assert call_kwargs.kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert call_kwargs.kwargs["model"] == "deepseek/deepseek-v4-flash"

    @patch("pose_director_root.nodes.prompt_enhancer.call_llm")
    def test_enhance_base_returns_string(self, mock_call_llm):
        mock_call_llm.return_value = "Enhanced prompt here\n---\nnegative prompt here"

        node = PromptEnhancerNode()
        (result,) = node.enhance(
            prompt="a cat",
            variant="base",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
            model="test-model",
        )

        assert "Enhanced prompt here" in result
        mock_call_llm.assert_called_once()

    @patch("pose_director_root.nodes.prompt_enhancer.call_llm")
    def test_enhance_turbo_system_prompt_contains_key_rules(self, mock_call_llm):
        mock_call_llm.return_value = "enhanced"

        node = PromptEnhancerNode()
        node.enhance(
            prompt="test",
            variant="turbo",
            base_url="http://localhost/v1",
            api_key="",
            model="m",
        )

        system_prompt = mock_call_llm.call_args.kwargs["system_prompt"]
        # Should contain key Z-Image knowledge
        assert "S3-DiT" in system_prompt
        assert "single-stream" in system_prompt.lower()
        assert "512 tokens" in system_prompt
        assert "80-200 words" in system_prompt
        # Should contain turbo-specific rules
        assert "CFG MUST be 0.0" in system_prompt
        assert "Do NOT generate a negative prompt" in system_prompt
        # Should contain enhancement methodology
        assert "Lock core elements" in system_prompt
        assert "No meta-tags" in system_prompt

    @patch("pose_director_root.nodes.prompt_enhancer.call_llm")
    def test_enhance_base_system_prompt_has_negative_advice(self, mock_call_llm):
        mock_call_llm.return_value = "enhanced"

        node = PromptEnhancerNode()
        node.enhance(
            prompt="test",
            variant="base",
            base_url="http://localhost/v1",
            api_key="",
            model="m",
        )

        system_prompt = mock_call_llm.call_args.kwargs["system_prompt"]
        assert "negative prompt" in system_prompt.lower()
        assert "CFG 3.0-5.0" in system_prompt

    @patch("pose_director_root.nodes.prompt_enhancer.call_llm")
    def test_enhance_strips_whitespace(self, mock_call_llm):
        mock_call_llm.return_value = "  Enhanced prompt  \n\n"

        node = PromptEnhancerNode()
        (result,) = node.enhance(
            prompt="test",
            variant="turbo",
            base_url="http://localhost/v1",
            api_key="",
            model="m",
        )

        assert result == "Enhanced prompt"

    @patch("pose_director_root.nodes.prompt_enhancer.call_llm")
    def test_enhance_passes_user_prompt_as_message(self, mock_call_llm):
        mock_call_llm.return_value = "enhanced"

        node = PromptEnhancerNode()
        node.enhance(
            prompt="A young woman with red hair holding a sword",
            variant="turbo",
            base_url="http://localhost/v1",
            api_key="",
            model="m",
        )

        user_message = mock_call_llm.call_args.kwargs["user_message"]
        assert user_message == "A young woman with red hair holding a sword"

    def test_variant_rules_cover_both_variants(self):
        assert "turbo" in _VARIANT_RULES
        assert "base" in _VARIANT_RULES
        assert "negative" not in _VARIANT_RULES["turbo"].lower() or "Do NOT" in _VARIANT_RULES["turbo"]
        assert "negative" in _VARIANT_RULES["base"].lower()
