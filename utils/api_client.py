"""Shared OpenAI-compatible API client with retry logic."""

import base64
import io
import json
import logging
import time

import torch
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError

logger = logging.getLogger("comfyui-pose-director")

DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 300


def _tensor_to_base64(image_tensor):
    """Convert a ComfyUI image tensor (H, W, 3) float32 0-1 to a base64 PNG string."""
    from PIL import Image as PILImage
    import numpy as np
    img = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = PILImage.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _retry_loop(client, model, messages, temperature, max_retries):
    """Shared retry loop for API calls."""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=16384,
            )
            msg = response.choices[0].message
            content = msg.content
            # For thinking models (deepseek reasoning format), the actual
            # answer goes in .content and thinking in .reasoning_content.
            # If content is empty, the model spent all tokens thinking — retry.
            if content and content.strip():
                return content
            # Fallback: some models put JSON in reasoning_content
            reasoning = getattr(msg, 'reasoning_content', None) or ''
            if reasoning and reasoning.strip():
                return reasoning
            logger.warning(f"LLM returned empty content (attempt {attempt + 1}/{max_retries})")
            last_error = RuntimeError("Empty LLM response")
        except APITimeoutError as e:
            last_error = e
            logger.warning(f"LLM request timed out (attempt {attempt + 1}/{max_retries})")
        except APIConnectionError as e:
            last_error = e
            logger.warning(f"LLM connection failed (attempt {attempt + 1}/{max_retries}): {e}")
        except APIStatusError as e:
            if e.status_code in (429, 503):
                last_error = e
                wait = min(2 ** attempt, 30)
                logger.warning(f"LLM rate limited / unavailable (attempt {attempt + 1}/{max_retries}), waiting {wait}s")
                time.sleep(wait)
            else:
                raise

        if attempt < max_retries - 1:
            time.sleep(min(2 ** attempt, 10))

    raise ConnectionError(f"LLM request failed after {max_retries} attempts: {last_error}")


def call_llm(base_url, api_key, model, system_prompt, user_message,
             temperature=0.7, max_retries=DEFAULT_MAX_RETRIES, timeout=DEFAULT_TIMEOUT):
    """Call an OpenAI-compatible API endpoint with retry logic.

    Args:
        base_url: API base URL (e.g. "http://localhost:11434/v1" or "https://api.openai.com/v1")
        api_key: API key (can be empty for local models)
        model: Model name (e.g. "gemma3:12b", "gpt-4o")
        system_prompt: System message
        user_message: User message
        temperature: Sampling temperature
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        str: The assistant's response text
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key or "not-needed",
        timeout=timeout,
        max_retries=0,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return _retry_loop(client, model, messages, temperature, max_retries)


def call_llm_vision(base_url, api_key, model, system_prompt, text_prompt, images,
                    temperature=0.7, max_retries=DEFAULT_MAX_RETRIES, timeout=DEFAULT_TIMEOUT):
    """Call an OpenAI-compatible vision API with images.

    Args:
        base_url: API base URL
        api_key: API key (can be empty for local models)
        model: Model name (must support vision, e.g. "models/gemini-2.5-flash")
        system_prompt: System message
        text_prompt: User text prompt
        images: List of ComfyUI image tensors, each (H, W, 3) float32 0-1
        temperature: Sampling temperature
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        str: The assistant's response text
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key or "not-needed",
        timeout=timeout,
        max_retries=0,
    )

    # Build user message with text + images
    content = [{"type": "text", "text": text_prompt}]
    for img_tensor in images:
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]  # Remove batch dim if present
        b64 = _tensor_to_base64(img_tensor)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    return _retry_loop(client, model, messages, temperature, max_retries)
