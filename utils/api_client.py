"""Shared OpenAI-compatible API client with retry logic."""

import base64
import io
import json
import logging
import os
import time
from pathlib import Path

import torch
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError

logger = logging.getLogger("comfyui-pose-director")

DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 300

# Load .env on import so keys are available when running inside ComfyUI
# (tests load it via conftest, so this is a no-op there)
_ENV_FILE = Path(__file__).parent.parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

# Map base_url patterns to environment variable names (checked in order)
_PROVIDER_ENV_MAP = {
    "openrouter": ["OPENROUTER_API_KEY"],
    "generativelanguage": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "zai": ["ZAI_API_KEY"],
    "localhost": ["LLM_API_KEY"],
    "127.0.0.1": ["LLM_API_KEY"],
}

# Map base_url patterns to ComfyUI frontend setting IDs
_PROVIDER_SETTINGS_MAP = {
    "openrouter": ["PoseDirector.openrouterApiKey"],
    "generativelanguage": ["PoseDirector.geminiApiKey"],
}

# Cached ComfyUI settings (loaded once, from user/default/comfy.settings.json)
_comfy_settings = None


def _load_comfy_settings():
    """Load ComfyUI frontend settings from comfy.settings.json (cached)."""
    global _comfy_settings
    if _comfy_settings is not None:
        return _comfy_settings

    _comfy_settings = {}
    try:
        # ComfyUI stores settings at <comfyui_root>/user/default/comfy.settings.json
        comfy_root = Path(__file__).parent.parent.parent.parent
        settings_path = comfy_root / "user" / "default" / "comfy.settings.json"
        if settings_path.exists():
            with open(settings_path) as f:
                _comfy_settings = json.load(f)
    except Exception:
        pass
    return _comfy_settings


def resolve_api_key(api_key: str, base_url: str = "") -> str:
    """Resolve API key from env vars / settings when the widget value is empty.

    Priority: widget value > provider env vars > ComfyUI settings > generic env vars.
    Local endpoints (localhost/127.0.0.1) get a fallback "not-needed" key.
    """
    if api_key and api_key.strip():
        return api_key.strip()

    base_lower = (base_url or "").lower()

    # 1. Provider-specific env vars
    for pattern, env_vars in _PROVIDER_ENV_MAP.items():
        if pattern in base_lower:
            for var in env_vars:
                val = os.environ.get(var, "")
                if val:
                    return val

    # 2. ComfyUI frontend settings
    settings = _load_comfy_settings()
    for pattern, setting_ids in _PROVIDER_SETTINGS_MAP.items():
        if pattern in base_lower:
            for sid in setting_ids:
                val = settings.get(sid, "")
                if val:
                    return val

    # 3. Generic env var fallbacks
    for var in ("POSE_DIRECTOR_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"):
        val = os.environ.get(var, "")
        if val:
            return val

    # Local models don't need a real key
    if any(p in base_lower for p in ("localhost", "127.0.0.1")):
        return "not-needed"

    return api_key  # empty — will fail downstream with a clear auth error


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
    resolved_key = resolve_api_key(api_key, base_url)
    client = OpenAI(
        base_url=base_url,
        api_key=resolved_key,
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
    resolved_key = resolve_api_key(api_key, base_url)
    client = OpenAI(
        base_url=base_url,
        api_key=resolved_key,
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
