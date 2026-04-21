"""Shared OpenAI-compatible API client."""

import json
import urllib.request
import urllib.error


def call_llm(base_url, api_key, model, system_prompt, user_message, temperature=0.7):
    """Call an OpenAI-compatible API endpoint.

    Args:
        base_url: API base URL (e.g. "http://localhost:11434/v1" or "https://api.openai.com/v1")
        api_key: API key (can be empty for local models)
        model: Model name (e.g. "gemma3:12b", "gpt-4o")
        system_prompt: System message
        user_message: User message
        temperature: Sampling temperature

    Returns:
        str: The assistant's response text
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
