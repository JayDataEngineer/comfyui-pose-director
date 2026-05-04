"""Shared ComfyUI API client functions.

Routes through Ray gateway by default (http://localhost:8000/comfyui).
Ray manages ComfyUI as a subprocess with GPU scheduling.

Override with COMFYUI_URL env var:
  - Ray (default):  COMFYUI_URL=http://localhost:8000/comfyui
  - Direct ComfyUI: COMFYUI_URL=http://127.0.0.1:8465
  - Remote Ray:     COMFYUI_URL=https://your-ray-host:8000/comfyui

Functions accept an optional `base_url` parameter for runtime URL override.
When not provided, reads COMFYUI_URL from environment (lazily, per call).
"""

import json
import os
import time

import requests


def _get_url(base_url: str | None = None) -> str:
    """Get ComfyUI base URL — explicit arg overrides env var."""
    if base_url:
        return base_url.rstrip("/")
    return os.environ.get("COMFYUI_URL", "http://localhost:8000/comfyui").rstrip("/")


# Backward-compatible module-level constant (read once at import)
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://localhost:8000/comfyui").rstrip("/")

# Direct ComfyUI address (for rsync, MCP tools, etc. that bypass Ray)
COMFYUI_DIRECT_URL = os.environ.get(
    "COMFYUI_DIRECT_URL", "http://127.0.0.1:8465"
).rstrip("/")

POLL_INTERVAL = 2
MAX_WAIT = 900  # 15 minutes max per workflow (first run includes model loading)


def submit_workflow(workflow: dict, timeout: int = MAX_WAIT, retries: int = 0, base_url: str | None = None) -> dict:
    """Submit a workflow to ComfyUI and wait for completion.

    Returns the output data from /history.
    Raises on errors including validation errors at submit time.
    If retries > 0, will retry on execution errors (e.g. API rate limits).
    """
    url = _get_url(base_url)
    last_error = None
    for attempt in range(retries + 1):
        try:
            return _submit_workflow_once(workflow, timeout, url)
        except RuntimeError as e:
            last_error = e
            err_msg = str(e)
            is_retryable = "ConnectionError" in err_msg or "429" in err_msg or "rate" in err_msg.lower()
            if is_retryable and attempt < retries:
                wait = 10 * (attempt + 1)
                print(f"  Workflow failed (attempt {attempt + 1}/{retries + 1}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise last_error


def _submit_workflow_once(workflow: dict, timeout: int, url: str) -> dict:
    """Single attempt to submit and wait for a workflow."""
    resp = requests.post(f"{url}/prompt", json={"prompt": workflow}, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Submit failed ({resp.status_code}): {resp.text}")
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"Workflow validation error: {json.dumps(data['error'], indent=2)}")
    if "node_errors" in data and data["node_errors"]:
        raise RuntimeError(f"Node errors: {json.dumps(data['node_errors'], indent=2)}")

    prompt_id = data["prompt_id"]

    start = time.time()
    while time.time() - start < timeout:
        hist = requests.get(f"{url}/history/{prompt_id}", timeout=120)
        if hist.status_code == 200 and hist.json().get(prompt_id):
            entry = hist.json()[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"Workflow execution failed: {json.dumps(msgs, indent=2)}")
            if "outputs" in entry:
                return entry
            if status.get("completed", False) and "outputs" not in entry:
                raise RuntimeError(
                    f"Workflow completed but has no outputs. Status: {json.dumps(status, indent=2)}"
                )
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Workflow {prompt_id} timed out after {timeout}s")


def get_output_images(outputs: dict, node_id: str | int, base_url: str | None = None) -> list:
    """Get output images from a SaveImage/PreviewImage node."""
    url = _get_url(base_url)
    node_id = str(node_id)
    node_out = outputs.get(node_id, {})
    images = node_out.get("images", [])
    result = []
    for img_info in images:
        resp = requests.get(
            f"{url}/view",
            params={"filename": img_info["filename"], "subfolder": img_info.get("subfolder", ""), "type": img_info.get("type", "output")},
            timeout=30,
        )
        assert resp.status_code == 200
        result.append(resp.content)
    return result


def get_output_video(outputs: dict, node_id: str | int, base_url: str | None = None) -> bytes | None:
    """Get output video from VHSVideoCombine node."""
    url = _get_url(base_url)
    node_id = str(node_id)
    node_out = outputs.get(node_id, {})
    gifs = node_out.get("gifs", [])
    if not gifs:
        return None
    resp = requests.get(
        f"{url}/view",
        params={"filename": gifs[0]["filename"], "subfolder": gifs[0].get("subfolder", ""), "type": gifs[0].get("type", "output")},
        timeout=30,
    )
    return resp.content if resp.status_code == 200 else None


def get_output_text(outputs: dict, node_id: str | int, base_url: str | None = None) -> str | None:
    """Fetch STRING output from a node."""
    node_out = outputs.get(str(node_id), {})
    texts = node_out.get("text", [])
    return texts[0] if texts else None


def upload_image(image_bytes: bytes, filename: str, subfolder: str = "PoseDirector", base_url: str | None = None) -> None:
    """Upload an image to ComfyUI's input directory."""
    url = _get_url(base_url)
    resp = requests.post(
        f"{url}/upload/image",
        files={"image": (filename, image_bytes, "image/png")},
        data={"overwrite": "true", "subfolder": subfolder},
    )
    assert resp.status_code == 200, f"Upload failed: {resp.status_code} {resp.text}"
