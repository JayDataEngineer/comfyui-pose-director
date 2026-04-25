"""Shared ComfyUI API client functions."""

import json
import os
import subprocess
import time
from pathlib import Path

import requests

COMFYUI_URL = "http://127.0.0.1:8465"
POLL_INTERVAL = 2
MAX_WAIT = 900  # 15 minutes max per workflow (first run includes model loading)

CUI_DIR = "/home/ubuntu/Documents/programs/img/comfyui"
CUI_PYTHON = os.path.join(CUI_DIR, ".venv/bin/python")
CUI_PORT = 8465


def restart_comfyui(timeout=120):
    """Kill ComfyUI, wait for CUDA release, start a fresh instance."""
    print("[ComfyUI] Stopping...")
    subprocess.run(["pkill", "-f", f"main.py.*{CUI_PORT}"], check=False)
    time.sleep(5)  # wait for CUDA memory release
    env = {**os.environ, "PYTHONWARNINGS": "ignore::FutureWarning,ignore::SyntaxWarning"}
    log_f = open("/tmp/cui_restart.log", "w")
    subprocess.Popen(
        [CUI_PYTHON, "main.py", "--port", str(CUI_PORT),
         "--use-flash-attention", "--dont-upcast-attention"],
        cwd=CUI_DIR, stdout=log_f, stderr=subprocess.STDOUT, env=env,
    )
    print("[ComfyUI] Waiting for startup...")
    for _ in range(timeout // 2):
        time.sleep(2)
        try:
            r = requests.get(f"http://127.0.0.1:{CUI_PORT}/system_stats", timeout=2)
            if r.status_code == 200:
                print("[ComfyUI] Ready.")
                time.sleep(2)  # extra settling
                return
        except Exception:
            pass
    raise RuntimeError(f"ComfyUI failed to restart within {timeout}s")


def stop_llama():
    """Stop the orchestrator-llama Docker container to free VRAM."""
    print("[llama.cpp] Stopping...")
    subprocess.run(["docker", "stop", "orchestrator-llama"], check=False, timeout=30)
    time.sleep(3)


def start_llama():
    """Start the orchestrator-llama Docker container."""
    print("[llama.cpp] Starting...")
    subprocess.run(["docker", "start", "orchestrator-llama"], check=False, timeout=30)
    time.sleep(5)


def submit_workflow(workflow: dict, timeout: int = MAX_WAIT, retries: int = 0) -> dict:
    """Submit a workflow to ComfyUI and wait for completion.

    Returns the output data from /history.
    Raises on errors including validation errors at submit time.
    If retries > 0, will retry on execution errors (e.g. API rate limits).
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            return _submit_workflow_once(workflow, timeout)
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


def _submit_workflow_once(workflow: dict, timeout: int) -> dict:
    """Single attempt to submit and wait for a workflow."""
    resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=120)
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
        hist = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=120)
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


def get_output_images(outputs: dict, node_id: str | int) -> list:
    """Get output images from a SaveImage/PreviewImage node."""
    node_id = str(node_id)
    node_out = outputs.get(node_id, {})
    images = node_out.get("images", [])
    result = []
    for img_info in images:
        resp = requests.get(
            f"{COMFYUI_URL}/view",
            params={"filename": img_info["filename"], "subfolder": img_info.get("subfolder", ""), "type": img_info.get("type", "output")},
            timeout=30,
        )
        assert resp.status_code == 200
        result.append(resp.content)
    return result


def get_output_video(outputs: dict, node_id: str | int) -> bytes | None:
    """Get output video from VHSVideoCombine node."""
    node_id = str(node_id)
    node_out = outputs.get(node_id, {})
    gifs = node_out.get("gifs", [])
    if not gifs:
        return None
    resp = requests.get(
        f"{COMFYUI_URL}/view",
        params={"filename": gifs[0]["filename"], "subfolder": gifs[0].get("subfolder", ""), "type": gifs[0].get("type", "output")},
        timeout=30,
    )
    return resp.content if resp.status_code == 200 else None


def get_output_text(outputs: dict, node_id: str | int) -> str | None:
    """Fetch STRING output from a node."""
    node_out = outputs.get(str(node_id), {})
    texts = node_out.get("text", [])
    return texts[0] if texts else None


def upload_image(image_bytes: bytes, filename: str, subfolder: str = "PoseDirector") -> None:
    """Upload an image to ComfyUI's input directory."""
    resp = requests.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (filename, image_bytes, "image/png")},
        data={"overwrite": "true", "subfolder": subfolder},
    )
    assert resp.status_code == 200, f"Upload failed: {resp.status_code} {resp.text}"
