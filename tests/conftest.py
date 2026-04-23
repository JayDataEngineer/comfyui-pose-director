"""Shared fixtures and ComfyUI API client for integration tests."""

import importlib
import io
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
import requests

# Make the project root importable as a Python package.
# The codebase uses relative imports (from ..utils) which require the root
# to be a proper parent package. We register it as "pose_director_root" to
# avoid issues with the hyphenated directory name.
PROJECT_ROOT = Path(__file__).parent.parent
ROOT_PKG = "pose_director_root"

# Register the project root as a package so relative imports work
if ROOT_PKG not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        ROOT_PKG,
        str(PROJECT_ROOT / "__init__.py"),
        submodule_search_locations=[str(PROJECT_ROOT)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[ROOT_PKG] = pkg
    spec.loader.exec_module(pkg)

# Also make subpackages available at their normal names for convenience
for sub in ["nodes", "utils"]:
    mod_name = f"{ROOT_PKG}.{sub}"
    if mod_name not in sys.modules:
        sub_path = PROJECT_ROOT / sub
        sub_spec = importlib.util.spec_from_file_location(
            mod_name,
            str(sub_path / "__init__.py"),
            submodule_search_locations=[str(sub_path)],
        )
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sys.modules[mod_name] = sub_mod
        sub_spec.loader.exec_module(sub_mod)

COMFYUI_URL = "http://127.0.0.1:8465"
POLL_INTERVAL = 2
MAX_WAIT = 600  # 10 minutes max per workflow

WORKFLOWS_DIR = PROJECT_ROOT / "workflows"

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
            # Only retry on execution errors (API connection/rate-limit), not validation errors
            is_retryable = "ConnectionError" in err_msg or "429" in err_msg or "rate" in err_msg.lower()
            if is_retryable and attempt < retries:
                wait = 10 * (attempt + 1)
                print(f"  Workflow failed (attempt {attempt + 1}/{retries + 1}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise last_error  # unreachable but keeps type checkers happy


def _submit_workflow_once(workflow: dict, timeout: int) -> dict:
    """Single attempt to submit and wait for a workflow."""
    resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Submit failed ({resp.status_code}): {resp.text}")
    data = resp.json()

    # Check for validation errors returned in the response
    if "error" in data:
        raise RuntimeError(f"Workflow validation error: {json.dumps(data['error'], indent=2)}")
    if "node_errors" in data and data["node_errors"]:
        raise RuntimeError(f"Node errors: {json.dumps(data['node_errors'], indent=2)}")

    prompt_id = data["prompt_id"]

    start = time.time()
    while time.time() - start < timeout:
        hist = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=30)
        if hist.status_code == 200 and hist.json().get(prompt_id):
            entry = hist.json()[prompt_id]
            status = entry.get("status", {})
            # Always check for execution errors first, even if outputs exist
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"Workflow execution failed: {json.dumps(msgs, indent=2)}")
            if "outputs" in entry:
                return entry
            # Check for completion without outputs
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


def upload_image(image_bytes: bytes, filename: str, subfolder: str = "PoseDirector") -> None:
    """Upload an image to ComfyUI's input directory."""
    resp = requests.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (filename, image_bytes, "image/png")},
        data={"overwrite": "true", "subfolder": subfolder},
    )
    assert resp.status_code == 200, f"Upload failed: {resp.status_code} {resp.text}"


def convert_ui_workflow_to_api(ui_json: dict) -> dict:
    """Convert a ComfyUI UI-format workflow JSON to API format.

    UI format has "nodes" and "links" arrays.
    API format is {"node_id": {"class_type": "...", "inputs": {...}}}

    Widget values are mapped to inputs by position. Links are resolved
    to node references ["source_id", output_index].
    """
    nodes_by_id = {}
    for node in ui_json.get("nodes", []):
        nodes_by_id[str(node["id"])] = {
            "class_type": node["type"],
            "widgets_values": node.get("widgets_values", []),
        }

    # Build link lookup: link_id -> (source_node_id, source_output, target_node_id, target_input_idx)
    link_map = {}
    for link in ui_json.get("links", []):
        link_id, src_id, src_out, tgt_id, tgt_in, dtype = link
        link_map[link_id] = {
            "src_id": str(src_id),
            "src_out": src_out,
            "tgt_id": str(tgt_id),
            "tgt_in": tgt_in,
            "type": dtype,
        }

    api_workflow = {}
    for node_id, node_data in nodes_by_id.items():
        entry = {"class_type": node_data["class_type"], "inputs": {}}

        # Collect incoming links for this node, sorted by target input index
        incoming = []
        for link_id, link in link_map.items():
            if link["tgt_id"] == node_id:
                incoming.append(link)
        incoming.sort(key=lambda l: l["tgt_in"])

        # Determine how many inputs come from links vs widgets
        num_link_inputs = len(incoming)

        # Add link inputs
        for link in incoming:
            entry["inputs"][f"input_{link['tgt_in']}"] = [link["src_id"], link["src_out"]]

        # Add widget values for inputs NOT covered by links
        widget_vals = node_data.get("widgets_values", [])
        for idx, val in enumerate(widget_vals):
            input_key = f"input_{num_link_inputs + idx}"
            entry["inputs"][input_key] = val

        api_workflow[node_id] = entry

    return api_workflow


# --- Pytest markers ---

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure Python tests, no ComfyUI or API needed")
    config.addinivalue_line("markers", "integration: requires running ComfyUI on port 8465")
    config.addinivalue_line("markers", "e2e: full end-to-end pipeline, very slow")
    config.addinivalue_line("markers", "requires_api_key: needs GEMINI_API_KEY env var")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on conditions."""
    skip_integration = pytest.mark.skip(reason="ComfyUI not running (set COMFYUI_URL to enable)")
    skip_e2e = pytest.mark.skip(reason="E2E tests need RUN_E2E=1 env var")
    skip_api = pytest.mark.skip(reason="Needs GEMINI_API_KEY env var")

    for item in items:
        if "e2e" in item.keywords and not os.environ.get("RUN_E2E"):
            item.add_marker(skip_e2e)
        if "integration" in item.keywords:
            try:
                r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=2)
                if r.status_code != 200:
                    raise ConnectionError
            except Exception:
                item.add_marker(skip_integration)
        if "requires_api_key" in item.keywords:
            has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
            has_llama = False
            try:
                r = requests.get("http://127.0.0.1:8001/health", timeout=2)
                has_llama = r.status_code == 200
            except Exception:
                pass
            if not has_gemini and not has_llama:
                item.add_marker(skip_api)


# --- Session-scoped fixtures ---

@pytest.fixture(scope="session")
def comfyui():
    """Verify ComfyUI is running and return the URL."""
    resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=10)
    assert resp.status_code == 200
    return COMFYUI_URL


@pytest.fixture(scope="session")
def gemini_api_key():
    """Return the Gemini API key from environment (if available)."""
    return os.environ.get("GEMINI_API_KEY", "")


@pytest.fixture(scope="session")
def llm_config(gemini_api_key):
    """Return (base_url, model, api_key) for LLM calls.

    Prefers local llama.cpp (no rate limits), falls back to Gemini API.
    """
    try:
        r = requests.get("http://127.0.0.1:8001/health", timeout=2)
        if r.status_code == 200:
            models = requests.get("http://127.0.0.1:8001/v1/models", timeout=2).json()
            model_id = models["data"][0]["id"]
            print(f"Using local LLM: {model_id}")
            return ("http://127.0.0.1:8001/v1", model_id, "sk-no-key-required")
    except Exception:
        pass

    if gemini_api_key:
        print("Using Gemini API (local LLM not available)")
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "models/gemini-2.5-flash",
            gemini_api_key,
        )

    pytest.skip("No LLM available (need llama.cpp on :8001 or GEMINI_API_KEY)")


@pytest.fixture(scope="session")
def character_image(comfyui):
    """Generate a character image using Z-Image Turbo. Shared across tests."""
    from PIL import Image

    workflow = {
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "6": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 1024, "batch_size": 1}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "Full body portrait, wide shot from head to toe. 90s OVA anime cel style. A young woman with long blue hair, red eyes, wearing a white school uniform with blue trim. Standing casually with arms at sides, slight smile. Clean white background. Detailed linework, vibrant colors.", "clip": ["6", 0]}},
        "11": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0, "model": ["3", 0], "positive": ["10", 0], "negative": ["10", 0], "latent_image": ["5", 0]}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["12", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/character", "images": ["13", 0]}},
    }
    result = submit_workflow(workflow, timeout=300)
    images = get_output_images(result["outputs"], "14")
    assert len(images) > 0, "No character image generated"
    assert len(images[0]) > 10000, "Character image too small"

    # Upload to PoseDirector subfolder for downstream workflows
    upload_image(images[0], "character.png", "PoseDirector")

    img = Image.open(io.BytesIO(images[0]))
    print(f"Character image: {len(images[0])} bytes, {img.size}")
    return images[0]


@pytest.fixture(scope="session")
def skeleton_keypoints(comfyui, character_image):
    """Run OpenposePreprocessor on the character image and return raw POSE_KEYPOINT data.

    Note: PoseExtractor outputs STRING/INT which ComfyUI doesn't include in /history.
    We use the OpenposePreprocessor's openpose_json (from UI output) and also
    save a rendered skeleton to verify the pipeline works.
    """
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/character.png", "upload": "image"}},
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": "enable", "detect_body": "enable", "detect_face": "disable",
            "resolution": 1024,
        }},
        "3": {"class_type": "PoseExtractor", "inputs": {"pose_keypoint": ["2", 1], "person_index": 0}},
        # Render extracted keypoints to verify PoseExtractor produced valid output
        "4": {"class_type": "PoseRenderer", "inputs": {
            "keypoints": ["3", 0],
            "width": 1024,
            "height": 1024,
            "line_width": 4,
            "point_radius": 6,
        }},
        "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/extractor_skeleton", "images": ["4", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/openpose_skeleton", "images": ["2", 0]}},
    }
    result = submit_workflow(workflow, timeout=120)

    # Verify the rendered skeleton was produced (proves PoseExtractor worked)
    rendered = get_output_images(result["outputs"], "5")
    assert len(rendered) > 0, "PoseExtractor -> PoseRenderer produced no output"

    # Extract keypoints from OpenposePreprocessor's UI output
    # The openpose_json comes from the 'ui' field in the history, not 'outputs'
    openpose_out = result["outputs"].get("2", {})
    # Try to get the openpose_json from UI outputs
    openpose_json_list = openpose_out.get("openpose_json", [])
    if openpose_json_list:
        pose_data = json.loads(openpose_json_list[0])
        if isinstance(pose_data, list):
            pose_data = pose_data[0]
        people = pose_data.get("people", [])
        if people:
            kpts = people[0]["pose_keypoints_2d"]
            keypoints_json = json.dumps(kpts)
            print(f"Skeleton keypoints: {len(kpts) // 3} points extracted from openpose_json")
            return keypoints_json

    # Fallback: can't extract from history, return empty marker
    # (Tests that depend on this fixture should handle this case)
    print("Warning: Could not extract keypoints from openpose_json, using empty")
    return "[]"


@pytest.fixture(scope="session")
def motion_keyframes(comfyui, character_image, skeleton_keypoints, llm_config):
    """Generate motion keyframes using MotionDirector. Returns the workflow result.

    Note: MotionDirector outputs (IMAGE, STRING, POSE_KEYPOINT). Only IMAGE is
    tracked in ComfyUI history. The STRING output won't appear in history outputs.
    We verify success via the SaveImage node that captures the IMAGE batch.
    """
    llm_url, llm_model, llm_key = llm_config
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "PoseDirector/character.png", "upload": "image"}},
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": "enable", "detect_body": "enable", "detect_face": "disable",
            "resolution": 1024,
        }},
        "3": {"class_type": "MotionDirector", "inputs": {
            "keypoints": "[]",
            "motion_description": "Character waves their right hand in a friendly greeting",
            "num_keyframes": 4,
            "total_frames": 32,
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
            "width": 1024,
            "height": 1024,
            "pose_keypoint": ["2", 1],
        }},
        "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/motion_keyframes", "images": ["3", 0]}},
    }
    result = submit_workflow(workflow, timeout=180, retries=2)

    # Verify images were produced
    images = get_output_images(result["outputs"], "4")
    assert len(images) > 0, "No keyframe images generated by MotionDirector"

    print(f"Motion keyframes: {len(images)} keyframe image(s) generated")
    return result
