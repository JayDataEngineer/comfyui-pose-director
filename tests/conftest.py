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

# Load .env file into environment (if present)
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

# Make the project root importable as a Python package.
# The codebase uses relative imports (from ..utils) which require the root
# to be a proper parent package. We register it as "pose_director_root" to
# avoid issues with the hyphenated directory name.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
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

# Re-export shared ComfyUI API client functions so existing imports keep working
from pipeline.comfyui_client import (  # noqa: E402
    COMFYUI_URL,
    POLL_INTERVAL,
    MAX_WAIT,
    restart_comfyui,
    stop_llama,
    start_llama,
    submit_workflow,
    get_output_images,
    get_output_video,
    upload_image,
)


# Re-export the new UI-to-API converter
from pipeline.ui_to_api import (  # noqa: E402
    load_ui_workflow,
    patch_ui_workflow,
    patch_connection,
    convert_ui_to_api,
)

WORKFLOWS_DIR = PROJECT_ROOT / "workflows"


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
            has_or = bool(os.environ.get("OPENROUTER_API_KEY"))
            has_zai = bool(os.environ.get("ZAI_API_KEY"))
            has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
            has_llama = False
            try:
                r = requests.get("http://127.0.0.1:8001/health", timeout=2)
                has_llama = r.status_code == 200
            except Exception:
                pass
            if not has_or and not has_zai and not has_gemini and not has_llama:
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

    Priority: OpenRouter (cheap, fast, no VRAM) → ZAI → llama.cpp → Gemini.
    """
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if or_key:
        print("Using OpenRouter (deepseek-v4-pro)")
        return (
            "https://openrouter.ai/api/v1",
            "deepseek/deepseek-v4-pro",
            or_key,
        )

    zai_key = os.environ.get("ZAI_API_KEY", "")
    if zai_key:
        print("Using ZAI cloud API (glm-4.5-air)")
        return (
            "https://api.z.ai/api/coding/paas/v4",
            "glm-4.5-air",
            zai_key,
        )

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

    pytest.skip("No LLM available (need ZAI_API_KEY, llama.cpp on :8001, or GEMINI_API_KEY)")


@pytest.fixture(scope="session")
def character_image(comfyui):
    """Generate a character image using Z-Image Turbo. Shared across tests."""
    from PIL import Image

    workflow = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
    # Fix missing VAEDecode→SaveImage connection if needed
    save_node = next((n for n in workflow["nodes"] if n["type"] == "SaveImage"), None)
    vaedecode_node = next((n for n in workflow["nodes"] if n["type"] == "VAEDecode"), None)
    if save_node and vaedecode_node and save_node.get("inputs", [{}])[0].get("link") is None:
        patch_connection(
            workflow,
            target_node_id=save_node["id"],
            target_input_name="images",
            source_node_id=vaedecode_node["id"],
            source_output_index=0,
        )
    patch_ui_workflow(workflow, {"SaveImage": {"filename_prefix": "pytest/character"}})
    api = convert_ui_to_api(workflow)
    result = submit_workflow(api, timeout=300)
    images = get_output_images(result["outputs"], str(save_node["id"]))
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
        "4": {"class_type": "PoseRenderer", "inputs": {
            "keypoints": ["3", 0],
            "width": 1024, "height": 1024, "line_width": 4, "point_radius": 6,
        }},
        "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/extractor_skeleton", "images": ["4", 0]}},
        "6": {"class_type": "PreviewImage", "inputs": {"filename_prefix": "pytest/openpose_skeleton", "images": ["2", 0]}},
    }
    result = submit_workflow(workflow, timeout=120)

    # Verify the rendered skeleton was produced (proves PoseExtractor worked)
    rendered = get_output_images(result["outputs"], "5")
    assert len(rendered) > 0, "PoseExtractor -> PoseRenderer produced no output"

    # Extract keypoints from OpenposePreprocessor's UI output
    openpose_out = result["outputs"].get("2", {})
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
    print("Warning: Could not extract keypoints from openpose_json, using empty")
    return "[]"


@pytest.fixture(scope="session")
def motion_keyframes(comfyui, llm_config):
    """Generate motion keyframes using MotionDirector (rotation-only). Returns the workflow result."""
    llm_url, llm_model, llm_key = llm_config
    workflow = load_ui_workflow(WORKFLOWS_DIR / "01_motion_keyframes_rotation.json")
    patch_ui_workflow(workflow, {
        "MotionDirector": {
            "motion_description": "Character waves their right hand in a friendly greeting",
            "num_keyframes": 4,
            "total_frames": 32,
            "base_url": llm_url,
            "model": llm_model,
        },
        "PreviewImage": {"filename_prefix": "pytest/motion_keyframes"},
    })
    api = convert_ui_to_api(workflow)
    result = submit_workflow(api, timeout=180, retries=2)

    # Verify images were produced (PreviewImage node id=4)
    images = get_output_images(result["outputs"], "4")
    assert len(images) > 0, "No keyframe images generated by MotionDirector"

    print(f"Motion keyframes: {len(images)} keyframe image(s) generated")
    return result
