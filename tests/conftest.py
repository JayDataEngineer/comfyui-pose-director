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
            has_zai = bool(os.environ.get("ZAI_API_KEY"))
            has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
            has_llama = False
            try:
                r = requests.get("http://127.0.0.1:8001/health", timeout=2)
                has_llama = r.status_code == 200
            except Exception:
                pass
            if not has_zai and not has_gemini and not has_llama:
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

    Prefers ZAI cloud API (no VRAM), falls back to local llama.cpp, then Gemini.
    """
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
    from pipeline.workflows import build_character_workflow

    workflow = build_character_workflow(filename_prefix="pytest/character")
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
    from pipeline.workflows import build_pose_extraction_workflow

    workflow = build_pose_extraction_workflow(
        image_path="PoseDirector/character.png",
        resolution=1024,
        include_renderer=True,
        filename_prefix_skeleton="pytest/extractor_skeleton",
        filename_prefix_openpose="pytest/openpose_skeleton",
    )
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
def motion_keyframes(comfyui, character_image, skeleton_keypoints, llm_config):
    """Generate motion keyframes using MotionDirector. Returns the workflow result.

    Note: MotionDirector outputs (IMAGE, STRING, POSE_KEYPOINT). Only IMAGE is
    tracked in ComfyUI history. The STRING output won't appear in history outputs.
    We verify success via the SaveImage node that captures the IMAGE batch.
    """
    from pipeline.workflows import build_motion_director_workflow

    llm_url, llm_model, llm_key = llm_config
    workflow = build_motion_director_workflow(
        image_path="PoseDirector/character.png",
        motion_description="Character waves their right hand in a friendly greeting",
        num_keyframes=4,
        total_frames=32,
        llm_url=llm_url,
        llm_key=llm_key,
        llm_model=llm_model,
        width=1024,
        height=1024,
        filename_prefix="pytest/motion_keyframes",
    )
    result = submit_workflow(workflow, timeout=180, retries=2)

    # Verify images were produced
    images = get_output_images(result["outputs"], "4")
    assert len(images) > 0, "No keyframe images generated by MotionDirector"

    print(f"Motion keyframes: {len(images)} keyframe image(s) generated")
    return result
