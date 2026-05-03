#!/usr/bin/env python3
"""Integration test: full sprite pipeline on remote Ray ComfyUI.

Tests the complete BodyMeshRenderer → OpenposePreprocessor → VNCCS_QWEN_Encoder
→ KSampler → SaveImage pipeline through the Ray Serve proxy.

Prerequisites:
  - Ray cluster running with ComfyUI deployed
  - VNCCS, comfyui_controlnet_aux, ComfyUI-GGUF installed
  - Model files in place (GGUF, CLIP, VAE, LoRA)
  - COMFYUI_URL set to Ray proxy (default: http://100.86.69.57:18800/comfyui)

Usage:
    COMFYUI_URL=http://100.86.69.57:18800/comfyui uv run pytest tests/integration/test_remote_pipeline.py -v -s
"""

import io
import os
import random
import sys
import time

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.ui_to_api import load_ui_workflow, patch_ui_workflow, convert_ui_to_api

COMFYUI = os.environ.get("COMFYUI_URL", "http://100.86.69.57:18800/comfyui")
WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "workflows", "05_sprite_bodymesh_edit.json")
TIMEOUT = 300  # 5 minutes max


@pytest.fixture(scope="module")
def comfyui_available():
    """Check if remote ComfyUI is reachable."""
    try:
        resp = requests.get(f"{COMFYUI}/system_stats", timeout=15)
        return resp.status_code == 200
    except Exception:
        pytest.skip("Remote ComfyUI not reachable")
        return False


@pytest.fixture(scope="module")
def uploaded_character():
    """Upload a test character image with real pixel content.

    ComfyUI validates images — solid-color test patterns are rejected.
    We create a noise+gradient image that passes validation.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.RandomState(42)
    W, H = 768, 1024
    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    xv, yv = np.meshgrid(xs, ys)
    noise = rng.randint(-15, 16, (H, W, 3), dtype=np.int16)
    r = (100 + 50 * xv + noise[:, :, 0]).clip(0, 255).astype(np.uint8)
    g = (50 + 80 * yv + noise[:, :, 1]).clip(0, 255).astype(np.uint8)
    b = (150 + 40 * (xv + yv) / 2 + noise[:, :, 2]).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(np.stack([r, g, b], axis=2))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resp = requests.post(
        f"{COMFYUI}/upload/image",
        files={"image": ("test_character.png", buf.getvalue(), "image/png")},
        data={"overwrite": "true", "subfolder": "PoseDirector"},
        timeout=15,
    )
    assert resp.status_code == 200
    return resp.json()["name"]


def _submit_and_wait(api: dict, timeout: int = TIMEOUT) -> dict:
    """Submit a workflow and wait for completion."""
    resp = requests.post(f"{COMFYUI}/prompt", json={"prompt": api}, timeout=60)
    assert resp.status_code == 200, f"Submit failed: {resp.text[:300]}"
    prompt_id = resp.json()["prompt_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(2)
        hist = requests.get(f"{COMFYUI}/history/{prompt_id}", timeout=10).json()
        if prompt_id in hist:
            status = hist[prompt_id].get("status", {})
            if status.get("completed") or status.get("status_str") == "error":
                return hist[prompt_id]
    pytest.fail(f"Workflow timed out after {timeout}s")


class TestRemoteSpritePipeline:
    """Full sprite pipeline tests against remote Ray ComfyUI."""

    def test_nodes_available(self, comfyui_available):
        """Verify all required nodes are registered."""
        resp = requests.get(f"{COMFYUI}/object_info", timeout=15)
        assert resp.status_code == 200
        nodes = resp.json()
        required = [
            "BodyMeshRenderer", "OpenposePreprocessor", "VNCCS_QWEN_Encoder",
            "UnetLoaderGGUF", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly",
            "KSampler", "VAEDecode", "SaveImage", "LoadImage",
        ]
        missing = [n for n in required if n not in nodes]
        assert not missing, f"Missing nodes: {missing}"

    def test_models_accessible(self, comfyui_available):
        """Verify model files are accessible through the loaders."""
        resp = requests.get(f"{COMFYUI}/object_info", timeout=15)
        nodes = resp.json()

        # Check UNET/GGUF
        unet_info = nodes.get("UnetLoaderGGUF", {}).get("input", {})
        unet_names = unet_info.get("required", {}).get("unet_name", [[], {}])[0]
        gguf_files = [f for f in unet_names if "Qwen-Image-Edit" in f]
        assert gguf_files, "No Qwen-Image-Edit GGUF in unet loader"

        # Check CLIP
        clip_info = nodes.get("CLIPLoader", {}).get("input", {})
        clip_names = clip_info.get("required", {}).get("clip_name", [[], {}])[0]
        qwen_clip = [f for f in clip_names if "qwen_2.5_vl" in f]
        assert qwen_clip, "No qwen_2.5_vl CLIP in loader"

        # Check VAE
        vae_info = nodes.get("VAELoader", {}).get("input", {})
        vae_names = vae_info.get("required", {}).get("vae_name", [[], {}])[0]
        qwen_vae = [f for f in vae_names if "qwen_image_vae" in f]
        assert qwen_vae, "No qwen_image_vae in VAE loader"

    def test_full_sprite_pipeline(self, comfyui_available, uploaded_character):
        """Run the complete sprite generation pipeline."""
        wf = load_ui_workflow(WORKFLOW_PATH)
        patch_ui_workflow(wf, {
            "LoadImage": {"image": uploaded_character},
            "BodyMeshRenderer": {
                "rotations_json": '{"right_shoulder": [0, 0, 90], "left_shoulder": [0, 0, -90]}',
                "model_rotation_y": 0.0,
            },
            "KSampler": {"seed": 99, "steps": 4, "cfg": 1.0},
            "SaveImage": {"filename_prefix": "test/integration_sprite"},
        })
        api = convert_ui_to_api(wf)

        result = _submit_and_wait(api, timeout=TIMEOUT)
        status = result.get("status", {})
        if status.get("status_str") == "error":
            # Extract actual error messages from node outputs
            msgs = []
            for nid, out in result.get("outputs", {}).items():
                if "error" in out:
                    msgs.append(f"Node {nid}: {out['error']}")
            pytest.fail(f"Workflow error: {status}. Node errors: {msgs or 'none in outputs'}")

        outputs = result.get("outputs", {})
        images = []
        for nid, out in outputs.items():
            images.extend(out.get("images", []))
        assert len(images) >= 1, "No output images"

        # Verify image is downloadable
        img_info = images[0]
        resp = requests.get(f"{COMFYUI}/view", params={
            "filename": img_info["filename"],
            "subfolder": img_info.get("subfolder", ""),
            "type": img_info.get("type", "output"),
        }, timeout=15)
        assert resp.status_code == 200
        assert len(resp.content) > 1000, "Image too small, likely corrupt"
        assert resp.content[:4] == b"\x89PNG", "Not a valid PNG"

    def test_different_pose(self, comfyui_available, uploaded_character):
        """Test a different pose to verify pose control works."""
        wf = load_ui_workflow(WORKFLOW_PATH)
        patch_ui_workflow(wf, {
            "LoadImage": {"image": uploaded_character},
            "BodyMeshRenderer": {
                "rotations_json": '{"spine": [10, 0, 0], "right_shoulder": [0, 0, -45], "left_shoulder": [0, 0, 45], "right_knee": [20, 0, 0]}',
                "model_rotation_y": 90.0,  # Side view
            },
            "KSampler": {"seed": 100, "steps": 4, "cfg": 1.0},
            "SaveImage": {"filename_prefix": "test/integration_side"},
        })
        api = convert_ui_to_api(wf)

        result = _submit_and_wait(api, timeout=TIMEOUT)
        status = result.get("status", {})
        if status.get("status_str") == "error":
            msgs = []
            for nid, out in result.get("outputs", {}).items():
                if "error" in out:
                    msgs.append(f"Node {nid}: {out['error']}")
            pytest.fail(f"Workflow error: {status}. Node errors: {msgs or 'none in outputs'}")

        outputs = result.get("outputs", {})
        images = []
        for nid, out in outputs.items():
            images.extend(out.get("images", []))
        assert len(images) >= 1, "No output images"
