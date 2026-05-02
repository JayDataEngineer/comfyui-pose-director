#!/usr/bin/env python3
"""Test the Ray ComfyUI proxy handles all API endpoints used by the pipeline.

Prerequisites:
  1. ComfyUI running on port 8465
  2. Ray cluster started: cd ~/Documents/programs/ray && bash scripts/start_cluster.sh
  3. ComfyUI proxy deployed: cd ~/Documents/programs/ray && .venv/bin/python -m services.image.comfyui_proxy_test &

Usage:
    python scripts/test_ray_proxy.py
"""

import io
import json
import os
import sys
import time

sys.path.insert(0, "/home/ubuntu/Documents/programs/img/comfyui-pose-director")

COMFYUI_DIRECT = os.environ.get("COMFYUI_DIRECT_URL", "http://127.0.0.1:8465")
COMFYUI_RAY = os.environ.get("COMFYUI_URL", "http://localhost:8000/comfyui")


def test_proxy():
    import requests

    tests_passed = 0
    tests_failed = 0

    def check(name, ray_resp, direct_resp=None):
        nonlocal tests_passed, tests_failed
        ok = ray_resp.status_code == 200
        if direct_resp and ok:
            ok = ray_resp.content == direct_resp.content
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name} (status={ray_resp.status_code}, size={len(ray_resp.content)})")
        if ok:
            tests_passed += 1
        else:
            tests_failed += 1
            if direct_resp and ray_resp.content != direct_resp.content:
                print(f"    Content mismatch: ray={len(ray_resp.content)} vs direct={len(direct_resp.content)}")
            if ray_resp.status_code != 200:
                print(f"    Error: {ray_resp.text[:200]}")
        return ok

    print("=" * 60)
    print("Testing ComfyUI API endpoints through Ray proxy")
    print("=" * 60)

    # Check if direct ComfyUI is reachable for comparison
    direct_available = False
    try:
        requests.get(f"{COMFYUI_DIRECT}/system_stats", timeout=3)
        direct_available = True
    except Exception:
        print(f"  (Direct ComfyUI at {COMFYUI_DIRECT} not reachable, skipping comparisons)")

    # Test 1: system_stats (dynamic RAM values, only check status)
    print("\n1. GET /system_stats")
    ray_resp = requests.get(f"{COMFYUI_RAY}/system_stats", timeout=10)
    check("system_stats (status only)", ray_resp)
    # Verify structure matches
    if ray_resp.status_code == 200:
        ray_data = ray_resp.json()
        keys = list(ray_data.get("system", {}).keys())
        print(f"    OK: {len(keys)} system keys ({', '.join(keys[:5])}...)")

    # Test 2: object_info
    print("\n2. GET /object_info")
    ray_resp = requests.get(f"{COMFYUI_RAY}/object_info", timeout=10)
    if check("object_info", ray_resp):
        ray_info = ray_resp.json()
        print(f"    OK: {len(ray_info)} node types available")

    # Test 3: upload/image
    print("\n3. POST /upload/image")
    from PIL import Image
    img = Image.new("RGB", (64, 64), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    files = {"image": ("test_proxy.png", img_bytes, "image/png")}
    data = {"overwrite": "true", "subfolder": "PoseDirector"}
    ray_resp = requests.post(f"{COMFYUI_RAY}/upload/image", files=files, data=data, timeout=10)
    check("upload_image", ray_resp)

    # Test 4: Submit workflow using only pose-director nodes (no VNCCS dependency)
    print("\n4. POST /prompt (pose-director workflow)")
    from pipeline.ui_to_api import load_ui_workflow, patch_ui_workflow, convert_ui_to_api

    wf = load_ui_workflow("/home/ubuntu/Documents/programs/img/comfyui-pose-director/workflows/05_sprite_bodymesh_edit.json")
    patch_ui_workflow(wf, {
        "LoadImage": {"image": "PoseDirector/character.png"},
        "BodyMeshRenderer": {
            "rotations_json": '{"right_shoulder": [0, 0, 90], "left_shoulder": [0, 0, -90]}',
            "model_rotation_y": 0.0,
        },
        "KSampler": {"seed": 42},
        "SaveImage": {"filename_prefix": "test/ray_proxy"},
    })
    api = convert_ui_to_api(wf)
    ray_resp = requests.post(
        f"{COMFYUI_RAY}/prompt",
        json={"prompt": api},
        timeout=30,
    )
    if check("submit_workflow", ray_resp):
        result = ray_resp.json()
        prompt_id = result.get("prompt_id")
        print(f"    prompt_id: {prompt_id}")

        # Test 5: Wait for completion and get history
        if prompt_id:
            print("\n5. GET /history/{prompt_id}")
            for _ in range(60):
                time.sleep(1)
                hist_resp = requests.get(f"{COMFYUI_RAY}/history/{prompt_id}", timeout=10)
                if hist_resp.status_code == 200:
                    hist = hist_resp.json()
                    if prompt_id in hist:
                        status = hist[prompt_id].get("status", {})
                        if status.get("completed", False) or status.get("status_str") == "error":
                            check("history_complete", hist_resp)
                            break
            else:
                print("  [FAIL] Workflow didn't complete in 60s")
                tests_failed += 1

    # Test 6: GET /view (image retrieval)
    print("\n6. GET /view (image download)")
    # Find an output image from the history
    try:
        hist_resp = requests.get(f"{COMFYUI_RAY}/history", timeout=10)
        if hist_resp.status_code == 200:
            hist = hist_resp.json()
            # Find the most recent prompt with output images
            for pid, entry in reversed(list(hist.items())):
                for node_id, outputs in entry.get("outputs", {}).items():
                    for img_info in outputs.get("images", []):
                        filename = img_info["filename"]
                        subfolder = img_info.get("subfolder", "")
                        view_resp = requests.get(
                            f"{COMFYUI_RAY}/view",
                            params={"filename": filename, "subfolder": subfolder, "type": "output"},
                            timeout=10,
                        )
                        check(f"view_image ({filename})", view_resp)
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                print("  [SKIP] No output images found in history")
    except Exception as e:
        print(f"  [FAIL] Error checking history: {e}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    return tests_failed == 0


if __name__ == "__main__":
    success = test_proxy()
    sys.exit(0 if success else 1)
