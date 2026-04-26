#!/usr/bin/env python3
"""Validate ComfyUI UI-format workflow files against a live ComfyUI instance.

Checks:
  1. All node types exist in ComfyUI
  2. All links reference valid nodes (source + target)
  3. Target inputs claim their links (bidirectional integrity)
  4. Required connection-type inputs have links (warns, not errors)
  5. Model files referenced in widgets_values exist on disk

Usage:
    python scripts/validate_workflows.py [workflow_files...]
    python scripts/validate_workflows.py              # checks all workflows/
"""

import json
import os
import sys
from pathlib import Path

import requests

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://localhost:8000/comfyui")
MODEL_BASE = os.environ.get(
    "MODEL_BASE",
    "/home/ubuntu/Documents/programs/shared-docker-infra/models/image-gen/comfyui",
)
WORKFLOWS_DIR = Path(__file__).parent.parent / "workflows"

# Types that are widget-only (dropdowns, text fields, sliders) — never link connections
WIDGET_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN"}


def is_widget_type(t):
    """Returns True if this input type is a widget (not a connection slot)."""
    if isinstance(t, list):
        return True  # COMBO options list like ["euler", "dpmpp_2m", ...]
    return t in WIDGET_TYPES


def get_object_info():
    resp = requests.get(f"{COMFYUI_URL}/object_info", timeout=10)
    resp.raise_for_status()
    return resp.json()


def check_model_exists(filename, model_type_dirs=None):
    """Check if a model file exists in any of the standard model subdirectories."""
    if not filename:
        return True, ""
    search_dirs = model_type_dirs or [
        "checkpoints",
        "clip",
        "text_encoders",
        "vae",
        "unet",
        "diffusion_models",
        "controlnet",
        "loras",
        "upscale_models",
        "embeddings",
    ]
    for subdir in search_dirs:
        path = os.path.join(MODEL_BASE, subdir, filename)
        if os.path.exists(path):
            return True, f"{subdir}/{filename}"
    return False, filename


def validate_workflow(path, object_info):
    """Validate a single workflow file. Returns (errors, warnings)."""
    with open(path) as f:
        wf = json.load(f)

    errors = []
    warnings = []

    nodes = wf.get("nodes", [])
    links = wf.get("links", [])

    node_by_id = {n["id"]: n for n in nodes}
    link_by_id = {l[0]: l for l in links}

    fname = os.path.basename(path)

    # --- Check 1: Node types registered ---
    for node in nodes:
        if node["type"] not in object_info:
            errors.append(f"Node {node['id']} type \"{node['type']}\" NOT registered in ComfyUI")

    # --- Check 2: Link integrity (bidirectional) ---
    for link in links:
        link_id, from_node, from_slot, to_node, to_slot, link_type = link

        # Source node exists?
        if from_node not in node_by_id:
            errors.append(f"Link {link_id}: source node {from_node} not found")
            continue

        # Target node exists?
        if to_node not in node_by_id:
            errors.append(f"Link {link_id}: target node {to_node} not found")
            continue

        src = node_by_id[from_node]
        dst = node_by_id[to_node]

        # Source output slot in range?
        src_outputs = src.get("outputs", [])
        if from_slot >= len(src_outputs):
            errors.append(
                f"Link {link_id}: {src['type']}({from_node}) output slot {from_slot} "
                f"out of range ({len(src_outputs)} outputs)"
            )

        # Target input claims this link?
        dst_inputs = dst.get("inputs", [])
        found = False
        for inp in dst_inputs:
            if inp.get("link") == link_id:
                found = True
                break
        if not found:
            errors.append(
                f"Link {link_id}: {dst['type']}({to_node}) slot {to_slot} "
                f"has no input claiming this link"
            )

    # --- Check 3: Node input references valid link ---
    for node in nodes:
        for inp in node.get("inputs", []):
            lid = inp.get("link")
            if lid is not None and lid not in link_by_id:
                errors.append(
                    f"{node['type']}({node['id']}): input \"{inp['name']}\" "
                    f"references missing link {lid}"
                )

    # --- Check 4: Required connection inputs have links ---
    for node in nodes:
        ntype = node["type"]
        if ntype not in object_info:
            continue

        node_info = object_info[ntype]
        required = node_info.get("input", {}).get("required", {})

        for inp_name, inp_def in required.items():
            inp_type = inp_def[0] if isinstance(inp_def, (list, tuple)) else inp_def
            if is_widget_type(inp_type):
                continue
            # This is a connection-type input (IMAGE, MODEL, CLIP, etc.)
            has_link = any(
                inp.get("link") is not None
                for inp in node.get("inputs", [])
                if inp["name"] == inp_name
            )
            if not has_link:
                warnings.append(
                    f"{node['type']}({node['id']}): required input \"{inp_name}\" "
                    f"({inp_type}) has no link"
                )

    # --- Check 5: Model files exist ---
    for node in nodes:
        ntype = node["type"]
        wv = node.get("widgets_values", [])
        if not wv:
            continue

        # Check common model loader patterns
        model_checks = []
        if ntype == "UnetLoaderGGUF" and len(wv) > 0:
            model_checks.append(("unet", wv[0]))
        elif ntype == "CheckpointLoaderSimple" and len(wv) > 0:
            model_checks.append(("checkpoints", wv[0]))
        elif ntype == "CLIPLoader" and len(wv) > 0:
            model_checks.append(("clip", wv[0]))
        elif ntype == "DualCLIPLoaderGGUF" and len(wv) > 1:
            model_checks.append(("clip", wv[0]))
            model_checks.append(("text_encoders", wv[1]))
        elif ntype == "VAELoader" and len(wv) > 0:
            model_checks.append(("vae", wv[0]))

        for subdir, filename in model_checks:
            found, location = check_model_exists(filename, [subdir])
            if not found:
                # Try broader search
                found, location = check_model_exists(filename)
            if not found:
                warnings.append(
                    f"{node['type']}({node['id']}): model file \"{filename}\" not found on disk"
                )

    return errors, warnings


def main():
    # Determine which files to check
    if len(sys.argv) > 1:
        files = [Path(f) for f in sys.argv[1:]]
    else:
        files = sorted(
            f for f in WORKFLOWS_DIR.glob("*.json") if "tests" not in str(f)
        )

    print(f"Connecting to ComfyUI at {COMFYUI_URL}...")
    try:
        object_info = get_object_info()
    except Exception as e:
        print(f"ERROR: Cannot reach ComfyUI: {e}")
        sys.exit(1)

    print(f"Loaded {len(object_info)} node types from ComfyUI\n")

    total_errors = 0
    total_warnings = 0
    results = []

    for fpath in files:
        errors, warnings = validate_workflow(str(fpath), object_info)
        results.append((fpath, errors, warnings))
        total_errors += len(errors)
        total_warnings += len(warnings)

    # Print report
    for fpath, errors, warnings in results:
        name = os.path.basename(str(fpath))
        status = "PASS" if not errors else "FAIL"
        print(f"{'='*60}")
        print(f"{name}: {status} ({len(errors)} errors, {len(warnings)} warnings)")
        print(f"{'='*60}")
        for e in errors:
            print(f"  ERROR: {e}")
        for w in warnings:
            print(f"  WARN:  {w}")
        if not errors and not warnings:
            print("  (clean)")
        print()

    print(f"Summary: {len(files)} workflows, {total_errors} errors, {total_warnings} warnings")
    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
