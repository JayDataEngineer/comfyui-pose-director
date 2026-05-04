"""API-format workflow patching — direct dict manipulation by node ID."""

from __future__ import annotations

import json
from pathlib import Path


def load_api_workflow(path: Path) -> dict:
    """Load an API-format workflow JSON."""
    with open(path) as f:
        return json.load(f)


def patch_api_workflow(workflow: dict, patches: dict[str, dict]) -> dict:
    """Patch API-format workflow by node ID.

    patches: {"node_id": {"input_name": value, ...}, ...}
    """
    for node_id, node_patches in patches.items():
        node_id = str(node_id)
        if node_id in workflow:
            if "inputs" not in workflow[node_id]:
                workflow[node_id]["inputs"] = {}
            workflow[node_id]["inputs"].update(node_patches)
    return workflow


def normalize_paths(workflow: dict) -> dict:
    """Fix Windows backslash paths to forward slashes (Linux ComfyUI).

    VNCCS workflows are authored on Windows and contain paths like
    'qwen\\model.safetensors'. On Linux this needs to be 'qwen/model.safetensors'.
    """
    for node in workflow.values():
        for key, value in node.get("inputs", {}).items():
            if isinstance(value, str) and "\\" in value:
                node["inputs"][key] = value.replace("\\", "/")
    return workflow


def fix_model_paths(workflow: dict, object_info: dict) -> dict:
    """Fix model filename paths that have wrong subfolders.

    VNCCS author's ComfyUI had files in subfolders (e.g. 'Illustrious/ILFlatMix.safetensors')
    but on our system they're flat ('ILFlatMix.safetensors'). For each model-type input,
    if the filename isn't in the COMBO list, try matching just the basename.
    """
    MODEL_INPUTS = {
        "ckpt_name", "lora_name", "vae_name", "clip_name", "unet_name",
        "control_net_name", "model_name",
    }

    for nid, node in workflow.items():
        ct = node.get("class_type", "")
        schema = object_info.get(ct)
        if not schema:
            continue

        inputs = node.get("inputs", {})
        for key in MODEL_INPUTS:
            value = inputs.get(key)
            if not isinstance(value, str):
                continue

            # Find the COMBO options for this input
            config = None
            for category in ("required", "optional"):
                cfg = schema.get("input", {}).get(category, {}).get(key)
                if cfg:
                    config = cfg
                    break

            if not config or not isinstance(config[0], list):
                continue

            options = config[0]
            if value in options:
                continue  # Already valid

            # Try basename match: 'Illustrious/ILFlatMix.safetensors' -> 'ILFlatMix.safetensors'
            basename = value.rsplit("/", 1)[-1]
            if basename in options:
                inputs[key] = basename
                continue

            # Try matching just the filename in any option's path
            for opt in options:
                if opt.endswith("/" + basename) or opt == basename:
                    inputs[key] = opt
                    break

    return workflow


def fix_inputs_from_schema(workflow: dict, object_info: dict) -> dict:
    """Reset misaligned inputs to schema defaults, preserving links and patches.

    For nodes with input validation errors (wrong type, out of range, not in list),
    resets the input to the schema default. Link references [node_id, output_idx]
    are always preserved — only literal values are corrected.
    """
    # Input name aliases: old_class_type -> {old_input -> new_input}
    # Maps input names for nodes whose class_type will be aliased by the sanitizer.
    NODE_INPUT_ALIASES: dict[str, dict[str, str]] = {
        "Text Find and Replace": {"text": "texts"},
        "ReplaceText": {"text": "texts"},
    }

    # Input names that are always runtime-patched and should never be reset
    RUNTIME_INPUTS = {
        "image", "upload", "video", "audio",
        "ckpt_name", "lora_name", "vae_name", "clip_name", "unet_name",
        "existing_character",  # COMBO populated at runtime, may not include our name
    }

    for nid, node in list(workflow.items()):
        ct = node.get("class_type", "")

        # Remap input names BEFORE schema lookup (class_type may not exist yet)
        input_aliases = NODE_INPUT_ALIASES.get(ct, {})
        inputs = node.get("inputs", {})
        for old_name, new_name in input_aliases.items():
            if old_name in inputs and new_name not in inputs:
                inputs[new_name] = inputs.pop(old_name)

        schema = object_info.get(ct)
        if not schema:
            continue

        # Build expected input configs: name -> (type, config_dict_with_default)
        input_configs = {}
        for category in ("required", "optional"):
            for name, config in schema.get("input", {}).get(category, {}).items():
                input_configs[name] = config

        to_fix = {}

        for key, value in list(inputs.items()):
            # Skip link references — always valid
            if isinstance(value, list) and len(value) == 2:
                continue
            # Skip runtime-patched inputs (filenames that are always overridden)
            if key in RUNTIME_INPUTS:
                continue

            config = input_configs.get(key)
            if not config:
                continue

            type_spec = config[0]
            opts = config[1] if len(config) > 1 else {}

            # Check for value errors
            needs_fix = False

            # Non-primitive types (IMAGE, CLIP, VAE, MODEL, etc.) — must be links, not literals
            is_primitive_type = type_spec in (
                "FLOAT", "INT", "BOOLEAN", "STRING",
            ) or isinstance(type_spec, list)  # COMBO

            if not is_primitive_type:
                # This input expects a connection, not a literal value.
                # If it got a literal (string, int, float), it's misaligned — reset to default.
                needs_fix = True
            elif isinstance(type_spec, list):
                # COMBO: value must be in the list
                if value not in type_spec:
                    needs_fix = True
            elif type_spec == "FLOAT":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    needs_fix = True
                elif "max" in opts and float(value) > opts["max"]:
                    needs_fix = True
                elif "min" in opts and float(value) < opts["min"]:
                    needs_fix = True
            elif type_spec == "INT":
                if not isinstance(value, int) or isinstance(value, bool):
                    needs_fix = True
                elif "max" in opts and value > opts["max"]:
                    needs_fix = True
                elif "min" in opts and value < opts["min"]:
                    needs_fix = True
            elif type_spec == "BOOLEAN":
                if not isinstance(value, bool):
                    needs_fix = True
            elif type_spec == "STRING":
                if not isinstance(value, str):
                    needs_fix = True

            if needs_fix:
                default = opts.get("default")
                if default is not None:
                    to_fix[key] = default
                else:
                    to_fix[key] = None  # Will be removed

        for key, default in to_fix.items():
            if default is not None:
                inputs[key] = default
            else:
                inputs.pop(key, None)

    return workflow
