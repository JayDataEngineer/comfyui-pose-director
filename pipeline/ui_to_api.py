"""Convert ComfyUI UI-format workflows to API format and patch them.

This module provides the bridge between human-friendly UI-format workflow JSONs
(loaded in ComfyUI's browser) and the API format required by the /prompt endpoint.

Usage:
    from pipeline.ui_to_api import load_ui_workflow, patch_ui_workflow, convert_ui_to_api

    workflow = load_ui_workflow("workflows/01_motion_keyframes_rotation.json")
    patch_ui_workflow(workflow, {"MotionDirector": {"motion_description": "wave"}})
    api_workflow = convert_ui_to_api(workflow)
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

WORKFLOWS_DIR = Path(__file__).parent.parent / "workflows"

# ---------------------------------------------------------------------------
# Node schemas: ordered list of ALL parameter names as they appear in the API.
# The order must match the order of widgets_values in the UI format.
# ---------------------------------------------------------------------------
NODE_SCHEMAS: dict[str, list[str]] = {
    # Built-in ComfyUI nodes
    "LoadImage": ["image", "upload"],
    "OpenposePreprocessor": [
        "image",
        "detect_hand",
        "detect_body",
        "detect_face",
        "resolution",
        "scale_stick_for_xinsr_cn",
    ],
    "KSampler": [
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "denoise",
    ],
    "VAEDecode": ["samples", "vae"],
    "SaveImage": ["images", "filename_prefix"],
    "PreviewImage": ["images", "filename_prefix"],
    "CLIPTextEncode": ["text", "clip"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "ModelSamplingAuraFlow": ["model", "shift"],
    "ConditioningZeroOut": ["conditioning"],
    "CheckpointLoaderSimple": ["ckpt_name"],
    "UnetLoaderGGUF": ["unet_name"],
    "LoraLoaderModelOnly": ["model", "lora_name", "strength_model"],
    "DualCLIPLoaderGGUF": ["clip_name1", "clip_name2", "type"],
    "EmptyLTXVLatentVideo": ["width", "height", "length", "batch_size"],
    "LTXVAddGuideAdvanced": [
        "positive",
        "negative",
        "vae",
        "latent",
        "image",
        "frame_idx",
        "strength",
        "crf",
        "blur_radius",
        "interpolation",
        "crop",
    ],
    "VHS_VideoCombine": [
        "images",
        "filename_prefix",
        "frame_rate",
        "loop_count",
        "format",
        "pingpong",
        "save_output",
    ],
    "ImpactImageBatchToImageList": ["image"],
    # LTX post-processing (EditAnything IC-LoRA)
    "LTXICLoRALoaderModelOnly": ["model", "lora_name", "strength_model"],
    "LTXVGemmaCLIPModelLoader": ["gemma_path", "ltxv_path", "max_length"],
    "LTXAddVideoICLoRAGuide": [
        "positive",
        "negative",
        "vae",
        "latent",
        "image",
        "frame_idx",
        "strength",
        "latent_downscale_factor",
        "crop",
        "use_tiled_encode",
        "tile_size",
        "tile_overlap",
    ],
    # Lip sync / Wan video
    "CLIPVisionLoader": ["clip_name"],
    "LoadAudio": ["audio"],
    "LoadAudioToText": [],
    "WanAudioEncoder": [],
    "WanImageToVideo": [
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "clip_vision_output",
        "start_image",
    ],
    # Custom Pose Director nodes
    "PoseManipulator": [
        "keypoints",
        "instruction",
        "base_url",
        "api_key",
        "model",
        "width",
        "height",
        "pose_keypoint",
        "base_pose",
    ],
    "MotionDirector": [
        "motion_description",
        "num_keyframes",
        "total_frames",
        "base_url",
        "model",
        "width",
        "height",
        "max_retries",
        "vision_feedback_rounds",
    ],
    "PoseRenderer": ["keypoints", "width", "height", "line_width", "point_radius"],
    "PoseExtractor": ["pose_keypoint", "person_index"],
    "KeyframeReviewer": ["images", "review_prompt", "pass_threshold", "base_url", "model"],
    "BodyMeshRenderer": ["rotations_json", "width", "height", "model_rotation_y", "backend"],
    "PromptEnhancer": ["prompt", "variant", "base_url", "model"],
    "AnimationExporter": [
        "rotations_json",
        "keyframes_json",
        "total_frames",
        "fps",
        "export_bvh",
        "export_video",
        "width",
        "height",
        "filename_prefix",
        "video_format",
        "backend",
    ],
    # HY-Motion (Tencent Hunyuan text-to-motion)
    "HYMotionLoadLLM": ["model_name", "quantization", "offload_to_cpu"],
    "HYMotionLoadLLMGGUF": ["gguf_file"],
    "HYMotionLoadNetwork": ["model_name"],
    "HYMotionEncodeText": ["llm", "text"],
    "HYMotionGenerate": ["network", "conditioning", "duration", "seed", "cfg_scale", "num_samples"],
    "HYMotionPreview": ["motion_data", "sample_index", "frame_step", "image_size"],
    "HYMotionSaveNPZ": ["motion_data", "output_dir", "filename_prefix"],
    "HYMotionExportFBX": ["motion_data", "output_dir", "filename_prefix", "custom_fbx_path", "yaw_offset"],
    "MotionMeshRenderer": ["motion_data", "sample_index", "frame_step", "num_keyframes", "width", "height"],
    "VNCCS_PoseStudio": ["pose_data"],
    "ImpactSwitch": ["select", "sel_mode"],
    "VNCCS_QWEN_Encoder": [
        "clip",
        "prompt",
        "vae",
        "latent_image_index",
        "image1",
        "image2",
        "image3",
        "image1_name",
        "image2_name",
        "image3_name",
        "target_size",
        "upscale_method",
        "crop_method",
        "weight1",
        "weight2",
        "weight3",
        "vl_size",
        "instruction",
        "qwen_2511",
    ],
    "VNCCS_RMBG2": [
        "image",
        "model",
        "sensitivity",
        "process_res",
        "mask_blur",
        "mask_offset",
        "invert_output",
        "refine_foreground",
        "background",
    ],
}

# Widget indices that are UI-only and should be skipped during API conversion.
# Format: node_type -> {widget_index: ignored_name}
UI_ONLY_WIDGETS: dict[str, dict[int, str]] = {
    "KSampler": {1: "control_after_generate"},
    "HYMotionGenerate": {2: "control_after_generate"},
}

# Default values for required inputs that ComfyUI's UI may omit from widgets_values.
# These are filled in when the widget value is missing during conversion.
NODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "KSampler": {"denoise": 1.0},
}


def _build_link_map(links: list[list]) -> dict[int, tuple[int, int]]:
    """Build lookup: link_id -> (source_node_id, source_output_index).

    UI link format: [link_id, src_node, src_slot, tgt_node, tgt_slot, type]
    """
    link_map: dict[int, tuple[int, int]] = {}
    for link in links:
        link_id, src_id, src_out, _, _, _ = link
        link_map[link_id] = (src_id, src_out)
    return link_map


def _convert_generic(node: dict, link_map: dict[int, tuple[int, int]]) -> dict:
    """Convert a single UI node to API format using the embedded schema."""
    node_type = node.get("type", "")
    schema = NODE_SCHEMAS.get(node_type, [])
    ui_only = UI_ONLY_WIDGETS.get(node_type, {})
    defaults = NODE_DEFAULTS.get(node_type, {})

    inputs: dict[str, Any] = {}
    widgets = list(node.get("widgets_values", []))
    ui_inputs = {inp["name"]: inp for inp in node.get("inputs", [])}

    widget_idx = 0

    for input_name in schema:
        # Skip UI-only widgets (e.g. KSampler control_after_generate)
        while widget_idx in ui_only:
            widget_idx += 1

        if input_name in ui_inputs:
            ui_inp = ui_inputs[input_name]
            if ui_inp.get("link") is not None:
                src_id, src_out = link_map[ui_inp["link"]]
                inputs[input_name] = [str(src_id), src_out]
            else:
                if widget_idx < len(widgets):
                    inputs[input_name] = widgets[widget_idx]
                    widget_idx += 1
                elif input_name in defaults:
                    inputs[input_name] = defaults[input_name]
                # else: omit — ComfyUI will use the node's built-in default
        else:
            # Hidden parameter (not listed in UI inputs array)
            if widget_idx < len(widgets):
                inputs[input_name] = widgets[widget_idx]
                widget_idx += 1
            elif input_name in defaults:
                inputs[input_name] = defaults[input_name]

    return {"class_type": node_type, "inputs": inputs}


def convert_ui_to_api(ui_workflow: dict, *, expand_subgraphs: bool = True) -> dict:
    """Convert a ComfyUI UI-format workflow to API format.

    Args:
        ui_workflow: Dict loaded from a ComfyUI UI-format JSON file.
        expand_subgraphs: If True, expand UUID subgraph nodes before conversion.

    Returns:
        Dict in ComfyUI API format suitable for the /prompt endpoint.
    """
    if expand_subgraphs:
        from pipeline.subgraph_expander import expand_subgraphs as _expand

        ui_workflow = _expand(ui_workflow)

    nodes = ui_workflow.get("nodes", [])
    links = ui_workflow.get("links", [])
    link_map = _build_link_map(links)

    # Node types that are UI-only and not real ComfyUI nodes
    SKIP_NODE_TYPES = {"Reroute"}

    api_workflow: dict[str, dict] = {}
    for node in nodes:
        node_type = node.get("type", "")
        if node_type in SKIP_NODE_TYPES:
            continue
        node_id = str(node.get("id"))
        api_workflow[node_id] = _convert_generic(node, link_map)

    return api_workflow


# ---------------------------------------------------------------------------
# Patching utilities
# ---------------------------------------------------------------------------

def find_nodes_by_type(ui_workflow: dict, node_type: str) -> list[dict]:
    """Return all nodes matching the given type."""
    return [n for n in ui_workflow.get("nodes", []) if n.get("type") == node_type]


def find_node_by_type(ui_workflow: dict, node_type: str, index: int = 0) -> dict | None:
    """Return the Nth node of the given type, or None if not found."""
    nodes = find_nodes_by_type(ui_workflow, node_type)
    return nodes[index] if 0 <= index < len(nodes) else None


def patch_widget_value(
    ui_workflow: dict,
    node_type: str,
    input_name: str,
    value: Any,
    node_index: int = 0,
) -> dict:
    """Patch a widget value in the UI workflow.

    The workflow is mutated in-place and also returned for convenience.

    Args:
        node_type: Node class type (e.g. "KSampler").
        input_name: Name of the input to patch (e.g. "seed").
        value: New value.
        node_index: If multiple nodes of this type exist, which one to patch.
    """
    node = find_node_by_type(ui_workflow, node_type, node_index)
    if node is None:
        raise ValueError(
            f"No node of type {node_type!r} (index {node_index}) found in workflow"
        )

    schema = NODE_SCHEMAS.get(node_type, [])
    ui_only = UI_ONLY_WIDGETS.get(node_type, {})
    ui_inputs = {inp["name"]: inp for inp in node.get("inputs", [])}

    # Determine the widget index for this input
    widget_idx = 0
    found = False
    for sch_name in schema:
        while widget_idx in ui_only:
            widget_idx += 1

        if sch_name == input_name:
            found = True
            break

        # Consume a widget slot if this schema entry maps to a widget
        if sch_name in ui_inputs and ui_inputs[sch_name].get("link") is None:
            widget_idx += 1
        elif sch_name not in ui_inputs:
            widget_idx += 1

    if not found:
        raise ValueError(f"Input {input_name!r} not found in schema for {node_type!r}")

    widgets = list(node.get("widgets_values", []))
    if widget_idx < len(widgets):
        widgets[widget_idx] = value
    else:
        # Extend widgets_values if necessary
        widgets.extend([None] * (widget_idx - len(widgets) + 1))
        widgets[widget_idx] = value

    node["widgets_values"] = widgets
    return ui_workflow


def patch_connection(
    ui_workflow: dict,
    target_node_id: int,
    target_input_name: str,
    source_node_id: int,
    source_output_index: int = 0,
) -> dict:
    """Add or update a connection in the UI workflow.

    Creates a new link if one doesn't exist, or reuses an existing link ID.
    Mutates the workflow in-place.
    """
    nodes = ui_workflow.get("nodes", [])
    links = ui_workflow.get("links", [])

    target_node = next((n for n in nodes if n.get("id") == target_node_id), None)
    source_node = next((n for n in nodes if n.get("id") == source_node_id), None)

    if target_node is None:
        raise ValueError(f"Target node {target_node_id} not found")
    if source_node is None:
        raise ValueError(f"Source node {source_node_id} not found")

    # Find or create the target input
    tgt_inp = next(
        (inp for inp in target_node.get("inputs", []) if inp["name"] == target_input_name),
        None,
    )
    if tgt_inp is None:
        raise ValueError(
            f"Input {target_input_name!r} not found on node {target_node_id}"
        )

    # Remove any existing link for this target input
    old_link_id = tgt_inp.get("link")
    if old_link_id is not None:
        links = [link for link in links if link[0] != old_link_id]
        # Remove from source node outputs as well
        for out in source_node.get("outputs", []):
            if old_link_id in out.get("links", []):
                out["links"].remove(old_link_id)

    # Generate new link ID
    last_link_id = ui_workflow.get("last_link_id", 0)
    new_link_id = last_link_id + 1
    ui_workflow["last_link_id"] = new_link_id

    # Determine input index
    tgt_input_idx = target_node["inputs"].index(tgt_inp)

    # Add new link
    new_link = [
        new_link_id,
        source_node_id,
        source_output_index,
        target_node_id,
        tgt_input_idx,
        source_node["outputs"][source_output_index]["type"],
    ]
    links.append(new_link)
    ui_workflow["links"] = links

    # Update target input
    tgt_inp["link"] = new_link_id

    # Update source output
    src_out = source_node["outputs"][source_output_index]
    src_out["links"] = src_out.get("links", []) + [new_link_id]

    return ui_workflow


def patch_ui_workflow(ui_workflow: dict, patches: dict[str, dict[str, Any]]) -> dict:
    """Batch-patch widget values by node type.

    Args:
        patches: Dict of node_type -> {input_name: value}
               If a node_type maps to a list of dicts, each dict patches a
               different instance of that node type in order.

    Returns:
        The mutated workflow dict.
    """
    for node_type, node_patches in patches.items():
        if isinstance(node_patches, list):
            for idx, inp_patches in enumerate(node_patches):
                for input_name, value in inp_patches.items():
                    patch_widget_value(ui_workflow, node_type, input_name, value, idx)
        else:
            for input_name, value in node_patches.items():
                patch_widget_value(ui_workflow, node_type, input_name, value, 0)

    return ui_workflow


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_ui_workflow(path: str | Path) -> dict:
    """Load a UI-format workflow from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ui_workflow(path: str | Path, workflow: dict) -> None:
    """Save a UI-format workflow to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)


def save_api_workflow(path: str | Path, workflow: dict) -> None:
    """Save an API-format workflow to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
