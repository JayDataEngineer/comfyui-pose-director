"""Sanitize API-format workflows by fixing unavailable nodes.

Two operations:
1. Remap node types to available equivalents (aliases)
2. Remove nodes whose types are still unavailable, then prune dangling refs

Works on API format (dict of node_id -> {class_type, inputs}) which is
much simpler than the UI-format graph manipulation.

Usage::

    from pipeline.workflow_sanitize import sanitize_api_workflow
    api = sanitize_api_workflow(api, available_types)
"""

from __future__ import annotations

from typing import Any


# Node type aliases: workflow uses this name -> we remap to available equivalent.
# Only add entries where the replacement has compatible inputs/outputs.
NODE_ALIASES: dict[str, str] = {
    "Text Find and Replace": "ReplaceText",
    "Text Concatenate": "VNCCS_PromptConcat",
}

# Input name remapping for aliased nodes: {old_class_type: {old_input: new_input}}
INPUT_ALIASES: dict[str, dict[str, str]] = {
    "Text Concatenate": {
        "text_a": "a", "text_b": "b", "text_c": "c", "text_d": "d",
        "delimiter": "separator",
    },
}


def sanitize_api_workflow(
    api: dict[str, dict],
    available: set[str],
    object_info: dict | None = None,
) -> dict[str, dict]:
    """Remap aliases, remove unavailable nodes, prune dangling references.

    Args:
        api: API-format workflow dict (node_id -> {class_type, inputs}).
        available: Set of registered class_type strings on the target ComfyUI.
        object_info: Optional full object_info dict for pruning unknown inputs
            after alias remapping.

    Returns:
        Cleaned API-format workflow dict.
    """
    # Phase 1: Remap aliases + input names
    for nid, node in api.items():
        ct = node["class_type"]
        if ct in NODE_ALIASES and NODE_ALIASES[ct] in available:
            # Remap input names for this node type
            input_map = INPUT_ALIASES.get(ct, {})
            inputs = node.get("inputs", {})
            for old_name, new_name in input_map.items():
                if old_name in inputs:
                    inputs[new_name] = inputs.pop(old_name)
            node["class_type"] = NODE_ALIASES[ct]

    # Phase 1b: Prune unknown inputs after alias remap (if schema available)
    if object_info:
        for nid, node in api.items():
            ct = node.get("class_type", "")
            schema = object_info.get(ct)
            if not schema:
                continue
            valid_inputs = set()
            for category in ("required", "optional"):
                valid_inputs.update(schema.get("input", {}).get(category, {}).keys())
            inputs = node.get("inputs", {})
            unknown = [k for k in inputs if k not in valid_inputs]
            for key in unknown:
                # Only remove literal values, not link references
                v = inputs[key]
                if not (isinstance(v, list) and len(v) == 2):
                    del inputs[key]

    # Phase 2: Remove unavailable nodes
    for nid in list(api.keys()):
        if api[nid]["class_type"] not in available:
            del api[nid]

    # Phase 3: Remove dangling link references
    # A link is [source_node_id, output_index] — source must exist
    valid_ids = set(api.keys())
    for nid, node in api.items():
        dangling = [
            k for k, v in node["inputs"].items()
            if isinstance(v, list) and len(v) == 2 and str(v[0]) not in valid_ids
        ]
        for key in dangling:
            del node["inputs"][key]

    return api
