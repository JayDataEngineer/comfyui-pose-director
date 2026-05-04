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

    # Phase 2: Remove unavailable nodes and bridge around them
    # When a node is removed, downstream nodes that link to it need to be
    # reconnected to the removed node's source (pass-through bridging).
    # This prevents chain breakage when non-critical nodes (SeedVR2, Image Comparers)
    # are unavailable on the target ComfyUI instance.
    removed_ids: set[str] = set()
    # Map: removed_node_id -> {input_name: link_ref} (the sources feeding into removed nodes)
    removed_sources: dict[str, dict[str, Any]] = {}

    for nid in list(api.keys()):
        if api[nid]["class_type"] not in available:
            removed_sources[nid] = {
                k: v for k, v in api[nid].get("inputs", {}).items()
                if isinstance(v, list) and len(v) == 2
            }
            del api[nid]
            removed_ids.add(nid)

    # Phase 3: Bridge dangling references through removed nodes
    # For each dangling link pointing to a removed node, trace backward
    # through the removed node's own links to find an available source.
    valid_ids = set(api.keys())
    max_bridge_depth = 5  # Prevent infinite loops

    for nid, node in api.items():
        dangling = {
            k: v for k, v in node["inputs"].items()
            if isinstance(v, list) and len(v) == 2 and str(v[0]) not in valid_ids
        }
        for key, link_ref in dangling.items():
            # Try to bridge through removed nodes
            current_target = str(link_ref[0])
            desired_output = link_ref[1]
            bridged = False

            for _ in range(max_bridge_depth):
                if current_target not in removed_sources:
                    break
                # Look at the removed node's source links
                sources = removed_sources[current_target]
                # Try to find a source that provides the output index we need
                # For pass-through nodes, output 0 usually comes from the first linked input
                for src_key, src_ref in sources.items():
                    src_node = str(src_ref[0])
                    if src_node in valid_ids:
                        # Found an available source — bridge to it
                        node["inputs"][key] = src_ref
                        bridged = True
                        break
                    elif src_node in removed_ids:
                        # This source is also removed — follow the chain
                        current_target = src_node
                        break
                if bridged:
                    break

            if not bridged:
                # No bridge found — remove the dangling link
                del node["inputs"][key]

    # Phase 4: Iteratively remove nodes with unsatisfied required inputs
    # After bridging, some nodes may still lack required inputs (e.g., FaceDetailer
    # missing bbox_detector because UltralyticsDetectorProvider was removed).
    # These nodes must also be removed and bridged around.
    if object_info:
        for iteration in range(5):  # Max iterations for cascading removals
            to_remove = set()
            for nid, node in list(api.items()):
                ct = node.get("class_type", "")
                schema = object_info.get(ct)
                if not schema:
                    continue
                required_inputs = set(schema.get("input", {}).get("required", {}).keys())
                inputs = node.get("inputs", {})
                # Check if any required input is missing (deleted during bridging)
                missing_required = [k for k in required_inputs if k not in inputs]
                if missing_required:
                    to_remove.add(nid)

            if not to_remove:
                break  # All remaining nodes have their required inputs

            # Remove nodes with unsatisfied inputs and record their sources
            for nid in to_remove:
                removed_sources[nid] = {
                    k: v for k, v in api[nid].get("inputs", {}).items()
                    if isinstance(v, list) and len(v) == 2
                }
                del api[nid]
                removed_ids.add(nid)

            # Re-bridge any new dangling references
            valid_ids = set(api.keys())
            for nid, node in api.items():
                dangling = {
                    k: v for k, v in list(node["inputs"].items())
                    if isinstance(v, list) and len(v) == 2 and str(v[0]) not in valid_ids
                }
                for key, link_ref in dangling.items():
                    current_target = str(link_ref[0])
                    bridged = False

                    for _ in range(max_bridge_depth):
                        if current_target not in removed_sources:
                            break
                        sources = removed_sources[current_target]
                        for src_key, src_ref in sources.items():
                            src_node = str(src_ref[0])
                            if src_node in valid_ids:
                                node["inputs"][key] = src_ref
                                bridged = True
                                break
                            elif src_node in removed_ids:
                                current_target = src_node
                                break
                        if bridged:
                            break

                    if not bridged:
                        del node["inputs"][key]

    return api
