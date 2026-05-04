"""Sanitize expanded workflows by removing unavailable nodes.

Connects the workflow graph through missing nodes where possible,
or removes dangling branches that depend on unavailable nodes.

Usage::

    from pipeline.workflow_sanitize import sanitize_workflow
    sanitized = sanitize_workflow(expanded_workflow, available_types)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# Node types that are safe to bypass (they pass through or do cleanup).
# These get removed and their connections rewired if they have a
# single input and single output of the same type.
BYPASS_SAFE = {
    "easy cleanGpuUsed",
    "Fast Groups Bypasser (rgthree)",
    "Image Comparer (rgthree)",
    "PreviewImage",
}

# Node types that should be completely removed along with their
# downstream dependents.
REMOVE_WITH_DOWNSTREAM = {
    "FaceDetailer",          # quality improvement, not essential
    "SeedVR2LoadDiTModel",   # upscaling, not essential
    "SeedVR2LoadVAEModel",   # upscaling, not essential
    "SeedVR2VideoUpscaler",  # upscaling, not essential
    "UltralyticsDetectorProvider",  # detection, used by FaceDetailer
}

# Nodes that process text/prompt but can be replaced with direct input
PASS_THROUGH_INPUTS = {
    "Text Find and Replace": {"input_name": "text", "output_name": "RESULT"},
}


def sanitize_workflow(
    ui_workflow: dict,
    available_types: set[str] | None = None,
) -> dict:
    """Remove nodes whose types aren't available and rewire the graph.

    Args:
        ui_workflow: Expanded UI-format workflow (no UUID nodes).
        available_types: Set of available class_type strings.
            If None, queries the configured ComfyUI instance.

    Returns:
        New workflow dict with unavailable nodes removed.
    """
    wf = deepcopy(ui_workflow)

    if available_types is None:
        available_types = _fetch_available_types()

    nodes = wf["nodes"]
    links = wf["links"]

    # Build link maps
    # link_by_id: link_id -> link tuple [id, src_node, src_slot, tgt_node, tgt_slot, type]
    link_by_id = {l[0]: l for l in links}

    # Build node lookup
    node_by_id = {n["id"]: n for n in nodes}

    # Identify unavailable nodes
    unavailable = set()
    for node in nodes:
        ntype = node.get("type", "")
        if ntype and ntype not in available_types and ntype != "Reroute":
            unavailable.add(node["id"])

    if not unavailable:
        return wf

    # Iteratively remove unavailable nodes
    changed = True
    while changed:
        changed = False
        for node_id in list(unavailable):
            node = node_by_id.get(node_id)
            if node is None:
                unavailable.discard(node_id)
                changed = True
                continue

            node_type = node.get("type", "")

            # Try to bypass: find if this node has a single input connected
            # to a single output of the same type, and rewire around it.
            removed = _try_remove_node(node, nodes, links, link_by_id, node_by_id)
            if removed:
                nodes.remove(node)
                del node_by_id[node_id]
                unavailable.discard(node_id)
                changed = True

    # Clean up any orphan links
    valid_node_ids = {n["id"] for n in nodes}
    links[:] = [l for l in links if l[1] in valid_node_ids and l[3] in valid_node_ids]

    # Update link references on remaining nodes
    valid_link_ids = {l[0] for l in links}
    for node in nodes:
        for inp in node.get("inputs", []):
            if inp.get("link") is not None and inp["link"] not in valid_link_ids:
                inp["link"] = None
        for out in node.get("outputs", []):
            out_links = out.get("links") or []
            out["links"] = [lid for lid in out_links if lid in valid_link_ids]

    wf["last_node_id"] = max(n.get("id", 0) for n in nodes) if nodes else 0
    wf["last_link_id"] = max(l[0] for l in links) if links else 0

    return wf


def _try_remove_node(
    node: dict,
    nodes: list[dict],
    links: list[list],
    link_by_id: dict[int, list],
    node_by_id: dict[int, dict],
) -> bool:
    """Try to remove a node by rewiring connections around it.

    Returns True if the node was successfully removed and connections rewired.
    """
    node_id = node["id"]
    node_type = node.get("type", "")

    # For bypass-safe nodes with one input/output of same type: rewire
    if node_type in BYPASS_SAFE:
        return _rewire_passthrough(node, links, link_by_id, node_by_id)

    # For pass-through text processors: rewire
    if node_type in PASS_THROUGH_INPUTS:
        return _rewire_passthrough(node, links, link_by_id, node_by_id)

    # For nodes with widget-only inputs and no essential outputs: just remove
    # (they produce optional side effects like saving, previewing, etc.)
    inputs_with_links = [
        inp for inp in node.get("inputs", [])
        if inp.get("link") is not None
    ]
    if not inputs_with_links:
        # No connected inputs - standalone node, safe to remove
        _disconnect_all_outputs(node, links, link_by_id, node_by_id)
        return True

    # For nodes with connected inputs but optional outputs:
    # Remove the node and disconnect all downstream
    outputs_with_links = [
        out for out in node.get("outputs", [])
        if out.get("links")
    ]
    if not outputs_with_links:
        # Has inputs but no connected outputs - it's a sink node
        _disconnect_all_inputs(node, links, link_by_id, node_by_id)
        return True

    # Complex case: try to rewire by matching types
    return _rewire_by_type(node, links, link_by_id, node_by_id)


def _rewire_passthrough(
    node: dict,
    links: list[list],
    link_by_id: dict[int, list],
    node_by_id: dict[int, dict],
) -> bool:
    """Rewire a node that passes through its input to its output."""
    node_id = node["id"]

    # Find the first connected input
    incoming_link_id = None
    incoming_input_slot = None
    for slot, inp in enumerate(node.get("inputs", [])):
        if inp.get("link") is not None:
            incoming_link_id = inp["link"]
            incoming_input_slot = slot
            break

    if incoming_link_id is None:
        # No connected input - disconnect outputs and remove
        _disconnect_all_outputs(node, links, link_by_id, node_by_id)
        return True

    incoming_link = link_by_id.get(incoming_link_id)
    if incoming_link is None:
        return False

    src_node_id = incoming_link[1]
    src_output_slot = incoming_link[2]
    src_type = incoming_link[5]

    # Find matching output (same type)
    matching_output_slot = None
    for slot, out in enumerate(node.get("outputs", [])):
        if out.get("type") == src_type and out.get("links"):
            matching_output_slot = slot
            break

    if matching_output_slot is None:
        # No matching output - disconnect everything
        _disconnect_all_inputs(node, links, link_by_id, node_by_id)
        _disconnect_all_outputs(node, links, link_by_id, node_by_id)
        return True

    # Rewire: connect source directly to all downstream targets
    outgoing_links = list(node["outputs"][matching_output_slot].get("links", []))
    for out_link_id in outgoing_links:
        out_link = link_by_id.get(out_link_id)
        if out_link is None:
            continue
        # Update this link to come from the source instead
        out_link[1] = src_node_id
        out_link[2] = src_output_slot
        # Update source node's output links
        src_node = node_by_id.get(src_node_id)
        if src_node and src_output_slot < len(src_node.get("outputs", [])):
            src_out = src_node["outputs"][src_output_slot]
            if out_link_id not in src_out.get("links", []):
                src_out.setdefault("links", []).append(out_link_id)

    # Remove the incoming link
    links[:] = [l for l in links if l[0] != incoming_link_id]
    del link_by_id[incoming_link_id]

    # Remove all other links to/from this node
    links[:] = [l for l in links if l[1] != node_id and l[3] != node_id]
    # Rebuild link_by_id
    link_by_id.clear()
    link_by_id.update({l[0]: l for l in links})

    # Update source node output (remove old link)
    src_node = node_by_id.get(src_node_id)
    if src_node and src_output_slot < len(src_node.get("outputs", [])):
        src_out_links = src_node["outputs"][src_output_slot].get("links", [])
        if incoming_link_id in src_out_links:
            src_out_links.remove(incoming_link_id)

    return True


def _disconnect_all_inputs(
    node: dict,
    links: list[list],
    link_by_id: dict[int, list],
    node_by_id: dict[int, dict],
) -> None:
    """Remove all links coming into this node's inputs."""
    node_id = node["id"]
    for inp in node.get("inputs", []):
        lid = inp.get("link")
        if lid is not None:
            link = link_by_id.get(lid)
            if link:
                # Remove from source node's output links
                src_node = node_by_id.get(link[1])
                if src_node:
                    for out in src_node.get("outputs", []):
                        if lid in out.get("links", []):
                            out["links"].remove(lid)
            links[:] = [l for l in links if l[0] != lid]
            link_by_id.pop(lid, None)
            inp["link"] = None


def _disconnect_all_outputs(
    node: dict,
    links: list[list],
    link_by_id: dict[int, list],
    node_by_id: dict[int, dict],
) -> None:
    """Remove all links going out from this node's outputs, recursively."""
    node_id = node["id"]
    for out in node.get("outputs", []):
        for lid in list(out.get("links", [])):
            link = link_by_id.get(lid)
            if link:
                # Clear target node's input
                tgt_node = node_by_id.get(link[3])
                if tgt_node:
                    for inp in tgt_node.get("inputs", []):
                        if inp.get("link") == lid:
                            inp["link"] = None
            links[:] = [l for l in links if l[0] != lid]
            link_by_id.pop(lid, None)
    out["links"] = []


def _rewire_by_type(
    node: dict,
    links: list[list],
    link_by_id: dict[int, list],
    node_by_id: dict[int, dict],
) -> bool:
    """Try to rewire a node by matching input and output types."""
    node_id = node["id"]

    # For each output, find an input of the same type
    for out_slot, out in enumerate(node.get("outputs", [])):
        out_type = out.get("type", "")
        out_links = list(out.get("links", []))

        for in_slot, inp in enumerate(node.get("inputs", [])):
            if inp.get("type", "") == out_type and inp.get("link") is not None:
                # Found matching input - rewire
                in_link_id = inp["link"]
                in_link = link_by_id.get(in_link_id)
                if in_link is None:
                    continue

                src_node_id = in_link[1]
                src_output_slot = in_link[2]

                # Rewire all outgoing links
                for out_link_id in out_links:
                    out_link = link_by_id.get(out_link_id)
                    if out_link is None:
                        continue
                    out_link[1] = src_node_id
                    out_link[2] = src_output_slot

                    # Update source
                    src_node = node_by_id.get(src_node_id)
                    if src_node:
                        src_out = src_node["outputs"][src_output_slot]
                        if out_link_id not in src_out.get("links", []):
                            src_out.setdefault("links", []).append(out_link_id)

                # Remove the incoming link
                links[:] = [l for l in links if l[0] != in_link_id]
                link_by_id.pop(in_link_id, None)
                break

    # Remove remaining links to/from this node
    links[:] = [l for l in links if l[1] != node_id and l[3] != node_id]
    link_by_id.clear()
    link_by_id.update({l[0]: l for l in links})

    return True


def _fetch_available_types() -> set[str]:
    """Fetch available node types from the configured ComfyUI instance."""
    import os
    import json
    import urllib.request

    url = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8465")
    req = urllib.request.Request(f"{url}/object_info")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return set(data.keys())
