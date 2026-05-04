"""Expand ComfyUI subgraph (UUID) nodes into real nodes.

ComfyUI's newer workflow format allows grouping nodes into "subgraphs" with
UUID class_types. The subgraph definitions live in ``definitions.subgraphs``.

Each subgraph definition contains:
- ``nodes``: internal nodes with real class_types
- ``links``: internal connections, using ``-10`` as input boundary and
  ``-20`` as output boundary
- ``inputs`` / ``outputs``: external interface definitions

This module expands UUID nodes so the workflow can be converted to API format
by the existing ``ui_to_api`` converter.

Usage::

    from pipeline.subgraph_expander import expand_subgraphs

    workflow = load_ui_workflow("workflows/vnccs_step1.1.json")
    expanded = expand_subgraphs(workflow)
    api_workflow = convert_ui_to_api(expanded)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _build_node_lookup(nodes: list[dict]) -> dict[int, dict]:
    """Build node_id -> node dict."""
    return {n["id"]: n for n in nodes}


def expand_subgraphs(ui_workflow: dict) -> dict:
    """Expand all subgraph (UUID) nodes into their constituent real nodes.

    The workflow is deep-copied. UUID nodes are replaced by the real nodes
    from their subgraph definitions, with all connections rewired.

    Args:
        ui_workflow: Dict loaded from a ComfyUI UI-format JSON file.

    Returns:
        New workflow dict with all subgraph nodes expanded.
    """
    wf = deepcopy(ui_workflow)
    definitions = {
        sg["id"]: sg for sg in wf.get("definitions", {}).get("subgraphs", [])
    }
    if not definitions:
        return wf

    nodes: list[dict] = wf["nodes"]
    links: list[list] = wf["links"]

    # Keep expanding until no UUID nodes remain. We process one at a time
    # because expanding a UUID node may change indices and connections.
    while True:
        uuid_node = None
        uuid_idx = -1
        for i, n in enumerate(nodes):
            if n.get("type", "") in definitions:
                uuid_node = n
                uuid_idx = i
                break

        if uuid_node is None:
            break

        _expand_one(uuid_node, uuid_idx, nodes, links, definitions, wf)

    # Clean up
    wf.pop("definitions", None)
    wf["last_node_id"] = max(n.get("id", 0) for n in nodes) if nodes else 0
    wf["last_link_id"] = max(l[0] for l in links) if links else 0

    return wf


def _expand_one(
    uuid_node: dict,
    uuid_idx: int,
    nodes: list[dict],
    links: list[list],
    definitions: dict[str, dict],
    wf: dict,
) -> None:
    """Expand a single UUID node in-place within the node/link lists."""
    sg_def = definitions[uuid_node["type"]]
    uuid_id: int = uuid_node["id"]

    # --- ID allocation ---
    max_node_id = max(n.get("id", 0) for n in nodes)
    max_link_id = max(l[0] for l in links) if links else wf.get("last_link_id", 0)

    # Map internal node IDs -> new unique top-level IDs
    id_map: dict[int, int] = {}
    for inode in sg_def["nodes"]:
        max_node_id += 1
        id_map[inode["id"]] = max_node_id

    # --- Boundary maps ---
    # input_boundary: uuid_input_slot -> (new_internal_node_id, internal_input_slot)
    input_boundary: dict[int, tuple[int, int]] = {}
    # output_boundary: uuid_output_slot -> (new_internal_node_id, internal_output_slot)
    output_boundary: dict[int, tuple[int, int]] = {}

    for sl in sg_def["links"]:
        if sl["origin_id"] == -10:
            input_boundary[sl["origin_slot"]] = (
                id_map[sl["target_id"]],
                sl["target_slot"],
            )
        elif sl["target_id"] == -20:
            output_boundary[sl["target_slot"]] = (
                id_map[sl["origin_id"]],
                sl["origin_slot"],
            )

    # --- Add expanded internal nodes ---
    for inode in sg_def["nodes"]:
        new_node = deepcopy(inode)
        new_node["id"] = id_map[inode["id"]]
        # Clear link references — they'll be rewired below
        for inp in new_node.get("inputs", []):
            inp["link"] = None
        for out in new_node.get("outputs", []):
            out["links"] = []
        nodes.append(new_node)

    # --- Build link lookup for the current top-level graph ---
    link_by_id: dict[int, list] = {l[0]: l for l in links}

    # --- Wire internal links (non-boundary) ---
    for sl in sg_def["links"]:
        if sl["origin_id"] == -10 or sl["target_id"] == -20:
            continue

        max_link_id += 1
        src_new = id_map[sl["origin_id"]]
        tgt_new = id_map[sl["target_id"]]

        new_link = [
            max_link_id,
            src_new,
            sl["origin_slot"],
            tgt_new,
            sl["target_slot"],
            sl["type"],
        ]
        links.append(new_link)

        # Update expanded node references
        node_lookup = _build_node_lookup(nodes)
        src_node = node_lookup.get(src_new)
        tgt_node = node_lookup.get(tgt_new)
        if src_node and sl["origin_slot"] < len(src_node.get("outputs", [])):
            src_node["outputs"][sl["origin_slot"]]["links"].append(max_link_id)
        if tgt_node and sl["target_slot"] < len(tgt_node.get("inputs", [])):
            tgt_node["inputs"][sl["target_slot"]]["link"] = max_link_id

    # --- Collect links to remove (old links to/from UUID node) ---
    links_to_remove: set[int] = set()

    # --- Rewire external INPUTS ---
    # For each UUID node input that has an external connection,
    # redirect to the corresponding internal node via input_boundary.
    for slot, inp in enumerate(uuid_node.get("inputs", [])):
        ext_link_id = inp.get("link")
        if ext_link_id is None or slot not in input_boundary:
            continue

        ext_link = link_by_id.get(ext_link_id)
        if ext_link is None:
            continue

        new_tgt_id, new_tgt_slot = input_boundary[slot]
        max_link_id += 1

        new_link = [
            max_link_id,
            ext_link[1],  # external source node ID
            ext_link[2],  # external source output slot
            new_tgt_id,
            new_tgt_slot,
            ext_link[5],  # type
        ]
        links.append(new_link)

        # Update external source node: swap old link for new
        node_lookup = _build_node_lookup(nodes)
        src_node = node_lookup.get(ext_link[1])
        if src_node:
            for out in src_node.get("outputs", []):
                old_links = out.get("links") or []
                if ext_link_id in old_links:
                    old_links.remove(ext_link_id)
                    old_links.append(max_link_id)
                    out["links"] = old_links

        # Update internal target node
        tgt_node = node_lookup.get(new_tgt_id)
        if tgt_node and new_tgt_slot < len(tgt_node.get("inputs", [])):
            tgt_node["inputs"][new_tgt_slot]["link"] = max_link_id

        links_to_remove.add(ext_link_id)

    # --- Rewire external OUTPUTS ---
    for slot, out in enumerate(uuid_node.get("outputs", [])):
        ext_link_ids = list(out.get("links") or [])
        if not ext_link_ids or slot not in output_boundary:
            continue

        new_src_id, new_src_slot = output_boundary[slot]

        for ext_link_id in ext_link_ids:
            ext_link = link_by_id.get(ext_link_id)
            if ext_link is None:
                continue

            max_link_id += 1
            new_link = [
                max_link_id,
                new_src_id,
                new_src_slot,
                ext_link[3],  # external target node ID
                ext_link[4],  # external target input slot
                ext_link[5],
            ]
            links.append(new_link)

            # Update internal source node
            node_lookup = _build_node_lookup(nodes)
            src_node = node_lookup.get(new_src_id)
            if src_node and new_src_slot < len(src_node.get("outputs", [])):
                out_links = src_node["outputs"][new_src_slot].get("links") or []
                out_links.append(max_link_id)
                src_node["outputs"][new_src_slot]["links"] = out_links

            # Update external target node: swap old link for new
            tgt_node = node_lookup.get(ext_link[3])
            if tgt_node:
                for inp in tgt_node.get("inputs", []):
                    if inp.get("link") == ext_link_id:
                        inp["link"] = max_link_id

            links_to_remove.add(ext_link_id)

    # --- Cleanup ---
    # Remove old links
    links[:] = [l for l in links if l[0] not in links_to_remove]

    # Remove UUID node
    nodes.pop(uuid_idx)

    # Update workflow metadata
    wf["last_node_id"] = max_node_id
    wf["last_link_id"] = max_link_id
