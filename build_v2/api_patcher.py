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
