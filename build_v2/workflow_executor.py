"""Unified workflow executor — load, patch, submit, save for both UI and API formats."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for pipeline imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline.comfyui_client import submit_workflow, get_output_images, upload_image
from pipeline.ui_to_api import load_ui_workflow, patch_ui_workflow, convert_ui_to_api


def log(msg: str):
    print(f"  {msg}", flush=True)


class WorkflowExecutor:
    """Handles the load→patch→submit→get pattern for both UI and API format workflows."""

    def __init__(self, comfyui_url: str):
        self.url = comfyui_url

    def run_ui_workflow(
        self,
        workflow_path: Path,
        patches: dict[str, dict],
        output_node_type: str = "SaveImage",
        timeout: int = 300,
    ) -> list[bytes]:
        """Load UI workflow, patch by node type, convert, submit, return output images.

        patches: {"NodeType": {"input_name": value, ...}, ...}
        """
        wf = load_ui_workflow(str(workflow_path))
        patch_ui_workflow(wf, patches)
        api = convert_ui_to_api(wf)
        result = submit_workflow(api, timeout=timeout)

        # Find the save node ID
        save_node = next(
            (n for n in wf["nodes"] if n["type"] == output_node_type), None
        )
        if not save_node:
            raise RuntimeError(f"No {output_node_type} node found in workflow")

        return get_output_images(result["outputs"], save_node["id"])

    def run_api_workflow(
        self,
        workflow_path: Path,
        patches: dict[str, dict],
        output_node_id: str,
        timeout: int = 300,
    ) -> list[bytes]:
        """Load API-format workflow, patch by node ID, submit, return output images.

        patches: {"node_id": {"input_name": value, ...}, ...}
        """
        from build_v2.api_patcher import load_api_workflow, patch_api_workflow

        api = load_api_workflow(workflow_path)
        patch_api_workflow(api, patches)
        result = submit_workflow(api, timeout=timeout)
        return get_output_images(result["outputs"], output_node_id)

    def upload_character(self, image_path: Path, name: str) -> str:
        """Upload a character image to ComfyUI. Returns the ComfyUI input path."""
        upload_image(image_path.read_bytes(), f"{name}.png", "PoseDirector")
        return f"PoseDirector/{name}.png"

    def upload_and_run_ui(
        self,
        workflow_path: Path,
        patches: dict[str, dict],
        character_image_path: Path,
        character_upload_name: str,
        output_node_type: str = "SaveImage",
        timeout: int = 300,
    ) -> list[bytes]:
        """Upload character image, then run UI workflow. Convenience method."""
        comfyui_name = self.upload_character(character_image_path, character_upload_name)
        if "LoadImage" in patches:
            patches["LoadImage"]["image"] = comfyui_name
        else:
            patches["LoadImage"] = {"image": comfyui_name}
        return self.run_ui_workflow(workflow_path, patches, output_node_type, timeout)
