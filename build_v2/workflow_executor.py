"""Unified workflow executor — load, patch, submit, save for both UI and API formats."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for pipeline imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline.comfyui_client import submit_workflow, get_output_images, upload_image
from pipeline.ui_to_api import load_ui_workflow, patch_ui_workflow, convert_ui_to_api
from pipeline.workflow_sanitize import sanitize_api_workflow


def log(msg: str):
    print(f"  {msg}", flush=True)


class WorkflowExecutor:
    """Handles the load→patch→submit→get pattern for both UI and API format workflows."""

    def __init__(self, comfyui_url: str):
        self.url = comfyui_url.rstrip("/")
        self._available_nodes: set[str] | None = None
        self._object_info: dict | None = None

    def _get_available_nodes(self) -> set[str]:
        """Fetch registered node types from ComfyUI (cached per executor)."""
        if self._available_nodes is None:
            import requests
            resp = requests.get(f"{self.url}/object_info", timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self._available_nodes = set(data.keys())
            self._object_info = data
        return self._available_nodes

    def _get_object_info(self) -> dict:
        """Full object_info dict (cached, fetched with available nodes)."""
        if self._object_info is None:
            self._get_available_nodes()
        return self._object_info

    def free_memory(self):
        """Free GPU and system memory on ComfyUI between stages."""
        import requests
        try:
            requests.post(f"{self.url}/free", json={"unload_models": True, "free_memory": True}, timeout=30)
        except Exception:
            pass  # Non-critical — best effort

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
        result = submit_workflow(api, timeout=timeout, base_url=self.url)

        save_node = next(
            (n for n in wf["nodes"] if n["type"] == output_node_type), None
        )
        if not save_node:
            raise RuntimeError(f"No {output_node_type} node found in workflow")

        return get_output_images(result["outputs"], save_node["id"], base_url=self.url)

    def run_api_workflow(
        self,
        workflow_path: Path,
        patches: dict[str, dict],
        output_node_id: str,
        timeout: int = 300,
        checkpoint_override: str | None = None,
        rmbg_override: str | None = None,
    ) -> list[bytes]:
        """Load API-format workflow, patch by node ID, submit, return output images.

        patches: {"node_id": {"input_name": value, ...}, ...}
        checkpoint_override: if set, swap all CheckpointLoaderSimple ckpt_name
        rmbg_override: if set, swap all VNCCS_RMBG2 model selection
        """
        from build_v2.api_patcher import (
            load_api_workflow, patch_api_workflow, normalize_paths,
            fix_inputs_from_schema, fix_model_paths,
        )

        api = load_api_workflow(workflow_path)
        normalize_paths(api)
        patch_api_workflow(api, patches)

        # Apply deployment overrides (before schema fix so they validate correctly)
        if checkpoint_override:
            for node in api.values():
                if node.get("class_type") == "CheckpointLoaderSimple":
                    node["inputs"]["ckpt_name"] = checkpoint_override
        if rmbg_override:
            for node in api.values():
                if node.get("class_type") == "VNCCS_RMBG2":
                    node["inputs"]["model"] = rmbg_override

        obj_info = self._get_object_info()
        fix_model_paths(api, obj_info)
        fix_inputs_from_schema(api, obj_info)
        sanitize_api_workflow(api, self._get_available_nodes(), obj_info)
        result = submit_workflow(api, timeout=timeout, base_url=self.url)
        return get_output_images(result["outputs"], output_node_id, base_url=self.url)

    def upload_character(self, image_path: Path, name: str) -> str:
        """Upload a character image to ComfyUI. Returns the ComfyUI input path."""
        upload_image(image_path.read_bytes(), f"{name}.png", "PoseDirector", base_url=self.url)
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
