"""Unit tests for pipeline/ui_to_api.py — UI-to-API workflow converter."""

import json
from pathlib import Path

import pytest

from pipeline.ui_to_api import (
    convert_ui_to_api,
    patch_ui_workflow,
    patch_widget_value,
    patch_connection,
    find_nodes_by_type,
    load_ui_workflow,
    NODE_SCHEMAS,
)

WORKFLOWS_DIR = Path(__file__).parent.parent.parent / "workflows"


class TestConvertUIToAPI:
    """Tests for convert_ui_to_api()."""

    def _load_and_convert(self, name: str) -> dict:
        ui = load_ui_workflow(WORKFLOWS_DIR / name)
        return convert_ui_to_api(ui)

    def test_motion_director_conversion(self):
        """MotionDirector should map rotation-only params correctly."""
        api = self._load_and_convert("01_motion_keyframes_rotation.json")
        md = api["1"]
        assert md["class_type"] == "MotionDirector"
        assert md["inputs"]["motion_description"] == "Character waves their right hand in a friendly greeting"
        assert md["inputs"]["num_keyframes"] == 4
        assert md["inputs"]["total_frames"] == 48
        assert md["inputs"]["base_url"] == "https://openrouter.ai/api/v1"
        assert md["inputs"]["model"] == "deepseek/deepseek-v4-flash"
        assert md["inputs"]["width"] == 1024
        assert md["inputs"]["height"] == 1024
        assert md["inputs"]["max_retries"] == 2
        # No legacy inputs
        assert "keypoints" not in md["inputs"]
        assert "pose_keypoint" not in md["inputs"]
        assert "output_mode" not in md["inputs"]
        assert "api_key" not in md["inputs"]

    def test_preview_image_conversion(self):
        """PreviewImage should map correctly."""
        api = self._load_and_convert("01_motion_keyframes_rotation.json")
        pi = api["4"]
        assert pi["class_type"] == "PreviewImage"
        assert pi["inputs"]["images"] == ["1", 0]  # connected to MotionDirector output 0

    def test_ksampler_skips_control_after_generate(self):
        """KSampler should skip the UI-only control_after_generate widget."""
        api = self._load_and_convert("00_z_image_character.json")
        ks = api["8"]
        assert ks["class_type"] == "KSampler"
        assert ks["inputs"]["seed"] == 42
        assert ks["inputs"]["steps"] == 8
        assert ks["inputs"]["cfg"] == 1.0
        assert ks["inputs"]["sampler_name"] == "res_multistep"
        assert ks["inputs"]["scheduler"] == "simple"
        assert "control_after_generate" not in ks["inputs"]

    def test_ltx_video_assembly(self):
        """LTX video workflow should convert all nodes correctly."""
        api = self._load_and_convert("04_ltx_video_assembly.json")
        assert api["1"]["class_type"] == "LoadImage"
        assert api["1"]["inputs"]["image"] == "PoseDirector/edited_kf_0.png"
        assert api["5"]["class_type"] == "UnetLoaderGGUF"
        assert api["5"]["inputs"]["unet_name"] == "ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf"
        assert api["9"]["class_type"] == "EmptyLTXVLatentVideo"
        assert api["9"]["inputs"]["width"] == 512
        assert api["9"]["inputs"]["length"] == 33
        assert api["10"]["class_type"] == "LTXVAddGuideAdvanced"
        assert api["10"]["inputs"]["frame_idx"] == 0
        assert api["11"]["inputs"]["frame_idx"] == 8
        assert api["16"]["class_type"] == "VHS_VideoCombine"
        assert api["16"]["inputs"]["frame_rate"] == 24

    def test_vnccs_encoder(self):
        """VNCCS_QWEN_Encoder should map its many inputs correctly."""
        api = self._load_and_convert("03_motion_keyframes_qwen_edit_mesh.json")
        vnccs = api["9"]
        assert vnccs["class_type"] == "VNCCS_QWEN_Encoder"
        assert vnccs["inputs"]["prompt"] == "Draw character from image2"
        assert vnccs["inputs"]["latent_image_index"] == 1
        assert vnccs["inputs"]["image1_name"] == "Character Reference"
        assert vnccs["inputs"]["target_size"] == 1024
        assert vnccs["inputs"]["weight1"] == 1.0
        assert vnccs["inputs"]["vl_size"] == 384
        assert vnccs["inputs"]["qwen_2511"] is True

    def test_clip_loader_device_omitted_when_default(self):
        """CLIPLoader should omit device when no widget value is provided."""
        api = self._load_and_convert("00_z_image_character.json")
        clip = api["2"]
        assert clip["inputs"]["clip_name"] == "qwen_3_4b.safetensors"
        assert clip["inputs"]["type"] == "lumina2"
        assert "device" not in clip["inputs"]

    def test_connection_references_are_strings(self):
        """All connection references should have string node IDs."""
        api = self._load_and_convert("04_ltx_video_assembly.json")
        for node_id, node_def in api.items():
            for val in node_def["inputs"].values():
                if isinstance(val, list):
                    assert isinstance(val[0], str), f"Node {node_id}: {val}"


class TestPatchWidgetValue:
    """Tests for patch_widget_value()."""

    def test_patch_ksampler_seed(self):
        ui = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
        patch_widget_value(ui, "KSampler", "seed", 999)
        api = convert_ui_to_api(ui)
        assert api["8"]["inputs"]["seed"] == 999

    def test_patch_clip_text(self):
        ui = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
        patch_widget_value(ui, "CLIPTextEncode", "text", "new prompt")
        api = convert_ui_to_api(ui)
        assert api["5"]["inputs"]["text"] == "new prompt"

    def test_patch_motion_director(self):
        ui = load_ui_workflow(WORKFLOWS_DIR / "01_motion_keyframes_rotation.json")
        patch_widget_value(ui, "MotionDirector", "motion_description", "jump")
        api = convert_ui_to_api(ui)
        assert api["1"]["inputs"]["motion_description"] == "jump"

    def test_patch_by_index(self):
        ui = load_ui_workflow(WORKFLOWS_DIR / "04_ltx_video_assembly.json")
        # There are 4 LoadImage nodes in this workflow
        patch_widget_value(ui, "LoadImage", "image", "kf_A.png", node_index=0)
        patch_widget_value(ui, "LoadImage", "image", "kf_B.png", node_index=1)
        api = convert_ui_to_api(ui)
        assert api["1"]["inputs"]["image"] == "kf_A.png"
        assert api["2"]["inputs"]["image"] == "kf_B.png"

    def test_raises_on_missing_node_type(self):
        ui = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
        with pytest.raises(ValueError, match="No node of type"):
            patch_widget_value(ui, "NonExistent", "x", 1)

    def test_raises_on_missing_input_name(self):
        ui = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
        with pytest.raises(ValueError, match="not found in schema"):
            patch_widget_value(ui, "KSampler", "nonexistent", 1)


class TestPatchConnection:
    """Tests for patch_connection()."""

    def test_connect_vaedecode_to_saveimage(self):
        """Fix the missing connection in 00_z_image_character.json."""
        ui = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
        # Node 9 is VAEDecode, node 10 is SaveImage
        patch_connection(ui, target_node_id=10, target_input_name="images",
                         source_node_id=9, source_output_index=0)
        api = convert_ui_to_api(ui)
        assert api["10"]["inputs"]["images"] == ["9", 0]


class TestPatchUIWorkflow:
    """Tests for patch_ui_workflow() batch patcher."""

    def test_batch_patch(self):
        ui = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
        patch_ui_workflow(ui, {
            "KSampler": {"seed": 123, "steps": 12},
            "CLIPTextEncode": {"text": "hello"},
        })
        api = convert_ui_to_api(ui)
        assert api["8"]["inputs"]["seed"] == 123
        assert api["8"]["inputs"]["steps"] == 12
        assert api["5"]["inputs"]["text"] == "hello"

    def test_patch_multiple_instances(self):
        ui = load_ui_workflow(WORKFLOWS_DIR / "04_ltx_video_assembly.json")
        patch_ui_workflow(ui, {
            "LoadImage": [
                {"image": "kf0.png"},
                {"image": "kf1.png"},
                {"image": "kf2.png"},
                {"image": "kf3.png"},
            ],
        })
        api = convert_ui_to_api(ui)
        assert api["1"]["inputs"]["image"] == "kf0.png"
        assert api["2"]["inputs"]["image"] == "kf1.png"
        assert api["3"]["inputs"]["image"] == "kf2.png"
        assert api["4"]["inputs"]["image"] == "kf3.png"


class TestFindNodesByType:
    """Tests for find_nodes_by_type()."""

    def test_find_all_loadimage_nodes(self):
        ui = load_ui_workflow(WORKFLOWS_DIR / "04_ltx_video_assembly.json")
        nodes = find_nodes_by_type(ui, "LoadImage")
        assert len(nodes) == 4

    def test_find_ksampler(self):
        ui = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
        nodes = find_nodes_by_type(ui, "KSampler")
        assert len(nodes) == 1
        assert nodes[0]["id"] == 8


class TestNodeSchemasCoverage:
    """Ensure we have schemas for all node types in our workflow files."""

    def test_all_workflow_node_types_have_schemas(self):
        missing = set()
        for wf_path in WORKFLOWS_DIR.glob("*.json"):
            if "_api" in wf_path.name:
                continue
            ui = load_ui_workflow(wf_path)
            for node in ui.get("nodes", []):
                node_type = node.get("type", "")
                if node_type not in NODE_SCHEMAS:
                    missing.add(node_type)

        if missing:
            pytest.fail(f"Missing schemas for node types: {sorted(missing)}")
