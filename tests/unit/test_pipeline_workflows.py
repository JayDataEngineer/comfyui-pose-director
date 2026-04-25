"""Unit tests for pipeline workflow loading and patching."""

import pytest

from pipeline.ui_to_api import (
    load_ui_workflow,
    patch_ui_workflow,
    patch_widget_value,
    convert_ui_to_api,
    find_nodes_by_type,
    WORKFLOWS_DIR,
)


class TestLoadWorkflow:
    """Tests for loading UI-format workflows."""

    def test_load_character_workflow(self):
        wf = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
        assert "nodes" in wf
        assert "links" in wf
        node_types = {n["type"] for n in wf["nodes"]}
        assert "UNETLoader" in node_types
        assert "CLIPLoader" in node_types
        assert "KSampler" in node_types

    def test_load_motion_director_workflow(self):
        wf = load_ui_workflow(WORKFLOWS_DIR / "01_motion_keyframes_rotation.json")
        node_types = {n["type"] for n in wf["nodes"]}
        assert "MotionDirector" in node_types
        assert "PreviewImage" in node_types
        # No legacy nodes
        assert "OpenposePreprocessor" not in node_types
        assert "LoadImage" not in node_types

    def test_load_vnccs_workflow(self):
        wf = load_ui_workflow(WORKFLOWS_DIR / "02_vnccs_edit_keyframe.json")
        node_types = {n["type"] for n in wf["nodes"]}
        assert "VNCCS_QWEN_Encoder" in node_types
        assert "KSampler" in node_types

    def test_load_ltx_video_workflow(self):
        wf = load_ui_workflow(WORKFLOWS_DIR / "04_ltx_video_assembly.json")
        node_types = {n["type"] for n in wf["nodes"]}
        assert "UnetLoaderGGUF" in node_types
        assert "LTXVAddGuideAdvanced" in node_types
        assert "VHS_VideoCombine" in node_types


class TestPatchWorkflow:
    """Tests for patching widget values."""

    def test_patch_seed(self):
        wf = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
        patch_widget_value(wf, "KSampler", "seed", 999)
        api = convert_ui_to_api(wf)
        ks = [n for n in api.values() if n["class_type"] == "KSampler"][0]
        assert ks["inputs"]["seed"] == 999

    def test_patch_prompt(self):
        wf = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
        patch_widget_value(wf, "CLIPTextEncode", "text", "test prompt")
        api = convert_ui_to_api(wf)
        te = [n for n in api.values() if n["class_type"] == "CLIPTextEncode"][0]
        assert te["inputs"]["text"] == "test prompt"

    def test_patch_multiple_loadimage_nodes(self):
        wf = load_ui_workflow(WORKFLOWS_DIR / "04_ltx_video_assembly.json")
        patch_ui_workflow(wf, {
            "LoadImage": [
                {"image": "kf0.png"},
                {"image": "kf1.png"},
            ],
        })
        api = convert_ui_to_api(wf)
        loadimages = [n for n in api.values() if n["class_type"] == "LoadImage"]
        assert loadimages[0]["inputs"]["image"] == "kf0.png"
        assert loadimages[1]["inputs"]["image"] == "kf1.png"

    def test_patch_motion_director_params(self):
        wf = load_ui_workflow(WORKFLOWS_DIR / "01_motion_keyframes_rotation.json")
        patch_ui_workflow(wf, {
            "MotionDirector": {
                "motion_description": "dance",
                "num_keyframes": 3,
                "total_frames": 24,
            },
        })
        api = convert_ui_to_api(wf)
        md = [n for n in api.values() if n["class_type"] == "MotionDirector"][0]
        assert md["inputs"]["motion_description"] == "dance"
        assert md["inputs"]["num_keyframes"] == 3
        assert md["inputs"]["total_frames"] == 24


class TestConvertWorkflow:
    """Tests for UI-to-API conversion."""

    def test_character_workflow_converts(self):
        wf = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
        api = convert_ui_to_api(wf)
        assert isinstance(api, dict)
        assert len(api) == len(wf["nodes"])
        for node_id, node_def in api.items():
            assert "class_type" in node_def
            assert "inputs" in node_def

    def test_ltx_workflow_has_guide_nodes(self):
        wf = load_ui_workflow(WORKFLOWS_DIR / "04_ltx_video_assembly.json")
        api = convert_ui_to_api(wf)
        guides = [n for n in api.values() if n["class_type"] == "LTXVAddGuideAdvanced"]
        assert len(guides) == 4
        frames = sorted(n["inputs"]["frame_idx"] for n in guides)
        assert frames == [0, 8, 16, 24]

    def test_vnccs_workflow_converts_correctly(self):
        wf = load_ui_workflow(WORKFLOWS_DIR / "02_vnccs_edit_keyframe.json")
        api = convert_ui_to_api(wf)
        vnccs = [n for n in api.values() if n["class_type"] == "VNCCS_QWEN_Encoder"][0]
        assert vnccs["inputs"]["prompt"] == "Draw character from image2"
        assert vnccs["inputs"]["target_size"] == 1024
        assert vnccs["inputs"]["qwen_2511"] is True
        assert vnccs["inputs"]["image1"] == ["1", 0]
        assert vnccs["inputs"]["image2"] == ["2", 0]
