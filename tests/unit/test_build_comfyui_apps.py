"""Tests for scripts/build_comfyui_apps.py"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from build_comfyui_apps import (
    WorkflowApp,
    analyze_workflow,
    determine_category,
    discover_workflows,
    extract_display_name,
    generate_app_json,
    generate_catalog,
    load_workflow,
    slugify,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_workflow(nodes=None, links=None, extra=None):
    """Create a minimal workflow dict for testing."""
    wf = {"nodes": nodes or [], "links": links or [], "extra": extra or {}}
    return wf


def make_node(node_id, node_type, widgets_values=None, inputs=None, properties=None):
    """Create a minimal node dict for testing."""
    node = {
        "id": node_id,
        "type": node_type,
        "widgets_values": widgets_values or [],
        "inputs": inputs or [],
        "properties": properties or {},
    }
    return node


@pytest.fixture
def z_image_workflow(tmp_path):
    """Create a minimal Z-Image character workflow."""
    wf = make_workflow(
        nodes=[
            make_node(1, "UNETLoader", ["z_image_turbo_bf16.safetensors", "bf16"]),
            make_node(2, "CLIPLoader", ["qwen_3_4b.safetensors", "lumina2", "default"]),
            make_node(3, "VAELoader", ["ae.safetensors"]),
            make_node(5, "CLIPTextEncode", ["a cute anime girl", None],
                      [{"name": "text", "type": "STRING", "link": None},
                       {"name": "clip", "type": "CLIP", "link": 2}]),
            make_node(7, "EmptySD3LatentImage", [768, 1024, 1],
                      [{"name": "width", "type": "INT", "link": None},
                       {"name": "height", "type": "INT", "link": None},
                       {"name": "batch_size", "type": "INT", "link": None}]),
            make_node(8, "KSampler",
                      ["model", 42, "fixed", 8, 1.0, "res_multistep", "simple",
                       None, None, None, 1.0],
                      [{"name": "seed", "type": "INT", "link": None},
                       {"name": "steps", "type": "INT", "link": None},
                       {"name": "cfg", "type": "FLOAT", "link": None}]),
            make_node(10, "SaveImage", [None, "PoseDirector/character"],
                      [{"name": "images", "type": "IMAGE", "link": 5}]),
        ],
        links=[[1, 1, 0, 8, 0, "MODEL"]],
    )
    path = tmp_path / "00_z_image_character.json"
    path.write_text(json.dumps(wf))
    return path, wf


# ---------------------------------------------------------------------------
# Tests: Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_discovers_json_files(self, tmp_path):
        wf_dir = tmp_path / "workflows"
        wf_dir.mkdir()
        (wf_dir / "test.json").write_text('{"nodes": [], "links": []}')
        (wf_dir / "test2.json").write_text('{"nodes": [], "links": []}')

        results = discover_workflows([("test-ext", wf_dir)])
        assert len(results) == 2
        assert results[0][1] == "test-ext"

    def test_skips_legacy_dirs(self, tmp_path):
        wf_dir = tmp_path / "workflows"
        wf_dir.mkdir()
        (wf_dir / "good.json").write_text('{}')
        legacy = wf_dir / "legacy"
        legacy.mkdir()
        (legacy / "old.json").write_text('{}')

        results = discover_workflows([("test-ext", wf_dir)])
        assert len(results) == 1
        assert results[0][0].name == "good.json"

    def test_skips_old_workflows(self, tmp_path):
        wf_dir = tmp_path / "workflows"
        wf_dir.mkdir()
        (wf_dir / "current.json").write_text('{}')
        old = wf_dir / "old_workflows"
        old.mkdir()
        (old / "old.json").write_text('{}')

        results = discover_workflows([("test-ext", wf_dir)])
        assert len(results) == 1

    def test_handles_missing_dir(self, tmp_path):
        results = discover_workflows([("ext", tmp_path / "nonexistent")])
        assert results == []


# ---------------------------------------------------------------------------
# Tests: Workflow loading
# ---------------------------------------------------------------------------


class TestLoadWorkflow:
    def test_loads_valid_workflow(self, tmp_path):
        wf = make_workflow()
        path = tmp_path / "test.json"
        path.write_text(json.dumps(wf))
        result = load_workflow(path)
        assert result is not None
        assert "nodes" in result

    def test_returns_none_for_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        assert load_workflow(path) is None

    def test_returns_none_for_api_format(self, tmp_path):
        """API format has numeric string keys, no 'nodes' key."""
        path = tmp_path / "api.json"
        path.write_text('{"1": {"class_type": "KSampler"}}')
        assert load_workflow(path) is None


# ---------------------------------------------------------------------------
# Tests: Analysis
# ---------------------------------------------------------------------------


class TestAnalyzeWorkflow:
    def test_finds_output_nodes(self):
        wf = make_workflow(nodes=[
            make_node(1, "SaveImage", [None, "output/test"]),
            make_node(2, "PreviewImage", [None, "output/preview"]),
        ])
        app = analyze_workflow(wf, Path("test.json"), "test-ext")
        assert len(app.outputs) == 2
        assert "1" in app.outputs
        assert "2" in app.outputs

    def test_exposes_ksampler_inputs(self):
        wf = make_workflow(nodes=[
            make_node(8, "KSampler",
                      ["model", 42, "fixed", 8, 1.0, "res_multistep", "simple",
                       None, None, None, 1.0],
                      [{"name": "seed", "type": "INT", "link": None},
                       {"name": "steps", "type": "INT", "link": None}]),
            make_node(10, "SaveImage", [None, "out"],
                      [{"name": "images", "type": "IMAGE", "link": None}]),
        ])
        app = analyze_workflow(wf, Path("test.json"), "test-ext")
        input_names = [name for _, name in app.inputs]
        assert "seed" in input_names
        assert "steps" in input_names
        assert "cfg" in input_names

    def test_skips_model_loaders(self):
        wf = make_workflow(nodes=[
            make_node(1, "UNETLoader", ["model.safetensors"]),
            make_node(10, "SaveImage", [None, "out"]),
        ])
        app = analyze_workflow(wf, Path("test.json"), "test-ext")
        assert len(app.inputs) == 0  # UNETLoader is skipped

    def test_exposes_text_encode(self):
        wf = make_workflow(nodes=[
            make_node(5, "CLIPTextEncode", ["a beautiful landscape", None],
                      [{"name": "text", "type": "STRING", "link": None},
                       {"name": "clip", "type": "CLIP", "link": 1}]),
            make_node(10, "SaveImage", [None, "out"]),
        ])
        app = analyze_workflow(wf, Path("test.json"), "test-ext")
        assert len(app.inputs) == 1
        assert app.inputs[0] == ("5", "text")

    def test_skips_no_output_workflow(self):
        wf = make_workflow(nodes=[
            make_node(1, "UNETLoader", ["model.safetensors"]),
        ])
        app = analyze_workflow(wf, Path("test.json"), "test-ext")
        assert app.skipped is True
        assert "No output" in app.skip_reason

    def test_handles_empty_nodes(self):
        wf = make_workflow(nodes=[])
        app = analyze_workflow(wf, Path("test.json"), "test-ext")
        assert app.skipped is True

    def test_exposes_custom_nodes(self):
        wf = make_workflow(nodes=[
            make_node(3, "MotionDirector",
                      ["wave hello", 4, 30, "http://localhost", "model", 512, 512, 3, 0]),
            make_node(10, "SaveImage", [None, "out"]),
        ])
        app = analyze_workflow(wf, Path("test.json"), "test-ext")
        input_names = [name for _, name in app.inputs]
        assert "motion_description" in input_names
        assert "num_keyframes" in input_names


# ---------------------------------------------------------------------------
# Tests: Category assignment
# ---------------------------------------------------------------------------


class TestCategoryAssignment:
    def test_pose_director_char_gen(self):
        cat = determine_category(Path("00_z_image_character.json"), "pose-director")
        assert cat == "Character Generation"

    def test_pose_director_video(self):
        cat = determine_category(Path("04_ltx_video_assembly.json"), "pose-director")
        assert cat == "Video"

    def test_vnccs_step1(self):
        cat = determine_category(Path("VN_Step1_QWEN_CharSheetGenerator_v1.json"), "vnccs")
        assert cat == "Character Generation"

    def test_vnccs_step2(self):
        cat = determine_category(Path("VN_Step2_QWEN_ClothesGenerator_v1.json"), "vnccs")
        assert cat == "Character Editing"

    def test_hy_motion(self):
        cat = determine_category(Path("workflow.json"), "hy-motion")
        assert cat == "Pose & Motion"

    def test_vnccs_utils(self):
        cat = determine_category(Path("VNCCS_Utils Pose Studio.json"), "vnccs-utils")
        assert cat == "Pose & Motion"


# ---------------------------------------------------------------------------
# Tests: Display name extraction
# ---------------------------------------------------------------------------


class TestDisplayName:
    def test_strips_vn_prefix(self):
        name = extract_display_name(
            Path("VN_Step1_QWEN_CharSheetGenerator_v1.json"), "vnccs")
        assert "Step1" not in name
        assert "CharSheetGenerator" in name

    def test_strips_workflow_prefix(self):
        name = extract_display_name(Path("workflow.json"), "hy-motion")
        assert "workflow" in name.lower()

    def test_strips_vnccs_utils_prefix(self):
        name = extract_display_name(
            Path("VNCCS_Utils Pose Studio.json"), "vnccs-utils")
        assert "Pose Studio" in name


# ---------------------------------------------------------------------------
# Tests: App JSON generation
# ---------------------------------------------------------------------------


class TestGenerateAppJson:
    def test_adds_linear_mode(self):
        wf = make_workflow()
        app = WorkflowApp(
            source_path=Path("test.json"),
            extension="test",
            category="Test",
            display_name="Test App",
            inputs=[("1", "text")],
            outputs=["2"],
        )
        result = generate_app_json(wf, app)
        assert result["extra"]["linearMode"] is True

    def test_sets_inputs_and_outputs(self):
        wf = make_workflow()
        app = WorkflowApp(
            source_path=Path("test.json"),
            extension="test",
            category="Test",
            display_name="Test App",
            inputs=[("5", "text"), ("8", "seed")],
            outputs=["10"],
        )
        result = generate_app_json(wf, app)
        ld = result["extra"]["linearData"]
        assert ld["inputs"] == [["5", "text"], ["8", "seed"]]
        assert ld["outputs"] == ["10"]

    def test_preserves_existing_extra(self):
        wf = make_workflow(extra={"existing": "data"})
        app = WorkflowApp(
            source_path=Path("test.json"),
            extension="test",
            category="Test",
            display_name="Test App",
            inputs=[],
            outputs=["1"],
        )
        result = generate_app_json(wf, app)
        assert result["extra"]["existing"] == "data"

    def test_does_not_mutate_original(self):
        wf = make_workflow()
        original_extra = dict(wf.get("extra", {}))
        app = WorkflowApp(
            source_path=Path("test.json"),
            extension="test",
            category="Test",
            display_name="Test App",
            inputs=[],
            outputs=["1"],
        )
        generate_app_json(wf, app)
        assert wf.get("extra", {}) == original_extra


# ---------------------------------------------------------------------------
# Tests: Slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello_world"

    def test_special_chars(self):
        assert slugify("VN_Step1 (v2.0)") == "vn_step1_v20"

    def test_length_limit(self):
        result = slugify("a" * 100)
        assert len(result) <= 60


# ---------------------------------------------------------------------------
# Tests: Catalog
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_generates_markdown(self):
        apps = [
            WorkflowApp(
                source_path=Path("test.json"),
                extension="test-ext",
                category="Test",
                display_name="Test App",
                inputs=[("1", "text")],
                outputs=["2"],
                node_count=5,
                node_types=["KSampler", "SaveImage"],
            ),
        ]
        catalog = generate_catalog(apps)
        assert "# ComfyUI Workflow Catalog" in catalog
        assert "Test App" in catalog
        assert "Test" in catalog

    def test_groups_by_category(self):
        apps = [
            WorkflowApp(Path("a.json"), "ext", "Alpha", "A", [], ["1"], 1, []),
            WorkflowApp(Path("b.json"), "ext", "Beta", "B", [], ["2"], 1, []),
            WorkflowApp(Path("c.json"), "ext", "Alpha", "C", [], ["3"], 1, []),
        ]
        catalog = generate_catalog(apps)
        assert "## Alpha" in catalog
        assert "## Beta" in catalog


# ---------------------------------------------------------------------------
# Tests: End-to-end with real workflow
# ---------------------------------------------------------------------------


class TestE2E:
    def test_z_image_workflow(self, z_image_workflow):
        path, wf = z_image_workflow
        app = analyze_workflow(wf, path, "pose-director")

        assert app.skipped is False
        assert len(app.outputs) == 1
        # Should expose text prompt and sampling params
        input_names = [name for _, name in app.inputs]
        assert "text" in input_names
        assert "seed" in input_names

    def test_generate_valid_app_json(self, z_image_workflow):
        path, wf = z_image_workflow
        app = analyze_workflow(wf, path, "pose-director")
        result = generate_app_json(wf, app)

        # Verify it's valid JSON
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert parsed["extra"]["linearMode"] is True
        assert isinstance(parsed["extra"]["linearData"]["inputs"], list)
        assert isinstance(parsed["extra"]["linearData"]["outputs"], list)
