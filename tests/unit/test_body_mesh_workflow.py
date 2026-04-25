"""Unit tests for build_body_mesh_workflow() in pipeline/workflows.py."""

import json

import pytest

from pipeline.workflows import build_body_mesh_workflow


class TestBuildBodyMeshWorkflow:
    """Tests for build_body_mesh_workflow()."""

    def test_returns_valid_dict(self):
        wf = build_body_mesh_workflow()
        assert isinstance(wf, dict)
        assert len(wf) > 0

    def test_has_required_nodes(self):
        wf = build_body_mesh_workflow()
        node_types = {v["class_type"] for v in wf.values()}
        assert "BodyMeshRenderer" in node_types
        assert "SaveImage" in node_types

    def test_default_params(self):
        wf = build_body_mesh_workflow()
        renderer = [v for v in wf.values() if v["class_type"] == "BodyMeshRenderer"][0]
        assert renderer["inputs"]["width"] == 1024
        assert renderer["inputs"]["height"] == 1024
        assert renderer["inputs"]["rotations_json"] == "{}"

    def test_custom_params(self):
        rotations = {"right_shoulder": [0, 0, 90]}
        wf = build_body_mesh_workflow(
            rotations_json=json.dumps(rotations),
            width=512,
            height=768,
            filename_prefix="test/mesh",
        )
        renderer = [v for v in wf.values() if v["class_type"] == "BodyMeshRenderer"][0]
        assert renderer["inputs"]["width"] == 512
        assert renderer["inputs"]["height"] == 768
        assert json.loads(renderer["inputs"]["rotations_json"]) == rotations

        save = [v for v in wf.values() if v["class_type"] == "SaveImage"][0]
        assert save["inputs"]["filename_prefix"] == "test/mesh"

    def test_image_link(self):
        """SaveImage should link to BodyMeshRenderer output."""
        wf = build_body_mesh_workflow()
        save = [v for v in wf.values() if v["class_type"] == "SaveImage"][0]
        assert save["inputs"]["images"] == ["1", 0]
