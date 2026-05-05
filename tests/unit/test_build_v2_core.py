"""Unit tests for validators, state management, and BuildContext."""

import tempfile
from pathlib import Path

import pytest

from build_v2.context import BuildContext
from build_v2.state import (
    compute_yaml_hash,
    file_hash,
    load_build_state,
    save_build_state,
    stage_needs_rebuild,
    update_stage_state,
)
from build_v2.validators import (
    get_enabled_stages,
    infer_tier,
    tier_gte,
    validate_character,
)


# ── Validators ──────────────────────────────────────────────────────────────


class TestValidators:
    def test_infer_tier_empty(self):
        assert infer_tier({}) == "t0"
        assert infer_tier({"generation": {"source": "import"}}) == "t0"

    def test_infer_tier_sheet(self):
        assert infer_tier({"sheet": {}}) == "t1"

    def test_infer_tier_animated(self):
        assert infer_tier({"sprites": {"animated": [{"name": "walk"}]}}) == "t1.5"

    def test_infer_tier_trellis(self):
        assert infer_tier({"trellis": True}) == "t2"

    def test_infer_tier_max(self):
        data = {"sheet": {}, "sprites": {"animated": [{"name": "walk"}]}, "trellis": True}
        assert infer_tier(data) == "t2"

    def test_tier_gte(self):
        assert tier_gte("t1", "t0")
        assert tier_gte("t1", "t1")
        assert tier_gte("t1.5", "t1")
        assert not tier_gte("t1", "t1.5")
        assert not tier_gte("t0", "t1")

    def test_enabled_stages_generate_only(self):
        assert get_enabled_stages({"generation": {"source": "import"}}, "t0") == {"generate"}

    def test_enabled_stages_t1(self):
        stages = get_enabled_stages({"sheet": {}, "emotions": {"list": ["happy"]}}, "t1")
        assert "generate" in stages
        assert "sheet" in stages
        assert "emotions" in stages

    def test_enabled_stages_tier_filter(self):
        """T1 character should not get T1.5 stages."""
        data = {"sprites": {"animated": [{"name": "walk"}]}}
        stages = get_enabled_stages(data, "t1")
        assert "sprites_animated" not in stages

    def test_validate_missing_name(self):
        with pytest.raises(ValueError, match="character.name"):
            validate_character({"character": {}}, Path("test.yaml"))

    def test_validate_import_no_image_ok(self):
        """Import without image is allowed (warning only)."""
        validate_character(
            {"character": {"name": "test"}, "generation": {"source": "import"}},
            Path("test.yaml"),
        )

    def test_validate_bad_direction(self):
        with pytest.raises(ValueError, match="invalid directions"):
            validate_character(
                {
                    "character": {"name": "test"},
                    "generation": {"source": "generate", "prompt": "a cat"},
                    "sprites": {"directions": ["upside_down"]},
                },
                Path("test.yaml"),
            )


class TestTierFromYAML:
    """Verify tier inference and stage enablement from real YAML structure."""

    def test_shopkeeper_tier_and_stages(self):
        import yaml
        data = yaml.safe_load(Path("characters/shopkeeper.yaml").read_text())
        assert infer_tier(data) >= "t1"
        stages = get_enabled_stages(data, data["character"]["tier"])
        assert "generate" in stages
        assert "sheet" in stages
        assert "emotions" in stages
        assert "sprites_static" in stages
        assert "sprites_animated" in stages

    def test_marty_tier_and_stages(self):
        import yaml
        data = yaml.safe_load(Path("characters/marty.yaml").read_text())
        tier = data["character"]["tier"]
        assert tier == "t2"
        stages = get_enabled_stages(data, tier)
        assert "generate" in stages
        assert "sheet" in stages
        assert "emotions" in stages
        assert "sprites_static" in stages
        assert "sprites_animated" in stages
        assert "outfit" in stages
        assert "state" in stages

    def test_shadow_beast_tier_and_stages(self):
        import yaml
        data = yaml.safe_load(Path("characters/shadow_beast.yaml").read_text())
        stages = get_enabled_stages(data, data["character"]["tier"])
        assert "generate" in stages
        assert "sheet" in stages
        assert "sprites_animated" in stages

    def test_import_tier_t0(self):
        data = {
            "character": {"name": "test", "tier": "t0"},
            "generation": {"source": "import", "image": "ref.png"},
        }
        assert infer_tier(data) == "t0"
        stages = get_enabled_stages(data, "t0")
        assert stages == {"generate"}

    def test_generate_only_tier_t0(self):
        data = {
            "character": {"name": "test", "tier": "t0"},
            "generation": {"source": "generate", "prompt": "a cat", "quality": "turbo"},
        }
        stages = get_enabled_stages(data, "t0")
        assert stages == {"generate"}

    def test_t1_with_portraits(self):
        data = {
            "character": {"name": "test", "tier": "t1"},
            "generation": {"source": "generate", "prompt": "a cat"},
            "packages": {
                "portraits": {"emotions": ["neutral", "happy"]},
            },
        }
        assert infer_tier(data) == "t1"
        stages = get_enabled_stages(data, "t1")
        assert "generate" in stages
        assert "sheet" in stages
        assert "emotions" in stages
        assert "sprites_animated" not in stages

    def test_t3_lora(self):
        data = {
            "character": {"name": "test", "tier": "t3"},
            "generation": {"source": "generate", "prompt": "a cat"},
            "packages": {
                "lora": {"enabled": True},
            },
        }
        assert infer_tier(data) == "t3"
        stages = get_enabled_stages(data, "t3")
        assert "lora" in stages


# ── State management ────────────────────────────────────────────────────────


class TestState:
    def test_yaml_hash_deterministic(self):
        data = {"character": {"name": "test"}, "generation": {"source": "import"}}
        h1 = compute_yaml_hash(data)
        h2 = compute_yaml_hash(data)
        assert h1 == h2

    def test_yaml_hash_changes(self):
        d1 = {"character": {"name": "a"}}
        d2 = {"character": {"name": "b"}}
        assert compute_yaml_hash(d1) != compute_yaml_hash(d2)

    def test_yaml_hash_ignores_internal(self):
        d1 = {"character": {"name": "test"}}
        d2 = {"character": {"name": "test"}, "_yaml_path": "/tmp/test.yaml"}
        assert compute_yaml_hash(d1) == compute_yaml_hash(d2)

    def test_load_save_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            state = load_build_state(outdir)
            assert state["stages"] == {}

            update_stage_state(state, "generate", "complete", artifacts=["design/approved.png"])
            save_build_state(outdir, state)

            loaded = load_build_state(outdir)
            assert loaded["stages"]["generate"]["status"] == "complete"
            assert loaded["stages"]["generate"]["artifacts"] == ["design/approved.png"]

    def test_needs_rebuild_new_stage(self):
        state = {"stages": {}}
        assert stage_needs_rebuild(state, "generate", "abc123", []) is True

    def test_needs_rebuild_complete(self):
        state = {"stages": {"generate": {"status": "complete", "input_hash": "abc123"}}}
        assert stage_needs_rebuild(state, "generate", "abc123", []) is False

    def test_needs_rebuild_hash_changed(self):
        state = {"stages": {"generate": {"status": "complete", "input_hash": "old"}}}
        assert stage_needs_rebuild(state, "generate", "new", []) is True

    def test_needs_rebuild_forced(self):
        state = {"stages": {"generate": {"status": "complete", "input_hash": "abc123"}}}
        assert stage_needs_rebuild(state, "generate", "abc123", ["generate"]) is True
        assert stage_needs_rebuild(state, "generate", "abc123", ["all"]) is True


# ── BuildContext ─────────────────────────────────────────────────────────────


class TestBuildContext:
    def test_from_yaml(self):
        data = {
            "character": {"name": "test", "tier": "t1"},
            "generation": {"source": "import", "seed": 99},
        }
        ctx = BuildContext.from_yaml(
            data=data,
            yaml_path="/tmp/test.yaml",
            project_root=Path("/project"),
            comfyui_url="http://localhost:8465",
        )
        assert ctx.character_name == "test"
        assert ctx.tier == "t1"
        assert ctx.seed == 99
        assert ctx.output_dir == Path("/project/output/test")
        assert ctx.approved_image == Path("/project/output/test/design/approved.png")

    def test_from_yaml_seed_override(self):
        data = {
            "character": {"name": "test"},
            "generation": {"source": "import", "seed": 99},
        }
        ctx = BuildContext.from_yaml(
            data=data, yaml_path="/tmp/test.yaml",
            project_root=Path("/project"), seed_override=42,
        )
        assert ctx.seed == 42

    def test_pkg_from_packages(self):
        data = {"packages": {"sprites": {"outfit": "armor"}}}
        ctx = BuildContext(character_data=data, character_name="test")
        assert ctx.pkg("sprites") == {"outfit": "armor"}

    def test_pkg_fallback_to_top_level(self):
        data = {"sheet": {"face_refinement": "facedetailer"}}
        ctx = BuildContext(character_data=data, character_name="test")
        assert ctx.pkg("sheet") == {"face_refinement": "facedetailer"}

    def test_pkg_missing_returns_empty_dict(self):
        ctx = BuildContext(character_data={}, character_name="test")
        assert ctx.pkg("nonexistent") == {}
