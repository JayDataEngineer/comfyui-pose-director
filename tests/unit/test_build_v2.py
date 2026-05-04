"""Unit tests for the build_v2 system — DAG, validators, state, stages."""

import json
import tempfile
from pathlib import Path

import pytest

from build_v2.context import BuildContext
from build_v2.dag import build_adjacency, topological_sort
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


class TestDAG:
    def test_simple_chain(self):
        """A -> B -> C should sort to [A, B, C]."""

        class A:
            name = "a"
            depends_on = []

        class B:
            name = "b"
            depends_on = ["a"]

        class C:
            name = "c"
            depends_on = ["b"]

        graph = build_adjacency([A(), B(), C()])
        order = topological_sort(graph)
        assert order.index("a") < order.index("b") < order.index("c")

    def test_diamond_dependency(self):
        """A -> B, A -> C, B -> D, C -> D."""

        class A:
            name = "a"
            depends_on = []

        class B:
            name = "b"
            depends_on = ["a"]

        class C:
            name = "c"
            depends_on = ["a"]

        class D:
            name = "d"
            depends_on = ["b", "c"]

        graph = build_adjacency([A(), B(), C(), D()])
        order = topological_sort(graph)
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_detection(self):
        class A:
            name = "a"
            depends_on = ["b"]

        class B:
            name = "b"
            depends_on = ["a"]

        graph = build_adjacency([A(), B()])
        with pytest.raises(ValueError, match="Circular"):
            topological_sort(graph)

    def test_no_deps(self):
        class A:
            name = "a"
            depends_on = []

        graph = build_adjacency([A()])
        order = topological_sort(graph)
        assert order == ["a"]


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


class TestGenerateStage:
    def test_import_complete(self):
        """Import stage should detect existing approved.png."""
        from build_v2.stages.generate import GenerateStage

        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(
                character_data={"generation": {"source": "import"}},
                character_name="test",
                output_dir=Path(tmp),
            )
            stage = GenerateStage(ctx)

            # Not complete initially
            assert stage.is_complete({}) is False

            # Create approved image
            ctx.design_dir.mkdir(parents=True, exist_ok=True)
            ctx.approved_image.write_bytes(b"fake png")
            assert stage.is_complete({}) is True


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
        assert stages == {"generate"}  # Only identity, no sheet/emotions/sprites

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


class TestAllStages:
    """Verify all 11 stages instantiate with correct tier and dependencies."""

    def test_stage_count(self):
        from build_v2.orchestrator import ALL_STAGE_CLASSES
        assert len(ALL_STAGE_CLASSES) == 11

    def test_generate_stage(self):
        from build_v2.stages.generate import GenerateStage
        s = GenerateStage(None)
        assert s.name == "generate"
        assert s.tier_required == "t0"
        assert s.depends_on == []

    def test_sheet_stage(self):
        from build_v2.stages.sheet import SheetStage
        s = SheetStage(None)
        assert s.name == "sheet"
        assert s.tier_required == "t1"
        assert "generate" in s.depends_on

    def test_emotions_stage(self):
        from build_v2.stages.emotions import EmotionsStage
        s = EmotionsStage(None)
        assert s.name == "emotions"
        assert s.tier_required == "t1"
        assert "sheet" in s.depends_on

    def test_sprites_static_stage(self):
        from build_v2.stages.sprites_static import SpritesStaticStage
        s = SpritesStaticStage(None)
        assert s.name == "sprites_static"
        assert s.tier_required == "t1"
        assert "sheet" in s.depends_on

    def test_sprites_animated_stage(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        s = SpritesAnimatedStage(None)
        assert s.name == "sprites_animated"
        assert s.tier_required == "t1.5"
        assert "generate" in s.depends_on

    def test_outfit_stage(self):
        from build_v2.stages.outfit import OutfitStage
        s = OutfitStage(None)
        assert s.name == "outfit"
        assert s.tier_required == "t2"
        assert "sheet" in s.depends_on

    def test_lora_stage(self):
        from build_v2.stages.lora import LoraStage
        s = LoraStage(None)
        assert s.name == "lora"
        assert s.tier_required == "t3"
        assert "sprites_static" in s.depends_on
        assert "sprites_animated" in s.depends_on

    def test_placeholder_stages(self):
        from build_v2.stages.placeholders import (
            TrellisStage, StateStage, VideoStage, GodotStage,
        )
        assert TrellisStage(None).tier_required == "t2"
        assert StateStage(None).tier_required == "t1"
        assert VideoStage(None).tier_required == "t1.5"
        assert GodotStage(None).tier_required == "t1"

    def test_stage_names_unique(self):
        from build_v2.orchestrator import ALL_STAGE_CLASSES
        names = [cls.name for cls in ALL_STAGE_CLASSES]
        assert len(names) == len(set(names)), f"Duplicate stage names: {names}"

    def test_dag_valid(self):
        """Verify DAG has no cycles with all stages."""
        from build_v2.orchestrator import ALL_STAGE_CLASSES
        from build_v2.dag import build_adjacency, topological_sort
        stages = [cls(None) for cls in ALL_STAGE_CLASSES]
        graph = build_adjacency(stages)
        assert len(graph) == len(stages)
        order = topological_sort(graph)
        assert len(order) == len(stages)
