"""Unit tests for build_character.py — YAML parsing, validation, and build planning."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from scripts.build_character import (
    validate_character,
    compute_yaml_hash,
    make_plan,
    load_character_yaml,
    BuildPlan,
    VALID_EMOTIONS,
    VALID_DIRECTIONS,
    TIER_DEFAULTS,
)

YAML_DIR = Path(__file__).parent.parent.parent / "characters"


class TestYAMLValidation:
    def test_shopkeeper_valid(self):
        data = load_character_yaml(YAML_DIR / "shopkeeper.yaml")
        assert data["character"]["name"] == "shopkeeper"
        assert data["generation"]["source"] == "generate"
        assert data["generation"]["quality"] == "turbo"

    def test_marty_valid(self):
        data = load_character_yaml(YAML_DIR / "marty.yaml")
        assert data["character"]["name"] == "marty"
        assert data["character"]["tier"] == "t2"
        assert len(data["packages"]["portraits"]["emotions"]) == 10
        sprites_anim = data["packages"]["sprites"].get("animated") or data["packages"]["sprites"].get("animations", [])
        assert len(sprites_anim) == 4

    def test_shadow_beast_valid(self):
        data = load_character_yaml(YAML_DIR / "shadow_beast.yaml")
        assert data["character"]["name"] == "shadow_beast"
        assert data["generation"]["source"] == "import"

    def test_missing_name_fails(self):
        import yaml
        data = yaml.safe_load("character:\n  description: test")
        with pytest.raises(SystemExit):
            validate_character(data, Path("test.yaml"))

    def test_generate_without_prompt_fails(self):
        import yaml
        data = yaml.safe_load("character:\n  name: test\ngeneration:\n  source: generate")
        with pytest.raises(SystemExit):
            validate_character(data, Path("test.yaml"))

    def test_invalid_direction_fails(self):
        import yaml
        data = yaml.safe_load("""
character:
  name: test
generation:
  source: generate
  prompt: "test prompt"
  quality: turbo
packages:
  sprites:
    directions: [front, upside_down]
    animations:
      - name: idle
        motion: shared
""")
        with pytest.raises(SystemExit):
            validate_character(data, Path("test.yaml"))


class TestBuildPlan:
    def _make_data(self, **overrides):
        base = {
            "character": {"name": "test_char", "tier": "t1"},
            "generation": {"source": "generate", "prompt": "test", "quality": "turbo", "seed": 42},
            "packages": {},
        }
        base.update(overrides)
        return base

    def test_empty_state_full_build(self):
        data = self._make_data()
        state = {}
        plan = make_plan(data, state)
        assert len(plan) == 1  # Just generation
        assert plan.steps[0]["stage"] == "generation"

    def test_generation_complete_skipped(self):
        data = self._make_data()
        state = {
            "yaml_hash": compute_yaml_hash(data),
            "generation": {"status": "complete", "source": "generate"},
        }
        plan = make_plan(data, state)
        assert len(plan) == 0  # Nothing to build

    def test_portraits_added(self):
        data = self._make_data(packages={
            "portraits": {"emotions": ["neutral", "happy"]},
        })
        state = {
            "yaml_hash": compute_yaml_hash(data),
            "generation": {"status": "complete"},
        }
        plan = make_plan(data, state)
        stages = [s["stage"] for s in plan.steps]
        assert "portraits" in stages
        portrait_steps = [s for s in plan.steps if s["stage"] == "portraits"]
        assert len(portrait_steps) == 2

    def test_sprites_added(self):
        data = self._make_data(packages={
            "sprites": {
                "outfit": "default",
                "directions": ["front"],
                "animations": [
                    {"name": "idle", "motion": "shared"},
                    {"name": "walk", "motion": "shared"},
                ],
            },
        })
        state = {
            "yaml_hash": compute_yaml_hash(data),
            "generation": {"status": "complete"},
        }
        plan = make_plan(data, state)
        sprite_steps = [s for s in plan.steps if s["stage"] == "sprites"]
        assert len(sprite_steps) == 2

    def test_incremental_add_animation(self):
        """Adding a new animation only generates that animation."""
        data = self._make_data(packages={
            "sprites": {
                "outfit": "default",
                "directions": ["front"],
                "animations": [
                    {"name": "idle", "motion": "shared"},
                    {"name": "walk", "motion": "shared"},
                ],
            },
        })
        state = {
            "yaml_hash": "old",
            "generation": {"status": "complete"},
            "sprites": {
                "default/normal/idle": {"status": "complete", "motion_source": "shared"},
            },
        }
        plan = make_plan(data, state)
        sprite_steps = [s for s in plan.steps if s["stage"] == "sprites"]
        assert len(sprite_steps) == 1
        assert sprite_steps[0]["details"]["animation"] == "walk"

    def test_outfit_variant(self):
        data = self._make_data(packages={
            "sprites": {
                "outfit": "casual",
                "directions": ["front"],
                "animations": [{"name": "idle", "motion": "shared"}],
            },
            "outfit": {
                "name": "armor",
                "description": "metal armor",
            },
        })
        state = {
            "yaml_hash": compute_yaml_hash(data),
            "generation": {"status": "complete"},
            "sprites": {
                "casual/normal/idle": {"status": "complete"},
            },
        }
        plan = make_plan(data, state)
        outfit_steps = [s for s in plan.steps if s["stage"] == "outfit"]
        assert len(outfit_steps) == 1
        assert outfit_steps[0]["details"]["outfit"] == "armor"

    def test_state_variant(self):
        data = self._make_data(packages={
            "portraits": {"emotions": ["neutral"]},
            "sprites": {
                "outfit": "default",
                "directions": ["front"],
                "animations": [{"name": "idle", "motion": "shared"}],
            },
            "state": {
                "name": "beatup",
                "description": "bruised",
                "apply_to": [
                    "portraits",
                    {"animations": ["idle"]},
                ],
            },
        })
        state = {
            "yaml_hash": compute_yaml_hash(data),
            "generation": {"status": "complete"},
            "portraits": {"default/neutral": {"status": "complete"}},
            "sprites": {"default/normal/idle": {"status": "complete"}},
        }
        plan = make_plan(data, state)
        state_steps = [s for s in plan.steps if s["stage"] == "state"]
        assert len(state_steps) == 2  # 1 portrait + 1 animation

    def test_rebuild_flag(self):
        data = self._make_data(packages={
            "sprites": {
                "outfit": "casual",
                "directions": ["front"],
                "animations": [
                    {"name": "idle", "motion": "shared"},
                    {"name": "walk", "motion": "shared"},
                ],
            },
        })
        state = {
            "yaml_hash": compute_yaml_hash(data),
            "generation": {"status": "complete"},
            "sprites": {
                "casual/normal/idle": {"status": "complete"},
                "casual/normal/walk": {"status": "complete"},
            },
        }
        # Force rebuild just idle
        plan = make_plan(data, state, rebuild_targets=["sprites.casual.normal.idle"])
        sprite_steps = [s for s in plan.steps if s["stage"] == "sprites"]
        assert len(sprite_steps) == 1
        assert sprite_steps[0]["details"]["animation"] == "idle"

    def test_tier_defaults_t2(self):
        data = self._make_data(
            character={"name": "test", "tier": "t2"},
            packages={
                "sprites": {
                    "outfit": "default",
                    "animations": [{"name": "idle", "motion": "custom", "prompt": "test"}],
                },
            },
        )
        state = {
            "yaml_hash": compute_yaml_hash(data),
            "generation": {"status": "complete"},
        }
        plan = make_plan(data, state)
        sprite_steps = [s for s in plan.steps if s["stage"] == "sprites"]
        assert len(sprite_steps) == 1
        assert sprite_steps[0]["details"]["directions"] == ["front", "back", "left", "right"]
        assert sprite_steps[0]["details"]["body_source"] == "trellis"

    def test_motion_prompt_change_triggers_rebuild(self):
        data = self._make_data(packages={
            "sprites": {
                "outfit": "default",
                "directions": ["front"],
                "animations": [
                    {"name": "idle", "motion": "custom", "prompt": "new prompt"},
                ],
            },
        })
        state = {
            "yaml_hash": "old",
            "generation": {"status": "complete"},
            "sprites": {
                "default/normal/idle": {
                    "status": "complete",
                    "motion_source": "custom",
                    "motion_prompt": "old prompt",
                },
            },
        }
        plan = make_plan(data, state)
        sprite_steps = [s for s in plan.steps if s["stage"] == "sprites"]
        assert len(sprite_steps) == 1


class TestBuildPlanSummary:
    def test_summary(self):
        plan = BuildPlan()
        plan.add("generation", "generate", {})
        plan.add("portraits", "generate", {})
        plan.add("portraits", "generate", {})
        plan.add("sprites", "generate", {})
        assert plan.summary() == "1 generation, 2 portraits, 1 sprites"
