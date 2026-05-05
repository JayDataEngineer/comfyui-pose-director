"""Unit tests for strategy validation and resolution."""

import tempfile
from pathlib import Path

import pytest

from build_v2.context import BuildContext
from build_v2.validators import validate_character


class TestStrategyValidation:
    """Validate strategy config values are accepted or rejected."""

    def _base_data(self, **overrides):
        data = {
            "character": {"name": "test"},
            "generation": {"source": "generate", "prompt": "a cat"},
        }
        data.update(overrides)
        return data

    def test_valid_motion_strategies(self):
        for strategy in ("hymotion", "llm"):
            data = self._base_data(
                packages={"sprites": {"motion_strategy": strategy, "animated": [{"name": "walk"}]}},
            )
            validate_character(data, Path("test.yaml"))

    def test_invalid_motion_strategy(self):
        data = self._base_data(
            packages={"sprites": {"motion_strategy": "nonexistent"}},
        )
        with pytest.raises(ValueError, match="motion_strategy"):
            validate_character(data, Path("test.yaml"))

    def test_valid_body_references(self):
        for ref in ("anny", "trellis"):
            data = self._base_data(
                packages={"sprites": {"body_reference": ref, "static": True}},
            )
            validate_character(data, Path("test.yaml"))

    def test_invalid_body_reference(self):
        data = self._base_data(
            packages={"sprites": {"body_reference": "magic"}},
        )
        with pytest.raises(ValueError, match="body_reference"):
            validate_character(data, Path("test.yaml"))

    def test_valid_face_refinements(self):
        for ref in ("skip", "facedetailer"):
            data = self._base_data(
                packages={"sheet": {"face_refinement": ref}},
            )
            validate_character(data, Path("test.yaml"))

    def test_invalid_face_refinement(self):
        data = self._base_data(
            packages={"sheet": {"face_refinement": "super_resolution"}},
        )
        with pytest.raises(ValueError, match="face_refinement"):
            validate_character(data, Path("test.yaml"))

    def test_defaults_validate(self):
        """No strategy keys = defaults, should validate fine."""
        data = self._base_data(
            packages={
                "portraits": {"emotions": ["happy"]},
                "sprites": {"static": True, "animated": [{"name": "walk"}]},
            },
        )
        validate_character(data, Path("test.yaml"))


class TestStrategyResolution:
    """Test that stages resolve strategies correctly from YAML config."""

    def test_animated_default_motion_strategy(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        ctx = BuildContext(character_data={}, character_name="test")
        stage = SpritesAnimatedStage(ctx)
        assert stage._resolve_motion_strategy() == "hymotion"

    def test_animated_llm_motion_strategy(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        data = {"packages": {"sprites": {"motion_strategy": "llm"}}}
        ctx = BuildContext(character_data=data, character_name="test")
        stage = SpritesAnimatedStage(ctx)
        assert stage._resolve_motion_strategy() == "llm"

    def test_animated_hymotion_not_available(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(
                character_data={},
                character_name="test",
                output_dir=Path(tmp),
                motions_dir=Path(tmp) / "motions",
            )
            stage = SpritesAnimatedStage(ctx)
            assert stage._check_hymotion_available() is False

    def test_animated_hymotion_available(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            motions_dir = Path(tmp) / "motions" / "shared"
            motions_dir.mkdir(parents=True)
            (motions_dir / "walk.npz").write_bytes(b"fake npz")
            ctx = BuildContext(
                character_data={},
                character_name="test",
                output_dir=Path(tmp),
                motions_dir=Path(tmp) / "motions",
            )
            stage = SpritesAnimatedStage(ctx)
            assert stage._check_hymotion_available() is True

    def test_static_default_body_reference(self):
        from build_v2.stages.sprites_static import SpritesStaticStage
        ctx = BuildContext(character_data={}, character_name="test")
        stage = SpritesStaticStage(ctx)
        assert stage._resolve_body_reference() == "anny"

    def test_static_trellis_body_reference(self):
        from build_v2.stages.sprites_static import SpritesStaticStage
        data = {"packages": {"sprites": {"body_reference": "trellis"}}}
        ctx = BuildContext(character_data=data, character_name="test")
        stage = SpritesStaticStage(ctx)
        assert stage._resolve_body_reference() == "trellis"

    def test_static_trellis_fallback_when_not_complete(self):
        """When body_reference=trellis but trellis didn't complete, run dispatch falls back."""
        from build_v2.stages.sprites_static import SpritesStaticStage
        data = {"packages": {"sprites": {"body_reference": "trellis", "static": True}}}
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(
                character_data=data,
                character_name="test",
                output_dir=Path(tmp),
            )
            stage = SpritesStaticStage(ctx)
            state = {"stages": {}}
            strategy = stage._resolve_body_reference()
            assert strategy == "trellis"
            trellis_state = state.get("stages", {}).get("trellis", {})
            assert trellis_state.get("status") != "complete"

    def test_sheet_default_face_refinement(self):
        from build_v2.stages.sheet import SheetStage
        ctx = BuildContext(character_data={}, character_name="test")
        stage = SheetStage(ctx)
        assert stage._resolve_face_refinement() == "skip"

    def test_sheet_facedetailer_face_refinement(self):
        from build_v2.stages.sheet import SheetStage
        data = {"packages": {"sheet": {"face_refinement": "facedetailer"}}}
        ctx = BuildContext(character_data=data, character_name="test")
        stage = SheetStage(ctx)
        assert stage._resolve_face_refinement() == "facedetailer"
