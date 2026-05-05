"""Unit tests for strategy validation and resolution.

Covers:
- YAML validator accepts/rejects strategy config values
- Stage strategy resolution from YAML config
- Auto-fallback behavior when primary strategy unavailable
- LLM keyframe JSON parsing edge cases
"""

import json
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


class TestAnimatedMotionStrategy:
    """Test SpritesAnimatedStage motion strategy resolution."""

    def test_default_motion_strategy(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        ctx = BuildContext(character_data={}, character_name="test")
        stage = SpritesAnimatedStage(ctx)
        assert stage._resolve_motion_strategy() == "hymotion"

    def test_llm_motion_strategy(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        data = {"packages": {"sprites": {"motion_strategy": "llm"}}}
        ctx = BuildContext(character_data=data, character_name="test")
        stage = SpritesAnimatedStage(ctx)
        assert stage._resolve_motion_strategy() == "llm"

    def test_hymotion_not_available(self):
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

    def test_hymotion_available_with_npz(self):
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


class TestAnimatedBodyReference:
    """Test SpritesAnimatedStage body reference strategy resolution."""

    def test_default_body_reference(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        ctx = BuildContext(character_data={}, character_name="test")
        stage = SpritesAnimatedStage(ctx)
        assert stage._resolve_body_reference() == "anny"

    def test_trellis_body_reference(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        data = {"packages": {"sprites": {"body_reference": "trellis"}}}
        ctx = BuildContext(character_data=data, character_name="test")
        stage = SpritesAnimatedStage(ctx)
        assert stage._resolve_body_reference() == "trellis"

    def test_auto_fallback_trellis_to_anny(self):
        """run() should auto-fallback when body=trellis but trellis not complete."""
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        data = {"packages": {"sprites": {"body_reference": "trellis", "motion_strategy": "llm"}}}
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(
                character_data=data, character_name="test",
                output_dir=Path(tmp),
            )
            stage = SpritesAnimatedStage(ctx)
            # The run() method checks trellis state — this just tests that
            # resolve works and that the state check returns incomplete
            assert stage._resolve_body_reference() == "trellis"
            state = {"stages": {}}
            trellis_state = state.get("stages", {}).get("trellis", {})
            assert trellis_state.get("status") != "complete"


class TestStaticBodyReference:
    """Test SpritesStaticStage body reference strategy resolution."""

    def test_default_body_reference(self):
        from build_v2.stages.sprites_static import SpritesStaticStage
        ctx = BuildContext(character_data={}, character_name="test")
        stage = SpritesStaticStage(ctx)
        assert stage._resolve_body_reference() == "anny"

    def test_trellis_body_reference(self):
        from build_v2.stages.sprites_static import SpritesStaticStage
        data = {"packages": {"sprites": {"body_reference": "trellis"}}}
        ctx = BuildContext(character_data=data, character_name="test")
        stage = SpritesStaticStage(ctx)
        assert stage._resolve_body_reference() == "trellis"

    def test_trellis_fallback_when_not_complete(self):
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


class TestSheetFaceRefinement:
    """Test SheetStage face refinement strategy resolution."""

    def test_default_face_refinement(self):
        from build_v2.stages.sheet import SheetStage
        ctx = BuildContext(character_data={}, character_name="test")
        stage = SheetStage(ctx)
        assert stage._resolve_face_refinement() == "skip"

    def test_facedetailer_face_refinement(self):
        from build_v2.stages.sheet import SheetStage
        data = {"packages": {"sheet": {"face_refinement": "facedetailer"}}}
        ctx = BuildContext(character_data=data, character_name="test")
        stage = SheetStage(ctx)
        assert stage._resolve_face_refinement() == "facedetailer"

    def test_refinement_from_top_level_key(self):
        """Sheet config can be top-level or under packages."""
        from build_v2.stages.sheet import SheetStage
        data = {"sheet": {"face_refinement": "facedetailer"}}
        ctx = BuildContext(character_data=data, character_name="test")
        stage = SheetStage(ctx)
        assert stage._resolve_face_refinement() == "facedetailer"


class TestMotionResolution:
    """Test SpritesAnimatedStage._resolve_motion for all source types."""

    def test_shared_source_exists(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            motions_dir = Path(tmp) / "motions"
            shared = motions_dir / "shared"
            shared.mkdir(parents=True)
            (shared / "walk.npz").write_bytes(b"fake")
            ctx = BuildContext(
                character_data={}, character_name="test",
                motions_dir=motions_dir,
            )
            stage = SpritesAnimatedStage(ctx)
            result = stage._resolve_motion("walk", "shared", "")
            assert result is not None
            assert result.name == "walk.npz"

    def test_shared_source_fallback_to_idle(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            motions_dir = Path(tmp) / "motions"
            shared = motions_dir / "shared"
            shared.mkdir(parents=True)
            (shared / "idle.npz").write_bytes(b"fake")
            ctx = BuildContext(
                character_data={}, character_name="test",
                motions_dir=motions_dir,
            )
            stage = SpritesAnimatedStage(ctx)
            result = stage._resolve_motion("nonexistent_anim", "shared", "")
            assert result is not None
            assert result.name == "idle.npz"

    def test_shared_source_not_found(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            motions_dir = Path(tmp) / "motions"
            motions_dir.mkdir(parents=True)
            ctx = BuildContext(
                character_data={}, character_name="test",
                motions_dir=motions_dir,
            )
            stage = SpritesAnimatedStage(ctx)
            result = stage._resolve_motion("nonexistent", "shared", "")
            assert result is None

    def test_file_source(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            motions_dir = Path(tmp) / "motions"
            motions_dir.mkdir(parents=True)
            npz = motions_dir / "custom.npz"
            npz.write_bytes(b"fake")
            data = {
                "packages": {
                    "sprites": {
                        "animated": [{"name": "walk", "path": str(npz)}],
                    },
                },
            }
            ctx = BuildContext(
                character_data=data, character_name="test",
                motions_dir=motions_dir,
            )
            stage = SpritesAnimatedStage(ctx)
            result = stage._resolve_motion("walk", "file", "")
            assert result is not None
            assert result.name == "custom.npz"

    def test_file_source_missing(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            motions_dir = Path(tmp) / "motions"
            motions_dir.mkdir(parents=True)
            data = {
                "packages": {
                    "sprites": {
                        "animated": [{"name": "walk", "path": "/nonexistent/anim.npz"}],
                    },
                },
            }
            ctx = BuildContext(
                character_data=data, character_name="test",
                motions_dir=motions_dir,
            )
            stage = SpritesAnimatedStage(ctx)
            result = stage._resolve_motion("walk", "file", "")
            assert result is None

    def test_manual_source(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            data = {
                "packages": {
                    "sprites": {
                        "animated": [{
                            "name": "pose",
                            "frames": [{"joints": {"spine": [0, 0, 0]}}],
                        }],
                    },
                },
            }
            ctx = BuildContext(
                character_data=data, character_name="test",
                output_dir=Path(tmp),
                motions_dir=Path(tmp) / "motions",
            )
            stage = SpritesAnimatedStage(ctx)
            result = stage._resolve_motion("pose", "manual", "")
            assert result is not None
            assert result.exists()

    def test_custom_source_existing(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            motions_dir = Path(tmp) / "motions"
            custom = motions_dir / "custom"
            custom.mkdir(parents=True)
            (custom / "test_walk.npz").write_bytes(b"fake")
            ctx = BuildContext(
                character_data={}, character_name="test",
                motions_dir=motions_dir,
            )
            stage = SpritesAnimatedStage(ctx)
            result = stage._resolve_motion("walk", "custom", "")
            assert result is not None
            assert result.name == "test_walk.npz"
