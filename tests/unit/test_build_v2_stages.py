"""Unit tests for individual stages — instantiation, completeness, behavior."""

import tempfile
from pathlib import Path

import pytest

from build_v2.context import BuildContext


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

    def test_state_stage(self):
        from build_v2.stages.state import StateStage
        s = StateStage(None)
        assert s.name == "state"
        assert s.tier_required == "t1"
        assert "sheet" in s.depends_on

    def test_trellis_stage(self):
        from build_v2.stages.trellis import TrellisStage
        s = TrellisStage(None)
        assert s.name == "trellis"
        assert s.tier_required == "t2"
        assert "generate" in s.depends_on

    def test_video_stage(self):
        from build_v2.stages.video import VideoStage
        s = VideoStage(None)
        assert s.name == "video"
        assert s.tier_required == "t1.5"
        assert "sprites_static" in s.depends_on

    def test_godot_stage(self):
        from build_v2.stages.godot import GodotStage
        s = GodotStage(None)
        assert s.name == "godot"
        assert s.tier_required == "t1"
        assert "sprites_static" in s.depends_on


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


class TestStateStage:
    """Test StateStage behavior (no ComfyUI needed)."""

    def test_is_complete_no_dir(self):
        from build_v2.stages.state import StateStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = StateStage(ctx)
            assert stage.is_complete({}) is False

    def test_is_complete_with_images(self):
        from build_v2.stages.state import StateStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            state_dir = Path(tmp) / "state"
            state_dir.mkdir()
            (state_dir / "beatup_000.png").write_bytes(b"fake png")
            stage = StateStage(ctx)
            assert stage.is_complete({}) is True

    def test_get_state_package_from_packages(self):
        from build_v2.stages.state import StateStage
        data = {"packages": {"state": {"name": "beatup", "description": "bruised"}}}
        ctx = BuildContext(character_data=data, character_name="test")
        stage = StateStage(ctx)
        pkg = stage._get_state_package()
        assert pkg["name"] == "beatup"

    def test_get_state_package_flat(self):
        from build_v2.stages.state import StateStage
        data = {"state": {"name": "wet", "description": "soaked"}}
        ctx = BuildContext(character_data=data, character_name="test")
        stage = StateStage(ctx)
        pkg = stage._get_state_package()
        assert pkg["name"] == "wet"

    def test_build_instruction(self):
        from build_v2.stages.state import StateStage
        ctx = BuildContext(character_name="test")
        stage = StateStage(ctx)
        instruction = stage._build_instruction("beatup", "bruised face, black eye")
        assert "beatup" in instruction
        assert "bruised face, black eye" in instruction
        assert "character identity" in instruction

    def test_run_no_state_declared(self):
        from build_v2.stages.state import StateStage
        ctx = BuildContext(character_data={}, character_name="test")
        stage = StateStage(ctx)
        result = stage.run({})
        assert result.status == "skipped"

    def test_run_no_sheet(self):
        from build_v2.stages.state import StateStage
        with tempfile.TemporaryDirectory() as tmp:
            data = {"packages": {"state": {"name": "beatup", "description": "bruised"}}}
            ctx = BuildContext(
                character_data=data, character_name="test",
                output_dir=Path(tmp),
            )
            stage = StateStage(ctx)
            result = stage.run({})
            assert result.status == "failed"
            assert "sheet" in result.error.lower()


class TestTrellisStage:
    """Test TrellisStage behavior (no TRELLIS service needed)."""

    def test_is_complete_no_dir(self):
        from build_v2.stages.trellis import TrellisStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = TrellisStage(ctx)
            assert stage.is_complete({}) is False

    def test_is_complete_with_glb(self):
        from build_v2.stages.trellis import TrellisStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            trellis_dir = Path(tmp) / "trellis"
            trellis_dir.mkdir()
            (trellis_dir / "test.glb").write_bytes(b"fake glb")
            stage = TrellisStage(ctx)
            assert stage.is_complete({}) is True

    def test_run_no_approved_image(self):
        from build_v2.stages.trellis import TrellisStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = TrellisStage(ctx)
            result = stage.run({})
            assert result.status == "failed"
            assert "image" in result.error.lower()

    def test_get_trellis_url_default(self):
        from build_v2.stages.trellis import TrellisStage
        ctx = BuildContext(character_name="test")
        stage = TrellisStage(ctx)
        assert stage._get_trellis_url() == "http://localhost:18080/3d/trellis"

    def test_get_trellis_url_env(self):
        from build_v2.stages.trellis import TrellisStage
        ctx = BuildContext(character_name="test")
        stage = TrellisStage(ctx)
        old = __import__("os").environ.get("TRELLIS_URL")
        try:
            __import__("os").environ["TRELLIS_URL"] = "http://custom:9999/trellis"
            assert stage._get_trellis_url() == "http://custom:9999/trellis"
        finally:
            if old is None:
                __import__("os").environ.pop("TRELLIS_URL", None)
            else:
                __import__("os").environ["TRELLIS_URL"] = old


class TestVideoStage:
    """Test VideoStage behavior (no ComfyUI needed)."""

    def test_is_complete_no_dir(self):
        from build_v2.stages.video import VideoStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = VideoStage(ctx)
            assert stage.is_complete({}) is False

    def test_is_complete_with_mp4(self):
        from build_v2.stages.video import VideoStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            video_dir = Path(tmp) / "video"
            video_dir.mkdir()
            (video_dir / "clip_000.mp4").write_bytes(b"fake mp4")
            stage = VideoStage(ctx)
            assert stage.is_complete({}) is True

    def test_run_no_sprites(self):
        from build_v2.stages.video import VideoStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = VideoStage(ctx)
            result = stage.run({})
            assert result.status == "skipped"
            assert "sprite" in result.error.lower()

    def test_get_sprite_frames_empty(self):
        from build_v2.stages.video import VideoStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = VideoStage(ctx)
            assert stage._get_sprite_frames() == []

    def test_get_sprite_frames_found(self):
        from build_v2.stages.video import VideoStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            static_dir = Path(tmp) / "sprites" / "static"
            static_dir.mkdir(parents=True)
            (static_dir / "sprite_001.png").write_bytes(b"fake")
            (static_dir / "sprite_002.png").write_bytes(b"fake")
            stage = VideoStage(ctx)
            frames = stage._get_sprite_frames()
            assert len(frames) == 2
