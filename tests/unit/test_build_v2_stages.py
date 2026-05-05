"""Unit tests for individual stages — instantiation, completeness, behavior.

Covers all 11 stages: metadata, is_complete(), and unit-testable logic paths
that don't require ComfyUI or external services.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from build_v2.context import BuildContext
from build_v2.stages.base import StageResult


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
        from build_v2.stages.generate import GenerateStage

        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(
                character_data={"generation": {"source": "import"}},
                character_name="test",
                output_dir=Path(tmp),
            )
            stage = GenerateStage(ctx)

            assert stage.is_complete({}) is False

            ctx.design_dir.mkdir(parents=True, exist_ok=True)
            ctx.approved_image.write_bytes(b"fake png")
            assert stage.is_complete({}) is True


class TestStateStage:
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

    def test_is_complete_png_only_not_complete(self):
        """PNG frames alone should not mark video as complete — only MP4 counts."""
        from build_v2.stages.video import VideoStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            video_dir = Path(tmp) / "video"
            video_dir.mkdir()
            (video_dir / "frame_0001.png").write_bytes(b"fake png")
            stage = VideoStage(ctx)
            assert stage.is_complete({}) is False

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

    def test_encode_mp4_no_frames(self):
        from build_v2.stages.video import VideoStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = VideoStage(ctx)
            result = stage._encode_mp4([], Path(tmp) / "out.mp4", ctx)
            assert result == []

    def test_encode_mp4_ffmpeg_not_found(self):
        """When ffmpeg is missing, should fall back to listing individual frames."""
        from build_v2.stages.video import VideoStage
        from PIL import Image as PILImage
        import io

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = BuildContext(output_dir=tmp_path, character_name="test")
            stage = VideoStage(ctx)

            # Create actual valid PNG frames
            frame_dir = tmp_path / "video" / "frames"
            frame_dir.mkdir(parents=True)
            frames = []
            for i in range(3):
                img = PILImage.new("RGB", (64, 64), color=(i * 80, 0, 0))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                p = frame_dir / f"frame_{i:04d}.png"
                p.write_bytes(buf.getvalue())
                frames.append(p)

            with patch("subprocess.run", side_effect=FileNotFoundError("no ffmpeg")):
                result = stage._encode_mp4(frames, tmp_path / "out.mp4", ctx)
                assert len(result) == 3
                assert all(r.endswith(".png") for r in result)

    def test_encode_mp4_timeout(self):
        """When ffmpeg times out, should fall back to listing individual frames."""
        from build_v2.stages.video import VideoStage
        from PIL import Image as PILImage
        import io

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = BuildContext(output_dir=tmp_path, character_name="test")
            stage = VideoStage(ctx)

            frame_dir = tmp_path / "video" / "frames"
            frame_dir.mkdir(parents=True)
            frames = []
            for i in range(2):
                img = PILImage.new("RGB", (64, 64), color=(0, i * 80, 0))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                p = frame_dir / f"frame_{i:04d}.png"
                p.write_bytes(buf.getvalue())
                frames.append(p)

            with patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("ffmpeg", 120)):
                result = stage._encode_mp4(frames, tmp_path / "out.mp4", ctx)
                assert len(result) == 2
                assert all(r.endswith(".png") for r in result)


class TestLoraStage:
    def test_is_complete_no_dir(self):
        from build_v2.stages.lora import LoraStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = LoraStage(ctx)
            assert stage.is_complete({}) is False

    def test_is_complete_with_png(self):
        from build_v2.stages.lora import LoraStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            lora_dir = Path(tmp) / "lora"
            lora_dir.mkdir()
            (lora_dir / "dataset_001.png").write_bytes(b"fake png")
            stage = LoraStage(ctx)
            assert stage.is_complete({}) is True

    def test_run_no_static_sprites(self):
        from build_v2.stages.lora import LoraStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = LoraStage(ctx)
            result = stage.run({})
            assert result.status == "skipped"
            assert "sprite" in result.error.lower()

    def test_fetch_dataset_handles_connection_error(self):
        """_fetch_dataset should not crash on connection errors."""
        from build_v2.stages.lora import LoraStage
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = BuildContext(
                output_dir=tmp_path,
                character_name="test",
                comfyui_url="http://localhost:99999",  # nothing listening
            )
            stage = LoraStage(ctx)
            lora_dir = tmp_path / "lora"
            lora_dir.mkdir()
            # Should not raise — just print a warning
            stage._fetch_dataset(ctx, lora_dir)


class TestSheetStage:
    def test_is_complete_no_dir(self):
        from build_v2.stages.sheet import SheetStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = SheetStage(ctx)
            assert stage.is_complete({}) is False

    def test_is_complete_with_sheet(self):
        from build_v2.stages.sheet import SheetStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            sheet_dir = Path(tmp) / "sheet"
            sheet_dir.mkdir()
            (sheet_dir / "sheet_000.png").write_bytes(b"fake png")
            stage = SheetStage(ctx)
            assert stage.is_complete({}) is True

    def test_generate_sheet_no_approved_image(self):
        from build_v2.stages.sheet import SheetStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = SheetStage(ctx)
            result = stage._generate_sheet()
            assert result.status == "failed"
            assert "approved" in result.error.lower()

    def test_check_bbox_model_missing(self):
        """When face_yolov8n.pt doesn't exist, bbox check should return False."""
        from build_v2.stages.sheet import SheetStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = SheetStage(ctx)
            assert stage._check_bbox_model_available() is False

    def test_check_bbox_model_exists(self):
        """When face_yolov8n.pt exists, bbox check should return True."""
        from build_v2.stages.sheet import SheetStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = SheetStage(ctx)
            # Create the model file in one of the expected locations
            model_dir = Path("models/ultralytics")
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "face_yolov8n.pt").write_bytes(b"fake model")
            try:
                assert stage._check_bbox_model_available() is True
            finally:
                (model_dir / "face_yolov8n.pt").unlink(missing_ok=True)
                model_dir.rmdir()

    def test_refine_faces_no_bbox_model(self):
        """FaceDetailer should return clear error when bbox model is missing."""
        from build_v2.stages.sheet import SheetStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = SheetStage(ctx)
            sheet_result = StageResult(status="complete", artifacts=["sheet/sheet_000.png"])
            result = stage._refine_faces(sheet_result)
            assert result.status == "failed"
            assert "face_yolov8n.pt" in result.error

    def test_refine_faces_no_sheet_images(self):
        """FaceDetailer should skip when no sheet images exist on disk."""
        from build_v2.stages.sheet import SheetStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = SheetStage(ctx)
            # Mock bbox model as available
            stage._check_bbox_model_available = lambda: True
            sheet_result = StageResult(
                status="complete",
                artifacts=["sheet/sheet_000.png"],
                meta={},
            )
            result = stage._refine_faces(sheet_result)
            # Should mark as skipped (no actual sheet PNGs on disk)
            assert result.meta.get("face_refinement") == "skipped_no_images"


class TestSpritesAnimatedStage:
    """Test SpritesAnimatedStage logic paths that don't need ComfyUI."""

    def test_is_complete_no_dir(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = SpritesAnimatedStage(ctx)
            assert stage.is_complete({}) is True  # no animations declared → vacuously true

    def test_is_complete_with_animation_missing_frames(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            data = {
                "packages": {
                    "sprites": {
                        "animated": [{"name": "walk"}],
                        "outfit": "default",
                        "directions": ["front"],
                    },
                },
            }
            ctx = BuildContext(
                character_data=data, character_name="test",
                output_dir=Path(tmp),
            )
            stage = SpritesAnimatedStage(ctx)
            assert stage.is_complete({}) is False

    def test_is_complete_with_frames_present(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            data = {
                "packages": {
                    "sprites": {
                        "animated": [{"name": "walk"}],
                        "outfit": "casual",
                        "directions": ["front"],
                    },
                },
            }
            ctx = BuildContext(
                character_data=data, character_name="test",
                output_dir=Path(tmp),
            )
            frame_dir = Path(tmp) / "sprites" / "casual" / "normal" / "walk" / "front"
            frame_dir.mkdir(parents=True)
            (frame_dir / "frame_001.png").write_bytes(b"fake")
            stage = SpritesAnimatedStage(ctx)
            assert stage.is_complete({}) is True

    def test_resolve_trellis_glb_from_artifacts(self):
        """Should find GLB path from trellis stage artifacts."""
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = BuildContext(output_dir=tmp_path, character_name="test")
            stage = SpritesAnimatedStage(ctx)

            # Create a fake GLB
            trellis_dir = tmp_path / "trellis"
            trellis_dir.mkdir()
            glb_path = trellis_dir / "test.glb"
            glb_path.write_bytes(b"fake glb")

            state = {
                "stages": {
                    "trellis": {
                        "artifacts": ["trellis/test.glb"],
                    },
                },
            }
            result = stage._resolve_trellis_glb(state)
            assert result is not None
            assert result.name == "test.glb"

    def test_resolve_trellis_glb_no_artifacts(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = SpritesAnimatedStage(ctx)
            assert stage._resolve_trellis_glb({}) is None
            assert stage._resolve_trellis_glb({"stages": {"trellis": {"artifacts": []}}}) is None

    def test_resolve_trellis_glb_file_missing(self):
        """Should return None when GLB path in artifacts doesn't exist on disk."""
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = SpritesAnimatedStage(ctx)
            state = {"stages": {"trellis": {"artifacts": ["trellis/missing.glb"]}}}
            assert stage._resolve_trellis_glb(state) is None

    def test_run_fallback_hymotion_to_llm(self):
        """When motion=hymotion but no NPZ files, should fall back to llm."""
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            data = {
                "packages": {
                    "sprites": {
                        "motion_strategy": "hymotion",
                        "animated": [{"name": "walk"}],
                    },
                },
            }
            ctx = BuildContext(
                character_data=data, character_name="test",
                output_dir=Path(tmp),
                motions_dir=Path(tmp) / "motions",
            )
            stage = SpritesAnimatedStage(ctx)
            assert stage._resolve_motion_strategy() == "hymotion"
            assert stage._check_hymotion_available() is False
            # The run() method should auto-fallback — test the logic path
            assert not (Path(tmp) / "motions" / "shared").exists()

    def test_build_manual_npz(self):
        """Should create NPZ from manual joint rotation data."""
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        with tempfile.TemporaryDirectory() as tmp:
            data = {
                "packages": {
                    "sprites": {
                        "animated": [{
                            "name": "pose",
                            "frames": [
                                {"joints": {"spine": [0, 0, 0]}},
                                {"joints": {"spine": [10, 0, 0]}},
                            ],
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
            npz_path = stage._build_manual_npz("pose")
            assert npz_path.exists()
            assert npz_path.name == "test_pose_manual.npz"

    def test_parse_rotations_json_direct_array(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        ctx = BuildContext(character_name="test")
        stage = SpritesAnimatedStage(ctx)

        rotations = [
            {"spine": [0, 0, 0], "left_arm": [10, 0, 0]},
            {"spine": [5, 0, 0], "left_arm": [20, 0, 0]},
            {"spine": [10, 0, 0], "left_arm": [30, 0, 0]},
        ]
        result = stage._parse_rotations_json(json.dumps(rotations), 2)
        assert len(result) == 2  # truncated to num_keyframes

    def test_parse_rotations_json_single_dict(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        ctx = BuildContext(character_name="test")
        stage = SpritesAnimatedStage(ctx)

        rotation = {"spine": [0, 0, 0]}
        result = stage._parse_rotations_json(json.dumps(rotation), 6)
        assert len(result) == 1
        assert result[0] == rotation

    def test_parse_rotations_json_markdown_wrapped(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        ctx = BuildContext(character_name="test")
        stage = SpritesAnimatedStage(ctx)

        text = '```json\n[{"spine": [0, 0, 0]}]\n```'
        result = stage._parse_rotations_json(text, 6)
        assert len(result) == 1

    def test_parse_rotations_json_garbage(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        ctx = BuildContext(character_name="test")
        stage = SpritesAnimatedStage(ctx)

        result = stage._parse_rotations_json("not json at all", 6)
        assert result == []

    def test_parse_rotations_json_trailing_comma(self):
        from build_v2.stages.sprites_animated import SpritesAnimatedStage
        ctx = BuildContext(character_name="test")
        stage = SpritesAnimatedStage(ctx)

        text = '[{"spine": [0, 0, 0]},]'
        result = stage._parse_rotations_json(text, 6)
        assert len(result) == 1

    def test_direction_rotations_constant(self):
        from build_v2.stages.sprites_animated import DIRECTION_ROTATIONS
        assert DIRECTION_ROTATIONS == {
            "front": 0.0,
            "right": 90.0,
            "back": 180.0,
            "left": 270.0,
        }


class TestSpritesStaticStage:
    """Test SpritesStaticStage strategy dispatch and Trellis path."""

    def test_is_complete_no_dir(self):
        from build_v2.stages.sprites_static import SpritesStaticStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = SpritesStaticStage(ctx)
            assert stage.is_complete({}) is False

    def test_is_complete_with_sprites(self):
        from build_v2.stages.sprites_static import SpritesStaticStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            static_dir = Path(tmp) / "sprites" / "static"
            static_dir.mkdir(parents=True)
            (static_dir / "sprite_000.png").write_bytes(b"fake")
            stage = SpritesStaticStage(ctx)
            assert stage.is_complete({}) is True

    def test_run_fallback_trellis_to_anny(self):
        """When body_reference=trellis but trellis stage not complete, falls back to anny."""
        from build_v2.stages.sprites_static import SpritesStaticStage
        with tempfile.TemporaryDirectory() as tmp:
            data = {"packages": {"sprites": {"body_reference": "trellis", "static": True}}}
            ctx = BuildContext(
                character_data=data, character_name="test",
                output_dir=Path(tmp),
            )
            stage = SpritesStaticStage(ctx)

            # Mock executor to avoid real ComfyUI calls
            stage.executor = MagicMock()
            stage.executor.run_api_workflow.return_value = [b"fake png"]

            state = {"stages": {"trellis": {"status": "failed"}}}
            result = stage.run(state)
            # Should have fallen back to anny and called run_api_workflow
            stage.executor.run_api_workflow.assert_called_once()

    def test_run_trellis_no_glb(self):
        """When body_reference=trellis and stage complete but no GLB artifact, should fail."""
        from build_v2.stages.sprites_static import SpritesStaticStage
        with tempfile.TemporaryDirectory() as tmp:
            data = {"packages": {"sprites": {"body_reference": "trellis"}}}
            ctx = BuildContext(
                character_data=data, character_name="test",
                output_dir=Path(tmp),
            )
            stage = SpritesStaticStage(ctx)
            state = {
                "stages": {
                    "trellis": {
                        "status": "complete",
                        "artifacts": [],  # no GLB
                    },
                },
            }
            result = stage.run(state)
            assert result.status == "failed"
            assert "glb" in result.error.lower()

    def test_direction_rotations_constant(self):
        from build_v2.stages.sprites_static import DIRECTION_ROTATIONS
        assert DIRECTION_ROTATIONS == {
            "front": 0.0,
            "right": 90.0,
            "back": 180.0,
            "left": 270.0,
        }


class TestStageBase:
    """Test Stage base class methods."""

    def test_check_prerequisites_all_met(self):
        from build_v2.stages.generate import GenerateStage
        ctx = BuildContext(character_name="test")
        stage = GenerateStage(ctx)
        # GenerateStage has no dependencies
        assert stage.check_prerequisites({}) == []

    def test_check_prerequisites_missing(self):
        from build_v2.stages.sheet import SheetStage
        ctx = BuildContext(character_name="test")
        stage = SheetStage(ctx)
        # SheetStage depends on "generate"
        assert stage.check_prerequisites({}) == ["generate"]

    def test_check_prerequisites_satisfied(self):
        from build_v2.stages.sheet import SheetStage
        ctx = BuildContext(character_name="test")
        stage = SheetStage(ctx)
        state = {"stages": {"generate": {"status": "complete"}}}
        assert stage.check_prerequisites(state) == []

    def test_save_artifacts(self):
        from build_v2.stages.generate import GenerateStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = GenerateStage(ctx)
            target = Path(tmp) / "design"
            result = stage._save_artifacts([b"img1", b"img2"], target, "approved")
            assert len(result) == 2
            assert (target / "approved_000.png").exists()
            assert (target / "approved_001.png").exists()

    def test_artifacts_exist_all_present(self):
        from build_v2.stages.generate import GenerateStage
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = BuildContext(output_dir=tmp_path, character_name="test")
            stage = GenerateStage(ctx)
            design = tmp_path / "design"
            design.mkdir()
            (design / "approved_000.png").write_bytes(b"fake")
            state = {"stages": {"generate": {"artifacts": ["design/approved_000.png"]}}}
            assert stage.artifacts_exist(state) is True

    def test_artifacts_exist_missing(self):
        from build_v2.stages.generate import GenerateStage
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BuildContext(output_dir=Path(tmp), character_name="test")
            stage = GenerateStage(ctx)
            state = {"stages": {"generate": {"artifacts": ["design/missing.png"]}}}
            assert stage.artifacts_exist(state) is False
