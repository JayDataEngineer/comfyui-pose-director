"""Unit tests for temporal stability analysis."""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from pose_director_root.utils.temporal_analysis import (
    analyze_temporal_stability,
    compute_ssim,
    extract_frames,
)


class TestComputeSSIM:
    """Tests for SSIM computation with numpy arrays."""

    def test_identical_images_ssim_one(self):
        """Two identical images should have SSIM of 1.0."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ssim = compute_ssim(img, img)
        assert ssim > 0.99

    def test_completely_different_images_low_ssim(self):
        """All-black vs all-white should have low SSIM."""
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        ssim = compute_ssim(black, white)
        assert ssim < 0.3

    def test_slight_noise_high_ssim(self):
        """Adding small noise should result in high SSIM (>0.9)."""
        np.random.seed(42)
        img_a = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        noise = np.random.randint(-5, 6, (64, 64, 3)).astype(np.int16)
        img_b = np.clip(img_a.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        ssim = compute_ssim(img_a, img_b)
        assert ssim > 0.9

    def test_different_shapes_handled(self):
        """Images with different shapes should be handled by cropping."""
        img_a = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img_b = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        ssim = compute_ssim(img_a, img_b)
        assert 0.0 <= ssim <= 1.0

    def test_small_image_handled(self):
        """Images smaller than window size should use whole-image comparison."""
        img = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        ssim = compute_ssim(img, img)
        assert ssim > 0.99


class TestAnalyzeTemporalStability:
    """Tests for the main analysis function using synthetic videos."""

    def _make_test_mp4(self, frames: list[np.ndarray], fps: int = 8) -> str:
        """Write numpy frames to a temp MP4 using ffmpeg."""
        from PIL import Image as PILImage

        tmpdir = tempfile.mkdtemp(prefix="test_temporal_")
        for i, frame in enumerate(frames):
            PILImage.fromarray(frame).save(os.path.join(tmpdir, f"frame_{i:04d}.png"))

        mp4_path = os.path.join(tmpdir, "test.mp4")
        frame_pattern = os.path.join(tmpdir, "frame_%04d.png")
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", frame_pattern,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4_path],
            capture_output=True, timeout=30,
        )
        return mp4_path

    def test_stable_video_high_score(self):
        """A video with consistent frames should have high stability score."""
        # All frames are the same
        frame = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        mp4 = self._make_test_mp4([frame] * 5)
        result = analyze_temporal_stability(mp4)
        assert result["temporal_stability_score"] > 0.95
        assert len(result["flicker_frames"]) == 0

    def test_flicker_detection(self):
        """A frame pair with low similarity should be detected as flicker."""
        np.random.seed(42)
        frames = [np.full((64, 64, 3), 128, dtype=np.uint8)] * 4
        # Insert a completely different frame in the middle
        frames[2] = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mp4 = self._make_test_mp4(frames)
        result = analyze_temporal_stability(mp4, threshold=0.8)
        assert len(result["flicker_frames"]) >= 1

    def test_missing_file_raises(self):
        """Should raise FileNotFoundError for non-existent MP4."""
        with pytest.raises(Exception):  # ffmpeg will fail
            analyze_temporal_stability("/nonexistent/video.mp4")

    def test_result_structure(self):
        """Result should have all expected keys."""
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        mp4 = self._make_test_mp4([frame] * 3)
        result = analyze_temporal_stability(mp4)
        assert "temporal_stability_score" in result
        assert "per_frame_metrics" in result
        assert "flicker_frames" in result
        assert "min_similarity" in result
        assert "max_similarity" in result
        assert "std_similarity" in result
        assert "num_frames" in result
        assert result["num_frames"] == 3

    def test_single_frame_defaults(self):
        """Single-frame video should return default similarity values."""
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        mp4 = self._make_test_mp4([frame])
        result = analyze_temporal_stability(mp4)
        assert result["num_frames"] == 1
        assert result["temporal_stability_score"] == 1.0
        assert result["per_frame_metrics"] == []
        assert result["flicker_frames"] == []


class TestExtractFrames:
    """Tests for extract_frames function."""

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing MP4."""
        with pytest.raises(FileNotFoundError):
            extract_frames("/nonexistent/path/video.mp4")

    def test_ffmpeg_failure(self):
        """Should raise RuntimeError when ffmpeg exits with error."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"not a valid mp4 file")
            bad_path = f.name
        try:
            with pytest.raises(RuntimeError, match="ffmpeg failed"):
                extract_frames(bad_path)
        finally:
            os.unlink(bad_path)

    def test_custom_output_dir(self):
        """Should extract frames to provided directory and not clean it up."""
        from PIL import Image as PILImage

        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        tmpdir = tempfile.mkdtemp(prefix="test_extract_")
        for i in range(3):
            PILImage.fromarray(frame).save(os.path.join(tmpdir, f"input_{i:04d}.png"))

        mp4_path = os.path.join(tmpdir, "test.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", "8", "-i", os.path.join(tmpdir, "input_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4_path],
            capture_output=True, timeout=30,
        )

        out_dir = tempfile.mkdtemp(prefix="test_output_")
        try:
            frames = extract_frames(mp4_path, output_dir=out_dir)
            assert len(frames) == 3
            assert all(os.path.exists(f) for f in frames)
            # Directory should still exist since cleanup=False
            assert os.path.isdir(out_dir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_auto_cleanup(self):
        """Without output_dir, temp directory should be cleaned up automatically."""
        from PIL import Image as PILImage

        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        tmpdir = tempfile.mkdtemp(prefix="test_extract_")
        for i in range(2):
            PILImage.fromarray(frame).save(os.path.join(tmpdir, f"input_{i:04d}.png"))

        mp4_path = os.path.join(tmpdir, "test.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", "8", "-i", os.path.join(tmpdir, "input_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4_path],
            capture_output=True, timeout=30,
        )

        try:
            frames = extract_frames(mp4_path)
            assert len(frames) == 2
            # After auto-cleanup, frames should not exist
            assert not os.path.exists(frames[0])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
