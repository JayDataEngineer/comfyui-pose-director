"""Tests for animation export utilities (render, MP4, GIF)."""

import os
import tempfile

import numpy as np
import pytest

from utils.animation_export import frames_to_mp4, frames_to_gif


def _solid_frames(n=10, w=64, h=64, color=(128, 64, 32)):
    """Generate n solid-color test frames."""
    return [np.full((h, w, 3), color, dtype=np.uint8) for _ in range(n)]


class TestFramesToMp4:
    def test_file_created(self):
        """MP4 file should be created."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            frames_to_mp4(_solid_frames(5), path, fps=24)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_empty_frames_raises(self):
        """Empty frame list should raise ValueError."""
        with pytest.raises(ValueError, match="No frames"):
            frames_to_mp4([], "/tmp/test.mp4")

    def test_different_sizes(self):
        """Should handle different frame sizes."""
        for w, h in [(32, 32), (128, 64), (64, 128)]:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                path = f.name
            try:
                frames_to_mp4(_solid_frames(3, w, h), path, fps=24)
                assert os.path.getsize(path) > 0
            finally:
                os.unlink(path)


class TestFramesToGif:
    def test_file_created(self):
        """GIF file should be created."""
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            path = f.name
        try:
            frames_to_gif(_solid_frames(5, w=32, h=32), path, fps=10)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_empty_frames_raises(self):
        """Empty frame list should raise ValueError."""
        with pytest.raises(ValueError, match="No frames"):
            frames_to_gif([], "/tmp/test.gif")

    def test_loop_parameter(self):
        """GIF with loop=1 should work."""
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            path = f.name
        try:
            frames_to_gif(_solid_frames(3, w=32, h=32), path, fps=10, loop=1)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)
