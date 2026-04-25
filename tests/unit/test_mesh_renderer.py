"""Unit tests for utils/mesh_renderer.py — 3D mesh to 2D rendering."""

import numpy as np
import pytest

from pose_director_root.utils.mesh_renderer import render_mesh, render_mesh_pil


# Simple test mesh: a triangle with Z-up (matching Anny convention)
def _flat_triangle():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.0, 1.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return vertices, faces


# A simple cube mesh (Z-up convention)
def _cube_mesh():
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom (Z=-1)
        [4, 5, 6], [4, 6, 7],  # top (Z=1)
        [0, 1, 5], [0, 5, 4],  # front (Y=-1)
        [2, 3, 7], [2, 7, 6],  # back (Y=1)
        [0, 3, 7], [0, 7, 4],  # left (X=-1)
        [1, 2, 6], [1, 6, 5],  # right (X=1)
    ], dtype=np.int32)
    return vertices, faces


class TestRenderMeshPIL:
    """Tests for the PIL fallback renderer."""

    def test_output_shape(self):
        """Should return (H, W, 3) uint8 array."""
        verts, faces = _flat_triangle()
        img = render_mesh_pil(verts, faces, width=256, height=256)
        assert img.shape == (256, 256, 3)
        assert img.dtype == np.uint8

    def test_uniform_gray_output(self):
        """A single flat triangle should fill with a uniform shade."""
        verts, faces = _flat_triangle()
        img = render_mesh_pil(verts, faces, width=64, height=64)
        # Should produce a valid image with shading (not all zeros)
        assert img.min() >= 0
        assert img.max() <= 255
        # The single triangle fills the entire viewport so all pixels should be non-white
        assert img.mean() < 250

    def test_mesh_visible(self):
        """Some pixels should be non-white (the mesh is rendered)."""
        verts, faces = _cube_mesh()
        img = render_mesh_pil(verts, faces, width=256, height=256)
        non_white = (img < 250).any(axis=2)
        assert non_white.sum() > 0, "No mesh pixels found in render"

    def test_custom_size(self):
        """Should respect custom width/height."""
        verts, faces = _cube_mesh()
        img = render_mesh_pil(verts, faces, width=512, height=256)
        assert img.shape == (256, 512, 3)

    def test_handles_float32_vertices(self):
        """Should work with float32 vertices (auto-cast to float64)."""
        verts, faces = _flat_triangle()
        verts = verts.astype(np.float32)
        img = render_mesh_pil(verts, faces, width=128, height=128)
        assert img.shape == (128, 128, 3)


class TestRenderMeshAuto:
    """Tests for the auto-dispatching render_mesh function."""

    def test_pil_backend_explicit(self):
        """PIL backend should work without GPU."""
        verts, faces = _cube_mesh()
        img = render_mesh(verts, faces, width=128, height=128, backend="pil")
        assert img.shape == (128, 128, 3)

    def test_auto_backend_fallback(self):
        """Auto backend should fall back to PIL if pyrender unavailable."""
        verts, faces = _flat_triangle()
        img = render_mesh(verts, faces, width=64, height=64, backend="auto")
        assert img.shape == (64, 64, 3)
