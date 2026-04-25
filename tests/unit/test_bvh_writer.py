"""Tests for BVH motion capture file writer."""

import os
import tempfile

import pytest
from scipy.spatial.transform import Rotation

from utils.bvh_writer import (
    BVH_JOINT_ORDER,
    JOINT_TO_BVH,
    _build_hierarchy_lines,
    _euler_xyz_to_bvh_zxy,
    write_bvh,
)
from utils.interpolation import JOINT_NAMES


def _tpose_frames(n=10):
    """Generate n T-pose frames."""
    return [{j: [0.0, 0.0, 0.0] for j in JOINT_NAMES} for _ in range(n)]


class TestEulerConversion:
    def test_zero_rotation(self):
        """Zero rotation should stay zero."""
        result = _euler_xyz_to_bvh_zxy([0, 0, 0])
        assert result == pytest.approx([0, 0, 0], abs=1e-6)

    def test_roundtrip(self):
        """XYZ → ZXY → back to XYZ should preserve the rotation."""
        original = [30, 45, 60]
        bvh = _euler_xyz_to_bvh_zxy(original)
        # Reconstruct from ZXY
        r_bvh = Rotation.from_euler("ZXY", bvh, degrees=True)
        r_orig = Rotation.from_euler("XYZ", original, degrees=True)
        # Rotation matrices should match
        import numpy as np
        assert np.allclose(r_bvh.as_matrix(), r_orig.as_matrix(), atol=1e-6)

    def test_single_axis_rotation(self):
        """Pure X rotation should convert correctly."""
        # Pure X rotation: XYZ=[90,0,0] → ZXY decomposition
        bvh = _euler_xyz_to_bvh_zxy([90, 0, 0])
        r_bvh = Rotation.from_euler("ZXY", bvh, degrees=True)
        r_orig = Rotation.from_euler("XYZ", [90, 0, 0], degrees=True)
        import numpy as np
        assert np.allclose(r_bvh.as_matrix(), r_orig.as_matrix(), atol=1e-5)

    def test_multiple_angles(self):
        """Test several known rotation combos roundtrip correctly."""
        import numpy as np
        test_cases = [
            [0, 0, 0], [45, 0, 0], [0, 45, 0], [0, 0, 45],
            [30, 60, 90], [-45, 30, -60], [10, 20, 30],
        ]
        for original in test_cases:
            bvh = _euler_xyz_to_bvh_zxy(original)
            r_bvh = Rotation.from_euler("ZXY", bvh, degrees=True)
            r_orig = Rotation.from_euler("XYZ", original, degrees=True)
            assert np.allclose(r_bvh.as_matrix(), r_orig.as_matrix(), atol=1e-5), \
                f"Failed for {original}: bvh={bvh}"


class TestBVHFileStructure:
    def test_hierarchy_section_present(self):
        """File should have HIERARCHY section."""
        lines = _build_hierarchy_lines()
        assert lines[0] == "HIERARCHY"

    def test_has_root_joint(self):
        """Hierarchy should have ROOT Hips."""
        lines = _build_hierarchy_lines()
        assert any("ROOT Hips" in line for line in lines)

    def test_joint_count(self):
        """Should have 1 ROOT + 15 JOINTs = 16 total joints."""
        lines = _build_hierarchy_lines()
        joint_lines = [l for l in lines if "JOINT" in l or "ROOT" in l]
        assert len(joint_lines) == 16  # 1 ROOT + 15 JOINTs

    def test_all_bvh_joints_present(self):
        """All expected BVH joints should appear in hierarchy."""
        lines_text = "\n".join(_build_hierarchy_lines())
        for joint in BVH_JOINT_ORDER:
            assert joint in lines_text, f"Missing joint: {joint}"

    def test_braces_balanced(self):
        """Opening and closing braces should match."""
        lines = _build_hierarchy_lines()
        text = "\n".join(lines)
        assert text.count("{") == text.count("}")


class TestWriteBVH:
    def test_file_created(self):
        """write_bvh should create a file."""
        with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
            path = f.name
        try:
            write_bvh(_tpose_frames(10), fps=30.0, output_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_file_has_both_sections(self):
        """File should contain both HIERARCHY and MOTION sections."""
        with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
            path = f.name
        try:
            write_bvh(_tpose_frames(10), fps=30.0, output_path=path)
            with open(path) as f:
                content = f.read()
            assert "HIERARCHY" in content
            assert "MOTION" in content
        finally:
            os.unlink(path)

    def test_frame_count_matches(self):
        """MOTION section should have correct frame count."""
        n_frames = 25
        with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
            path = f.name
        try:
            write_bvh(_tpose_frames(n_frames), fps=30.0, output_path=path)
            with open(path) as f:
                content = f.read()
            assert f"Frames: {n_frames}" in content
        finally:
            os.unlink(path)

    def test_frame_time_correct(self):
        """Frame time should be 1/fps."""
        with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
            path = f.name
        try:
            write_bvh(_tpose_frames(10), fps=24.0, output_path=path)
            with open(path) as f:
                content = f.read()
            assert "0.0416666" in content  # 1/24 ≈ 0.04166667
        finally:
            os.unlink(path)

    def test_motion_data_lines(self):
        """Number of motion data lines should match frame count."""
        n_frames = 15
        with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
            path = f.name
        try:
            write_bvh(_tpose_frames(n_frames), fps=30.0, output_path=path)
            with open(path) as f:
                lines = f.readlines()
            # Find MOTION section, skip header lines (Frames, Frame Time)
            motion_start = None
            for i, line in enumerate(lines):
                if line.strip().startswith("MOTION"):
                    motion_start = i
                    break
            assert motion_start is not None
            # +2 for Frames and Frame Time lines
            data_lines = lines[motion_start + 3:]
            # Strip empty trailing lines
            data_lines = [l for l in data_lines if l.strip()]
            assert len(data_lines) == n_frames
        finally:
            os.unlink(path)

    def test_tpose_all_zeros(self):
        """T-pose should produce all-zero rotation channels."""
        with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
            path = f.name
        try:
            write_bvh(_tpose_frames(3), fps=30.0, output_path=path)
            with open(path) as f:
                content = f.read()
            # Find motion data lines
            lines = content.strip().split("\n")
            motion_idx = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
            data_lines = lines[motion_idx + 3:]
            for line in data_lines:
                values = [float(v) for v in line.strip().split()]
                # Root position is 0, rotation channels should all be ~0
                for v in values:
                    assert abs(v) < 1e-4, f"Expected zero, got {v}"
        finally:
            os.unlink(path)

    def test_nonzero_rotation_in_output(self):
        """Non-zero rotation should appear in the BVH file."""
        frames = [{j: [0, 0, 0] for j in JOINT_NAMES} for _ in range(5)]
        frames[2]["head"] = [30, 45, 60]
        with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
            path = f.name
        try:
            write_bvh(frames, fps=30.0, output_path=path)
            with open(path) as f:
                content = f.read()
            lines = content.strip().split("\n")
            motion_idx = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
            data_lines = lines[motion_idx + 3:]
            # Frame 2 (index 2) should have non-zero values
            frame2_vals = [float(v) for v in data_lines[2].strip().split()]
            assert any(abs(v) > 1.0 for v in frame2_vals), \
                f"Frame 2 should have non-zero rotations, got {frame2_vals}"
        finally:
            os.unlink(path)

    def test_joint_name_mapping_complete(self):
        """All our joints should map to a BVH joint name."""
        for joint in JOINT_NAMES:
            assert joint in JOINT_TO_BVH, f"Missing mapping for {joint}"
