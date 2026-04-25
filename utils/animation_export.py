"""Animation export: render frames, encode as MP4 or GIF.

Takes interpolated per-frame rotation dicts, renders each as a 3D body mesh
using the existing Anny + pyrender pipeline, then encodes as video.
"""

import subprocess
import tempfile

import numpy as np


def render_animation_frames(
    all_frame_rotations: list[dict[str, list[float]]],
    width: int = 512,
    height: int = 512,
    backend: str = "auto",
) -> list[np.ndarray]:
    """Render all interpolated frames as body mesh images.

    Args:
        all_frame_rotations: List of rotation dicts, one per frame.
        width: Render width.
        height: Render height.
        backend: "auto", "pyrender", or "pil".

    Returns:
        List of (H, W, 3) uint8 numpy arrays.
    """
    from .anny_compat import forward_anny, get_faces
    from .mesh_renderer import render_mesh

    faces = get_faces().cpu().numpy()
    images = []

    for i, rotations in enumerate(all_frame_rotations):
        result = forward_anny(rotations)
        vertices = result["vertices"][0].detach().cpu().numpy()
        image = render_mesh(vertices, faces, width, height, backend=backend)
        images.append(image)

        if (i + 1) % 10 == 0:
            print(f"[animation_export] Rendered {i + 1}/{len(all_frame_rotations)} frames")

    return images


def frames_to_mp4(
    frames: list[np.ndarray],
    output_path: str,
    fps: int = 30,
) -> str:
    """Write frames to MP4 using ffmpeg pipe-in.

    Args:
        frames: List of (H, W, 3) uint8 numpy arrays.
        output_path: Output file path.
        fps: Frame rate.

    Returns:
        The output_path.
    """
    if not frames:
        raise ValueError("No frames to encode")

    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", output_path,
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    stderr = proc.stderr.read()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (code {proc.returncode}): {stderr.decode()[:500]}")

    return output_path


def frames_to_gif(
    frames: list[np.ndarray],
    output_path: str,
    fps: int = 15,
    loop: int = 0,
) -> str:
    """Write frames to GIF using imageio.

    Args:
        frames: List of (H, W, 3) uint8 numpy arrays.
        output_path: Output file path.
        fps: Frame rate (GIF typically uses lower fps).
        loop: Number of loops (0 = infinite).

    Returns:
        The output_path.
    """
    if not frames:
        raise ValueError("No frames to encode")

    import imageio.v2 as imageio

    imageio.mimsave(output_path, frames, duration=int(1000 / fps), loop=loop)
    return output_path


def get_output_dir() -> str:
    """Get the ComfyUI output directory, or a temp dir as fallback."""
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except ImportError:
        return tempfile.gettempdir()
