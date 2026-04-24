"""Temporal stability analysis for generated video files.

Extracts frames from MP4 using ffmpeg and computes frame-to-frame
similarity using SSIM (via numpy). Returns a stability score and
identifies flicker/morph frames.
"""

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def extract_frames(mp4_path: str, output_dir: str | None = None) -> list[str]:
    """Extract frames from MP4 using ffmpeg subprocess.

    Args:
        mp4_path: Path to the MP4 file.
        output_dir: Directory for extracted frames. If None, uses a temp dir.
    Returns:
        List of extracted frame file paths, sorted by frame number.
    Raises:
        FileNotFoundError: If mp4_path doesn't exist.
        RuntimeError: If ffmpeg fails.
    """
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"Video file not found: {mp4_path}")

    cleanup = False
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="temporal_analysis_")
        cleanup = True

    try:
        output_pattern = os.path.join(output_dir, "frame_%06d.png")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp4_path, "-q:v", "2", output_pattern],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

        frames = sorted(Path(output_dir).glob("frame_*.png"))
        return [str(f) for f in frames]
    finally:
        if cleanup and os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute simplified SSIM between two numpy arrays (H, W, C) uint8.

    Uses a windowed approach with 8x8 blocks. Returns 0.0-1.0 (1.0 = identical).
    """
    if img_a.shape != img_b.shape:
        # Resize to match
        min_h = min(img_a.shape[0], img_b.shape[0])
        min_w = min(img_a.shape[1], img_b.shape[1])
        img_a = img_a[:min_h, :min_w]
        img_b = img_b[:min_h, :min_w]

    # Convert to grayscale luminance
    if img_a.ndim == 3:
        a_gray = np.dot(img_a[..., :3], [0.299, 0.587, 0.114]).astype(np.float64)
        b_gray = np.dot(img_b[..., :3], [0.299, 0.587, 0.114]).astype(np.float64)
    else:
        a_gray = img_a.astype(np.float64)
        b_gray = img_b.astype(np.float64)

    # Constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Use block-based computation with stride
    window = 8
    h, w = a_gray.shape
    if h < window or w < window:
        # Fall back to whole-image comparison
        mu_a = np.mean(a_gray)
        mu_b = np.mean(b_gray)
        sigma_a = np.var(a_gray)
        sigma_b = np.var(b_gray)
        sigma_ab = np.mean((a_gray - mu_a) * (b_gray - mu_b))
        numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
        denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
        return float(numerator / denominator) if denominator > 0 else 0.0

    # Collect local statistics
    ssim_vals = []
    for y in range(0, h - window + 1, window):
        for x in range(0, w - window + 1, window):
            a_block = a_gray[y:y + window, x:x + window]
            b_block = b_gray[y:y + window, x:x + window]

            mu_a = np.mean(a_block)
            mu_b = np.mean(b_block)
            sigma_a = np.var(a_block)
            sigma_b = np.var(b_block)
            sigma_ab = np.mean((a_block - mu_a) * (b_block - mu_b))

            numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
            denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)

            if denominator > 0:
                ssim_vals.append(numerator / denominator)

    return float(np.mean(ssim_vals)) if ssim_vals else 0.0


def analyze_temporal_stability(
    mp4_path: str,
    threshold: float = 0.85,
    max_resolution: int = 256,
) -> dict:
    """Analyze temporal stability of an MP4 video.

    Args:
        mp4_path: Path to the MP4 file.
        threshold: SSIM below this value indicates a flicker frame.
        max_resolution: Downscale frames to this size for speed.
    Returns:
        {
            "temporal_stability_score": float,  # 0-1, mean SSIM
            "per_frame_metrics": [{"frame": int, "similarity": float}],
            "flicker_frames": [int],
            "min_similarity": float,
            "max_similarity": float,
            "std_similarity": float,
            "num_frames": int,
        }
    """
    from PIL import Image

    tmpdir = tempfile.mkdtemp(prefix="temporal_analysis_")
    try:
        # Extract frames to temp dir (don't clean up yet, we need them)
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir)
        output_pattern = os.path.join(frames_dir, "frame_%06d.png")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp4_path, "-q:v", "2", output_pattern],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

        frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
        if len(frame_paths) == 0:
            return {
                "temporal_stability_score": 1.0,
                "per_frame_metrics": [],
                "flicker_frames": [],
                "min_similarity": 1.0,
                "max_similarity": 1.0,
                "std_similarity": 0.0,
                "num_frames": 0,
            }

        # Load and downscale frames
        images = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            if max(img.size) > max_resolution:
                ratio = max_resolution / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            images.append(np.array(img, dtype=np.uint8))

        # Compute pairwise SSIM between consecutive frames
        per_frame = []
        for i in range(len(images) - 1):
            ssim_val = compute_ssim(images[i], images[i + 1])
            per_frame.append({"frame": i, "similarity": ssim_val})

        similarities = [p["similarity"] for p in per_frame]
        flicker_frames = [p["frame"] for p in per_frame if p["similarity"] < threshold]

        if similarities:
            mean_ssim = float(np.mean(similarities))
            min_ssim = float(np.min(similarities))
            max_ssim = float(np.max(similarities))
            std_ssim = float(np.std(similarities))
        else:
            mean_ssim = min_ssim = max_ssim = 1.0
            std_ssim = 0.0

        return {
            "temporal_stability_score": mean_ssim,
            "per_frame_metrics": per_frame,
            "flicker_frames": flicker_frames,
            "min_similarity": min_ssim,
            "max_similarity": max_ssim,
            "std_similarity": std_ssim,
            "num_frames": len(images),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
