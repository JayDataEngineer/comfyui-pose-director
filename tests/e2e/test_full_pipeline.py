"""E2E test: Full pipeline from character generation to video output.

Stages (each is a separate ComfyUI workflow with restart between stages):
  A. Generate character (Z-Image Turbo)
  C. Extract pose → Generate keyframes (MotionDirector via local LLM)
  D. Edit ALL keyframes with VNCCS QWEN Encoder (matches 03_static_pose workflow)
  E. Assemble video with LTX Video 2.3 using edited character images as guides

Skipped unless RUN_E2E=1 env var is set.
"""

import io

import pytest
from PIL import Image

from conftest import (
    submit_workflow, get_output_images, get_output_video, upload_image,
    restart_comfyui,
)
from pipeline.workflows import (
    build_character_workflow,
    build_motion_director_workflow,
    build_vnccs_edit_workflow,
    build_ltx_video_workflow,
    VNCCS_PROMPT,
    VNCCS_INSTRUCTION,
)

pytestmark = [pytest.mark.e2e, pytest.mark.requires_api_key]


@pytest.fixture(scope="module")
def e2e_result(comfyui, llm_config):
    """Run the full E2E pipeline and return the final video workflow result."""
    llm_url, llm_model, llm_key = llm_config

    # ── Step A: Generate character ──────────────────────────────────────
    print("\n[E2E] Step A: Generating character with Z-Image Turbo...")
    char_workflow = build_character_workflow(
        seed=42,
        prompt=(
            "Full body portrait, wide shot from head to toe. 90s OVA anime cel style. "
            "A cute young woman with long flowing pink hair, big sparkling blue eyes, "
            "wearing a cute sailor school uniform with pink ribbon. Standing with a cheerful smile, "
            "hands clasped together. Clean white background. Detailed linework, vibrant pastel colors."
        ),
        width=768,
        height=1024,
        filename_prefix="e2e/character",
    )
    char_result = submit_workflow(char_workflow, timeout=600, retries=2)
    char_images = get_output_images(char_result["outputs"], "14")
    assert len(char_images) > 0, "E2E: No character image generated"
    assert len(char_images[0]) > 10000, f"E2E: Character image too small: {len(char_images[0])} bytes"
    upload_image(char_images[0], "e2e_character.png", "PoseDirector")
    print(f"[E2E] Step A done: character image {len(char_images[0])} bytes")

    # Restart ComfyUI to clear Z-Image model from VRAM
    restart_comfyui()

    # ── Step C: Extract pose + generate keyframes ──────────────────────
    print("[E2E] Step C: Extracting pose and generating motion keyframes...")
    kf_workflow = build_motion_director_workflow(
        image_path="PoseDirector/e2e_character.png",
        motion_description="Character performs a cute anime dance, waving arms and shifting weight side to side with playful poses",
        num_keyframes=3,
        total_frames=25,
        llm_url=llm_url,
        llm_key=llm_key,
        llm_model=llm_model,
        width=1024,
        height=1024,
        filename_prefix="e2e/keyframes",
    )
    kf_result = submit_workflow(kf_workflow, timeout=600, retries=2)
    kf_images = get_output_images(kf_result["outputs"], "4")
    assert len(kf_images) > 0, "E2E: No keyframe images generated"
    print(f"[E2E] Step C done: {len(kf_images)} keyframe image(s)")

    # Upload each keyframe individually so Step D can load them by name
    for i, kf_bytes in enumerate(kf_images):
        upload_image(kf_bytes, f"e2e_kf_{i}.png", "PoseDirector")

    # Restart ComfyUI to clear OpenPose + MotionDirector from VRAM before loading Qwen Edit
    restart_comfyui()

    # ── Step D: Edit ALL keyframes with VNCCS QWEN Encoder ────────────
    edited_keyframes = []
    for kf_idx in range(len(kf_images)):
        print(f"[E2E] Step D: Editing keyframe {kf_idx} with VNCCS QWEN Encoder...")
        edit_workflow = build_vnccs_edit_workflow(
            character_image_path="PoseDirector/e2e_character.png",
            keyframe_image_path=f"PoseDirector/e2e_kf_{kf_idx}.png",
            seed=42,
            steps=40,
            prompt=VNCCS_PROMPT,
            instruction=VNCCS_INSTRUCTION,
            filename_prefix=f"e2e/edited_kf{kf_idx}",
        )
        edit_result = submit_workflow(edit_workflow, timeout=900, retries=2)
        edited_imgs = get_output_images(edit_result["outputs"], "10")
        assert len(edited_imgs) > 0, f"E2E: No edited keyframe {kf_idx} generated"
        assert len(edited_imgs[0]) > 10000, f"E2E: Edited keyframe {kf_idx} too small: {len(edited_imgs[0])} bytes"
        edited_keyframes.append(edited_imgs[0])
        print(f"[E2E] Step D: keyframe {kf_idx} edited — {len(edited_imgs[0])} bytes")

    print(f"[E2E] Step D done: {len(edited_keyframes)} keyframes edited")

    # Restart ComfyUI to clear Qwen Edit models from VRAM before LTX Video loads
    restart_comfyui()

    # ── Step E: Assemble video with LTX Video 2.3 ─────────────────────
    print("[E2E] Step E: Assembling video with LTX Video 2.3...")

    # Upload edited character images as LTX video guides (NOT raw skeletons)
    for i, kf_bytes in enumerate(edited_keyframes):
        img = Image.open(io.BytesIO(kf_bytes)).convert("RGB")
        img = img.resize((256, 256), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        upload_image(buf.getvalue(), f"edited_kf_{i}.png", "PoseDirector")
        print(f"[E2E] Step E: uploaded edited keyframe {i} ({len(kf_bytes)} bytes)")

    video_workflow = build_ltx_video_workflow(
        keyframe_image_paths=[f"PoseDirector/edited_kf_{i}.png" for i in range(len(edited_keyframes))],
        seed=42,
        width=256,
        height=256,
        length=9,
        steps=20,
        fps=24,
        filename_prefix="e2e/output",
    )
    video_result = submit_workflow(video_workflow, timeout=900, retries=2)
    print("[E2E] Step E done")
    return video_result


def test_e2e_produces_video(e2e_result):
    """E2E pipeline should produce a video file."""
    video = get_output_video(e2e_result["outputs"], "16")
    assert video is not None, "No video output from E2E pipeline"
    assert len(video) > 10000, f"Video too small: {len(video)} bytes"


def test_e2e_video_is_mp4(e2e_result):
    """The output video should be a valid MP4 file."""
    video = get_output_video(e2e_result["outputs"], "16")
    if video is None:
        pytest.skip("No video output")
    assert len(video) > 8
    assert b'ftyp' in video[:32], "Video doesn't appear to be a valid MP4"
