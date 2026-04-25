"""E2E test: Full pipeline from character generation to video output.

Stages (each is a separate ComfyUI workflow with restart between stages):
  A. Generate character (Z-Image Turbo) — workflows/00_z_image_character.json
  C. Generate keyframes (MotionDirector in rotation mode → Anny 3D body mesh renders)
  D. Edit ALL keyframes with VNCCS QWEN Encoder (body mesh as image2)
  E. Assemble video with LTX Video 2.3 using edited character images as guides

Skipped unless RUN_E2E=1 env var is set.
"""

import io
import tempfile

import pytest
from PIL import Image

from conftest import (
    submit_workflow, get_output_images, get_output_video, upload_image,
    restart_comfyui, load_ui_workflow, patch_ui_workflow, patch_connection,
    convert_ui_to_api, WORKFLOWS_DIR,
)

pytestmark = [pytest.mark.e2e, pytest.mark.requires_api_key]


@pytest.fixture(scope="module")
def e2e_result(comfyui, llm_config):
    """Run the full E2E pipeline and return the final video workflow result."""
    llm_url, llm_model, llm_key = llm_config

    # ── Step A: Generate character ──────────────────────────────────────
    print("\n[E2E] Step A: Generating character with Z-Image Turbo...")
    char_workflow = load_ui_workflow(WORKFLOWS_DIR / "00_z_image_character.json")
    # Fix missing connection if needed
    save_node = next((n for n in char_workflow["nodes"] if n["type"] == "SaveImage"), None)
    vaedecode_node = next((n for n in char_workflow["nodes"] if n["type"] == "VAEDecode"), None)
    if save_node and vaedecode_node and save_node.get("inputs", [{}])[0].get("link") is None:
        patch_connection(
            char_workflow,
            target_node_id=save_node["id"],
            target_input_name="images",
            source_node_id=vaedecode_node["id"],
            source_output_index=0,
        )
    patch_ui_workflow(char_workflow, {
        "CLIPTextEncode": {"text": (
            "Full body portrait, wide shot from head to toe. 90s OVA anime cel style. "
            "A cute young woman with long flowing pink hair, big sparkling blue eyes, "
            "wearing a cute sailor school uniform with pink ribbon. Standing with a cheerful smile, "
            "hands clasped together. Clean white background. Detailed linework, vibrant pastel colors."
        )},
        "EmptySD3LatentImage": {"width": 768, "height": 1024},
        "KSampler": {"seed": 42},
        "SaveImage": {"filename_prefix": "e2e/character"},
    })
    char_api = convert_ui_to_api(char_workflow)
    char_result = submit_workflow(char_api, timeout=600, retries=2)
    char_images = get_output_images(char_result["outputs"], str(save_node["id"]))
    assert len(char_images) > 0, "E2E: No character image generated"
    assert len(char_images[0]) > 10000, f"E2E: Character image too small: {len(char_images[0])} bytes"
    upload_image(char_images[0], "e2e_character.png", "PoseDirector")
    print(f"[E2E] Step A done: character image {len(char_images[0])} bytes")

    # Restart ComfyUI to clear Z-Image model from VRAM
    restart_comfyui()

    # ── Step C: Generate keyframes via rotation mode (Anny body mesh) ──
    print("[E2E] Step C: Generating motion keyframes (rotation mode → Anny body mesh)...")
    kf_workflow = load_ui_workflow(WORKFLOWS_DIR / "01_motion_keyframes_rotation.json")
    patch_ui_workflow(kf_workflow, {
        "LoadImage": {"image": "PoseDirector/e2e_character.png"},
        "MotionDirector": {
            "motion_description": "Character performs a cute anime dance, waving arms and shifting weight side to side with playful poses",
            "num_keyframes": 3,
            "total_frames": 25,
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
        },
        "PreviewImage": {"filename_prefix": "e2e/keyframes"},
    })
    kf_api = convert_ui_to_api(kf_workflow)
    kf_result = submit_workflow(kf_api, timeout=600, retries=2)
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
        edit_workflow = load_ui_workflow(WORKFLOWS_DIR / "02_vnccs_edit_keyframe.json")
        patch_ui_workflow(edit_workflow, {
            "LoadImage": [
                {"image": "PoseDirector/e2e_character.png"},
                {"image": f"PoseDirector/e2e_kf_{kf_idx}.png"},
            ],
            "KSampler": {"seed": 42, "steps": 40},
            "VNCCS_QWEN_Encoder": {
                "prompt": "Draw character from image2",
                "instruction": (
                    "Modify the character's pose to match the body shape and pose shown in Picture 2. "
                    "Picture 2 shows a 3D body mesh rendering — match the posture, limb positions, and overall body orientation. "
                    "Maintain the character's exact appearance, clothing, and identity."
                ),
            },
            "SaveImage": {"filename_prefix": f"e2e/edited_kf{kf_idx}"},
        })
        edit_api = convert_ui_to_api(edit_workflow)
        edit_result = submit_workflow(edit_api, timeout=900, retries=2)
        save_node = next(n for n in edit_workflow["nodes"] if n["type"] == "SaveImage")
        edited_imgs = get_output_images(edit_result["outputs"], str(save_node["id"]))
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

    video_workflow = load_ui_workflow(WORKFLOWS_DIR / "04_ltx_video_assembly.json")
    loadimage_patches = [
        {"image": f"PoseDirector/edited_kf_{i}.png"}
        for i in range(len(edited_keyframes))
    ]
    patch_ui_workflow(video_workflow, {
        "LoadImage": loadimage_patches,
        "CLIPTextEncode": {"text": (
            "A cute anime girl performing a playful dance, waving arms and swaying. "
            "Smooth natural motion. Clean scene with no text, letters, signs, or watermarks."
        )},
        "EmptyLTXVLatentVideo": {"width": 256, "height": 256, "length": 9},
        "KSampler": {"seed": 42, "steps": 20},
        "VHS_VideoCombine": {"filename_prefix": "e2e/output"},
    })
    video_api = convert_ui_to_api(video_workflow)
    video_result = submit_workflow(video_api, timeout=900, retries=2)
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


def test_e2e_temporal_stability(e2e_result):
    """Video should have reasonable temporal stability (low flicker)."""
    video = get_output_video(e2e_result["outputs"], "16")
    if video is None:
        pytest.skip("No video output")

    # Write video to temp file for ffmpeg analysis
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video)
        tmp_path = f.name

    try:
        from pose_director_root.utils.temporal_analysis import analyze_temporal_stability
        result = analyze_temporal_stability(tmp_path, threshold=0.85)
        score = result["temporal_stability_score"]
        n_frames = result["num_frames"]
        flicker = result["flicker_frames"]
        print(f"[E2E] Temporal stability: score={score:.3f}, "
              f"frames={n_frames}, flicker={flicker}")
        assert score > 0.5, (
            f"Temporal stability too low: {score:.3f}"
        )
    finally:
        import os
        os.unlink(tmp_path)
