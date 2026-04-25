"""Unit tests for pipeline/workflows.py — workflow builder functions."""

import pytest

from pipeline.workflows import (
    build_character_workflow,
    build_pose_extraction_workflow,
    build_motion_director_workflow,
    build_vnccs_edit_workflow,
    build_ltx_video_workflow,
    VNCCS_PROMPT,
    VNCCS_INSTRUCTION,
    DEFAULT_LTX_PROMPT,
)


class TestBuildCharacterWorkflow:
    """Tests for build_character_workflow()."""

    def test_returns_valid_dict(self):
        wf = build_character_workflow()
        assert isinstance(wf, dict)
        assert len(wf) > 0

    def test_has_required_nodes(self):
        wf = build_character_workflow()
        node_types = {v["class_type"] for v in wf.values()}
        assert "UNETLoader" in node_types
        assert "CLIPLoader" in node_types
        assert "KSampler" in node_types
        assert "VAEDecode" in node_types
        assert "SaveImage" in node_types

    def test_custom_params(self):
        wf = build_character_workflow(seed=99, width=512, height=768, prompt="test prompt")
        # Find KSampler
        ks = [v for v in wf.values() if v["class_type"] == "KSampler"][0]
        assert ks["inputs"]["seed"] == 99
        # Find latent
        latent = [v for v in wf.values() if v["class_type"] == "EmptySD3LatentImage"][0]
        assert latent["inputs"]["width"] == 512
        assert latent["inputs"]["height"] == 768

    def test_filename_prefix(self):
        wf = build_character_workflow(filename_prefix="test/char")
        save = [v for v in wf.values() if v["class_type"] == "SaveImage"][0]
        assert save["inputs"]["filename_prefix"] == "test/char"


class TestBuildPoseExtractionWorkflow:
    """Tests for build_pose_extraction_workflow()."""

    def test_basic_no_renderer(self):
        wf = build_pose_extraction_workflow(image_path="test/img.png")
        node_types = {v["class_type"] for v in wf.values()}
        assert "OpenposePreprocessor" in node_types
        assert "PoseExtractor" not in node_types

    def test_with_renderer(self):
        wf = build_pose_extraction_workflow(
            image_path="test/img.png", include_renderer=True
        )
        node_types = {v["class_type"] for v in wf.values()}
        assert "PoseExtractor" in node_types
        assert "PoseRenderer" in node_types

    def test_custom_resolution(self):
        wf = build_pose_extraction_workflow(image_path="test/img.png", resolution=512)
        opp = [v for v in wf.values() if v["class_type"] == "OpenposePreprocessor"][0]
        assert opp["inputs"]["resolution"] == 512


class TestBuildMotionDirectorWorkflow:
    """Tests for build_motion_director_workflow()."""

    def test_has_required_nodes(self):
        wf = build_motion_director_workflow(
            image_path="test/img.png",
            motion_description="wave",
            num_keyframes=3,
            total_frames=25,
            llm_url="http://test/v1",
            llm_model="m",
        )
        node_types = {v["class_type"] for v in wf.values()}
        assert "MotionDirector" in node_types
        assert "OpenposePreprocessor" in node_types
        assert "SaveImage" in node_types

    def test_llm_params_propagated(self):
        wf = build_motion_director_workflow(
            image_path="test/img.png",
            motion_description="dance",
            num_keyframes=4,
            total_frames=32,
            llm_url="http://llm:8000/v1",
            llm_model="big-model",
        )
        md = [v for v in wf.values() if v["class_type"] == "MotionDirector"][0]
        assert md["inputs"]["base_url"] == "http://llm:8000/v1"
        assert md["inputs"]["model"] == "big-model"
        assert md["inputs"]["num_keyframes"] == 4

    def test_pose_keypoint_linked(self):
        wf = build_motion_director_workflow(
            image_path="test/img.png",
            motion_description="wave",
            num_keyframes=3,
            total_frames=25,
            llm_url="u", llm_model="m",
        )
        md = [v for v in wf.values() if v["class_type"] == "MotionDirector"][0]
        assert md["inputs"]["pose_keypoint"] == ["2", 1]


class TestBuildVNCCSEditWorkflow:
    """Tests for build_vnccs_edit_workflow()."""

    def test_has_required_nodes(self):
        wf = build_vnccs_edit_workflow(
            character_image_path="char.png",
            keyframe_image_path="kf.png",
        )
        node_types = {v["class_type"] for v in wf.values()}
        assert "VNCCS_QWEN_Encoder" in node_types
        assert "KSampler" in node_types
        assert "VAEDecode" in node_types
        assert "SaveImage" in node_types

    def test_gguf_unet_by_default(self):
        wf = build_vnccs_edit_workflow(
            character_image_path="char.png",
            keyframe_image_path="kf.png",
        )
        unet_nodes = [v for v in wf.values() if v["class_type"] == "UnetLoaderGGUF"]
        assert len(unet_nodes) == 1
        assert unet_nodes[0]["inputs"]["unet_name"] == "Qwen-Image-Edit-2511-Q5_0.gguf"

    def test_checkpoint_fallback(self):
        wf = build_vnccs_edit_workflow(
            character_image_path="char.png",
            keyframe_image_path="kf.png",
            use_gguf_unet=False,
        )
        ckpt_nodes = [v for v in wf.values() if v["class_type"] == "CheckpointLoaderSimple"]
        assert len(ckpt_nodes) == 1

    def test_vnccs_params(self):
        wf = build_vnccs_edit_workflow(
            character_image_path="char.png",
            keyframe_image_path="kf.png",
            steps=40,
            seed=123,
            target_size=512,
            vl_size=256,
        )
        vnccs = [v for v in wf.values() if v["class_type"] == "VNCCS_QWEN_Encoder"][0]
        assert vnccs["inputs"]["image1_name"] == "Character Reference"
        assert vnccs["inputs"]["image2_name"] == "Pose Guide"
        assert vnccs["inputs"]["target_size"] == 512
        assert vnccs["inputs"]["vl_size"] == 256
        assert vnccs["inputs"]["weight1"] == 1.0
        assert vnccs["inputs"]["weight2"] == 1.0
        assert vnccs["inputs"]["qwen_2511"] is True

        ks = [v for v in wf.values() if v["class_type"] == "KSampler"][0]
        assert ks["inputs"]["steps"] == 40
        assert ks["inputs"]["seed"] == 123

    def test_image_links(self):
        wf = build_vnccs_edit_workflow(
            character_image_path="PoseDirector/char.png",
            keyframe_image_path="PoseDirector/kf.png",
        )
        vnccs = [v for v in wf.values() if v["class_type"] == "VNCCS_QWEN_Encoder"][0]
        # image1 -> LoadImage node 1, image2 -> LoadImage node 21
        assert vnccs["inputs"]["image1"] == ["1", 0]
        assert vnccs["inputs"]["image2"] == ["21", 0]

    def test_default_prompt_and_instruction(self):
        wf = build_vnccs_edit_workflow(
            character_image_path="c.png",
            keyframe_image_path="k.png",
        )
        vnccs = [v for v in wf.values() if v["class_type"] == "VNCCS_QWEN_Encoder"][0]
        assert vnccs["inputs"]["prompt"] == VNCCS_PROMPT
        assert vnccs["inputs"]["instruction"] == VNCCS_INSTRUCTION


class TestBuildLtxVideoWorkflow:
    """Tests for build_ltx_video_workflow()."""

    def test_single_keyframe(self):
        wf = build_ltx_video_workflow(
            keyframe_image_paths=["kf0.png"],
        )
        node_types = {v["class_type"] for v in wf.values()}
        assert "UnetLoaderGGUF" in node_types
        assert "DualCLIPLoaderGGUF" in node_types
        assert "VAELoader" in node_types
        assert "EmptyLTXVLatentVideo" in node_types
        assert "LTXVAddGuideAdvanced" in node_types

    def test_multiple_keyframes(self):
        wf = build_ltx_video_workflow(
            keyframe_image_paths=["kf0.png", "kf1.png", "kf2.png"],
        )
        guides = [v for v in wf.values() if v["class_type"] == "LTXVAddGuideAdvanced"]
        assert len(guides) == 3

    def test_frame_indices(self):
        wf = build_ltx_video_workflow(
            keyframe_image_paths=["kf0.png", "kf1.png", "kf2.png"],
        )
        guides = [(k, v) for k, v in wf.items() if v["class_type"] == "LTXVAddGuideAdvanced"]
        frames = [v["inputs"]["frame_idx"] for _, v in sorted(guides, key=lambda x: int(x[0]))]
        assert frames == [0, 8, 16]

    def test_uses_full_vae(self):
        wf = build_ltx_video_workflow(keyframe_image_paths=["kf.png"])
        vae = [v for v in wf.values() if v["class_type"] == "VAELoader"][0]
        assert vae["inputs"]["vae_name"] == "LTX23_video_vae_bf16.safetensors"

    def test_cfg_is_one(self):
        """Distilled model uses CFG=1.0."""
        wf = build_ltx_video_workflow(keyframe_image_paths=["kf.png"])
        ks = [v for v in wf.values() if v["class_type"] == "KSampler"][0]
        assert ks["inputs"]["cfg"] == 1.0

    def test_default_prompt_has_no_text_instruction(self):
        """Prompt should contain anti-text instruction."""
        assert "no text" in DEFAULT_LTX_PROMPT.lower()

    def test_video_combine_node(self):
        wf = build_ltx_video_workflow(keyframe_image_paths=["kf.png"])
        vhs = [v for v in wf.values() if v["class_type"] == "VHS_VideoCombine"]
        assert len(vhs) == 1
        assert vhs[0]["inputs"]["format"] == "video/h264-mp4"


class TestSharedConstants:
    """Tests for shared prompt constants."""

    def test_vnccs_prompt(self):
        assert "image2" in VNCCS_PROMPT.lower()

    def test_vnccs_instruction(self):
        assert "skeleton" in VNCCS_INSTRUCTION.lower() or "pose" in VNCCS_INSTRUCTION.lower()
