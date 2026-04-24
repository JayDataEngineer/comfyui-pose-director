"""Shared ComfyUI workflow builders for the Pose Director pipeline."""

DEFAULT_CHARACTER_PROMPT = (
    "Full body portrait, wide shot from head to toe. 90s OVA anime cel style. "
    "A young woman with long blue hair, red eyes, wearing a white school uniform with blue trim. "
    "Standing casually with arms at sides, slight smile. Clean white background. "
    "Detailed linework, vibrant colors."
)

VNCCS_PROMPT = "Draw character from image2"
VNCCS_INSTRUCTION = (
    "Modify the character's pose to match the skeleton guide shown in Picture 2. "
    "Maintain the character's exact appearance, clothing, and identity."
)

DEFAULT_LTX_PROMPT = (
    "A cute anime girl performing a playful dance, waving arms and swaying. "
    "Smooth natural motion. Clean scene with no text, letters, signs, or watermarks."
)


def build_character_workflow(
    seed: int = 42,
    prompt: str = DEFAULT_CHARACTER_PROMPT,
    width: int = 768,
    height: int = 1024,
    filename_prefix: str = "pipeline/character",
) -> dict:
    """Build a Z-Image Turbo character generation workflow."""
    return {
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "6": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["6", 0]}},
        "11": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 8, "cfg": 1.0,
            "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0,
            "model": ["3", 0], "positive": ["10", 0], "negative": ["10", 0],
            "latent_image": ["5", 0],
        }},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["12", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["13", 0]}},
    }


def build_pose_extraction_workflow(
    image_path: str,
    resolution: int = 1024,
    detect_hand: str = "enable",
    detect_body: str = "enable",
    detect_face: str = "disable",
    include_renderer: bool = False,
    filename_prefix_skeleton: str = "pipeline/skeleton",
    filename_prefix_openpose: str = "pipeline/openpose",
) -> dict:
    """Build an OpenPose extraction workflow.

    If include_renderer is True, includes PoseExtractor and PoseRenderer nodes
    to verify the pipeline.
    """
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_path, "upload": "image"}},
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": detect_hand, "detect_body": detect_body, "detect_face": detect_face,
            "resolution": resolution,
        }},
    }
    if include_renderer:
        workflow["3"] = {"class_type": "PoseExtractor", "inputs": {"pose_keypoint": ["2", 1], "person_index": 0}}
        workflow["4"] = {"class_type": "PoseRenderer", "inputs": {
            "keypoints": ["3", 0],
            "width": resolution,
            "height": resolution,
            "line_width": 4,
            "point_radius": 6,
        }}
        workflow["5"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix_skeleton, "images": ["4", 0]}}
        workflow["6"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix_openpose, "images": ["2", 0]}}
    else:
        workflow["3"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix_openpose, "images": ["2", 0]}}
    return workflow


def build_motion_director_workflow(
    image_path: str,
    motion_description: str,
    num_keyframes: int,
    total_frames: int,
    llm_url: str,
    llm_key: str,
    llm_model: str,
    width: int = 1024,
    height: int = 1024,
    keypoints: str = "[]",
    filename_prefix: str = "pipeline/keyframes",
) -> dict:
    """Build a MotionDirector keyframe generation workflow."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_path, "upload": "image"}},
        "2": {"class_type": "OpenposePreprocessor", "inputs": {
            "image": ["1", 0],
            "detect_hand": "enable", "detect_body": "enable", "detect_face": "disable",
            "resolution": 1024,
        }},
        "3": {"class_type": "MotionDirector", "inputs": {
            "keypoints": keypoints,
            "motion_description": motion_description,
            "num_keyframes": num_keyframes,
            "total_frames": total_frames,
            "base_url": llm_url,
            "api_key": llm_key,
            "model": llm_model,
            "width": width,
            "height": height,
            "pose_keypoint": ["2", 1],
        }},
        "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["3", 0]}},
    }


def build_vnccs_edit_workflow(
    character_image_path: str,
    keyframe_image_path: str,
    seed: int = 42,
    steps: int = 40,
    use_gguf_unet: bool = True,
    unet_name: str = "Qwen-Image-Edit-2511-Q5_0.gguf",
    ckpt_name: str = "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui.safetensors",
    clip_name: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    vae_name: str = "qwen_image_vae.safetensors",
    target_size: int = 1024,
    vl_size: int = 384,
    prompt: str = VNCCS_PROMPT,
    instruction: str = VNCCS_INSTRUCTION,
    filename_prefix: str = "pipeline/edited_kf",
) -> dict:
    """Build a VNCCS QWEN Encoder editing workflow for a single keyframe."""
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": character_image_path, "upload": "image"}},
        "21": {"class_type": "LoadImage", "inputs": {"image": keyframe_image_path}},
    }

    if use_gguf_unet:
        workflow["4"] = {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": unet_name},
        }
    else:
        workflow["4"] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name},
        }

    workflow["5"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": clip_name, "type": "qwen_image"},
    }
    workflow["6"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": vae_name},
    }

    workflow["7"] = {
        "class_type": "VNCCS_QWEN_Encoder",
        "inputs": {
            "prompt": prompt,
            "latent_image_index": 1,
            "image3": None,
            "image1_name": "Character Reference",
            "image2_name": "Pose Guide",
            "image3_name": "Picture 3",
            "target_size": target_size,
            "upscale_method": "lanczos",
            "crop_method": "center",
            "weight1": 1.0,
            "weight2": 1.0,
            "weight3": 1.0,
            "vl_size": vl_size,
            "instruction": instruction,
            "qwen_2511": True,
            "clip": ["5", 0],
            "vae": ["6", 0],
            "image1": ["1", 0],
            "image2": ["21", 0],
        },
    }

    workflow["8"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["7", 0],
            "negative": ["7", 1],
            "latent_image": ["7", 2],
        },
    }

    workflow["9"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["8", 0], "vae": ["6", 0]},
    }

    workflow["10"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": filename_prefix, "images": ["9", 0]},
    }

    return workflow


def build_ltx_video_workflow(
    keyframe_image_paths: list[str],
    seed: int = 42,
    width: int = 256,
    height: int = 256,
    length: int = 9,
    steps: int = 20,
    fps: int = 24,
    prompt: str = DEFAULT_LTX_PROMPT,
    unet_name: str = "ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf",
    clip_name1: str = "gemma-3-12b-it-IQ4_XS.gguf",
    clip_name2: str = "ltxv_text_projection_dual.safetensors",
    vae_name: str = "LTX23_video_vae_bf16.safetensors",
    filename_prefix: str = "pipeline/output",
    video_combine_class: str = "VHS_VideoCombine",
) -> dict:
    """Build an LTX Video 2.3 assembly workflow with keyframe guides."""
    workflow = {}

    # Load keyframes
    for i, path in enumerate(keyframe_image_paths):
        workflow[str(1 + i)] = {
            "class_type": "LoadImage",
            "inputs": {"image": path},
        }

    # Load models
    workflow["5"] = {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": unet_name},
    }
    workflow["6"] = {
        "class_type": "DualCLIPLoaderGGUF",
        "inputs": {
            "clip_name1": clip_name1,
            "clip_name2": clip_name2,
            "type": "ltxv",
        },
    }
    workflow["7"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": vae_name},
    }

    # Text prompt
    workflow["8"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": prompt, "clip": ["6", 0]},
    }

    # Latent video
    workflow["9"] = {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {"width": width, "height": height, "length": length, "batch_size": 1},
    }

    # Guide keyframes
    prev_node = "9"
    for i, _ in enumerate(keyframe_image_paths):
        frame_idx = i * 8  # 0, 8, 16, 24, ...
        node_id = str(10 + i)
        if i == 0:
            workflow[node_id] = {
                "class_type": "LTXVAddGuideAdvanced",
                "inputs": {
                    "frame_idx": frame_idx,
                    "strength": 1.0,
                    "crf": 29,
                    "blur_radius": 0,
                    "interpolation": "lanczos",
                    "crop": "disabled",
                    "positive": ["8", 0],
                    "negative": ["8", 0],
                    "vae": ["7", 0],
                    "latent": ["9", 0],
                    "image": [str(1 + i), 0],
                },
            }
        else:
            workflow[node_id] = {
                "class_type": "LTXVAddGuideAdvanced",
                "inputs": {
                    "frame_idx": frame_idx,
                    "strength": 1.0,
                    "crf": 29,
                    "blur_radius": 0,
                    "interpolation": "lanczos",
                    "crop": "disabled",
                    "positive": [prev_node, 0],
                    "negative": [prev_node, 1],
                    "vae": ["7", 0],
                    "latent": [prev_node, 2],
                    "image": [str(1 + i), 0],
                },
            }
        prev_node = node_id

    # Sample
    last_guide = str(10 + len(keyframe_image_paths) - 1)
    workflow["14"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["5", 0],
            "positive": [last_guide, 0],
            "negative": [last_guide, 1],
            "latent_image": [last_guide, 2],
        },
    }

    # Decode
    workflow["15"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["14", 0], "vae": ["7", 0]},
    }

    # Combine to video
    workflow["16"] = {
        "class_type": video_combine_class,
        "inputs": {
            "filename_prefix": filename_prefix,
            "frame_rate": fps,
            "loop_count": 0,
            "format": "video/h264-mp4",
            "pingpong": False,
            "save_output": True,
            "images": ["15", 0],
        },
    }

    return workflow
