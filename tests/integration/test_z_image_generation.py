"""Integration test: Z-Image character generation (Stage A)."""

import json

import pytest

from conftest import submit_workflow, get_output_images


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def z_image_result(comfyui):
    """Generate a character image and return the result."""
    workflow = {
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "6": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 1024, "batch_size": 1}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {
            "text": "Full body portrait, wide shot from head to toe. 90s OVA anime cel style. A young woman with long blue hair, red eyes, wearing a white school uniform with blue trim. Standing casually with arms at sides, slight smile. Clean white background. Detailed linework, vibrant colors.",
            "clip": ["6", 0],
        }},
        "11": {"class_type": "KSampler", "inputs": {
            "seed": 42, "steps": 8, "cfg": 1.0,
            "sampler_name": "res_multistep", "scheduler": "simple",
            "denoise": 1.0,
            "model": ["3", 0], "positive": ["10", 0], "negative": ["10", 0],
            "latent_image": ["5", 0],
        }},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["12", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": "pytest/z_image_test", "images": ["13", 0]}},
    }
    return submit_workflow(workflow, timeout=300)


def test_z_image_generates_image(z_image_result):
    """Z-Image should produce at least one output image."""
    images = get_output_images(z_image_result["outputs"], "14")
    assert len(images) > 0, "No images generated"


def test_z_image_reasonable_size(z_image_result):
    """Generated image should be reasonably sized (not blank/corrupt)."""
    images = get_output_images(z_image_result["outputs"], "14")
    assert len(images[0]) > 10000, f"Image too small: {len(images[0])} bytes"


def test_z_image_dimensions(z_image_result):
    """Generated image should match requested dimensions (768x1024)."""
    from PIL import Image
    import io

    images = get_output_images(z_image_result["outputs"], "14")
    img = Image.open(io.BytesIO(images[0]))
    assert img.size == (768, 1024), f"Unexpected dimensions: {img.size}"
