# Pose Director: Asset Generation Handoff

This document explains how to generate 2D game sprite assets using the Pose Director extension on the Tech Noir Ray cluster. Written for an AI agent that will programmatically create character sprites.

## Architecture

```
ComfyUI (port 18465) → Ray Serve proxy (port 18800) → Tailscale network
```

All API calls go through: `http://100.86.69.57:18800/comfyui`

## CRITICAL: Do Not Rebuild The Workflow

**Do NOT construct the workflow JSON from scratch.** A previous attempt to do this produced completely wrong results — different characters per frame, no consistency, hallucinated backgrounds. The VNCCS encoder is sensitive to exact parameter values.

**Use one of these two approaches:**

1. **Best**: Use the Python pipeline module (see below) — it loads the tested workflow JSON and patches only the fields that change.
2. **OK**: Copy the exact workflow JSON below and change ONLY the 4 marked fields. Do not touch anything else.

## Pipeline: Character → Posed Sprite (Transparent Background)

```
Character Image (768x1024 PNG)
        ↓
BodyMeshRenderer (pose spec → 3D mesh render)
        ↓
OpenposePreprocessor (mesh → OpenPose skeleton)
        ↓
VNCCS_QWEN_Encoder (character + mesh + skeleton → conditioned latents)
        ↓
KSampler (4-step Lightning diffusion)
        ↓
VAEDecode → VNCCS_RMBG2 (background removal, transparent output)
        ↓
SaveImage (1024x1024 RGBA PNG output)
```

### Step 1: Upload Character Image

```bash
curl -X POST http://100.86.69.57:18800/comfyui/upload/image \
  -F "image=@character.png" \
  -F "subfolder=PoseDirector" \
  -F "overwrite=true"
```

Response: `{"name": "PoseDirector/character.png", "subfolder": "PoseDirector", "type": "input"}`

### Step 2: Submit Workflow

POST the workflow JSON to `http://100.86.69.57:18800/comfyui/prompt`.

**Change ONLY these 4 fields:**

| Field | Location | What to set |
|-------|----------|-------------|
| `image` | Node 1 (LoadImage) | Uploaded filename, e.g. `"PoseDirector/character.png"` |
| `rotations_json` | Node 2 (BodyMeshRenderer) | Pose JSON string (see Pose Reference below) |
| `model_rotation_y` | Node 2 (BodyMeshRenderer) | Viewing angle: 0=front, 90=right, 180=back, 270=left |
| `seed` | Node 10 (KSampler) | Change for variations, same for reproducibility |

### Working Workflow JSON (with background removal)

```json
{
  "1": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "PoseDirector/character.png",
      "upload": "image"
    }
  },
  "2": {
    "class_type": "BodyMeshRenderer",
    "inputs": {
      "rotations_json": "{\"right_shoulder\": [0, 0, 90], \"left_shoulder\": [0, 0, -90]}",
      "width": 1024,
      "height": 1024,
      "model_rotation_y": 0.0,
      "backend": "auto"
    }
  },
  "3": {
    "class_type": "OpenposePreprocessor",
    "inputs": {
      "image": ["2", 0],
      "detect_hand": "enable",
      "detect_body": "enable",
      "detect_face": "enable",
      "resolution": 1024,
      "scale_stick_for_xinsr_cn": "enable"
    }
  },
  "4": {
    "class_type": "UnetLoaderGGUF",
    "inputs": {
      "unet_name": "Qwen-Image-Edit-2511-Q5_0.gguf"
    }
  },
  "5": {
    "class_type": "CLIPLoader",
    "inputs": {
      "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
      "type": "qwen_image"
    }
  },
  "6": {
    "class_type": "VAELoader",
    "inputs": {
      "vae_name": "qwen_image_vae.safetensors"
    }
  },
  "7": {
    "class_type": "LoraLoaderModelOnly",
    "inputs": {
      "model": ["4", 0],
      "lora_name": "qwen/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
      "strength_model": 1.0
    }
  },
  "8": {
    "class_type": "LoraLoaderModelOnly",
    "inputs": {
      "model": ["7", 0],
      "lora_name": "qwen/VNCCS/VNCCS_PoseStudioQIE2511_V2.safetensors",
      "strength_model": 1.0
    }
  },
  "9": {
    "class_type": "VNCCS_QWEN_Encoder",
    "inputs": {
      "clip": ["5", 0],
      "prompt": "Draw character from image2",
      "vae": ["6", 0],
      "latent_image_index": 1,
      "image1": ["2", 0],
      "image2": ["1", 0],
      "image3": ["3", 0],
      "image1_name": "Pose Guide",
      "image2_name": "Character Reference",
      "image3_name": "OpenPose",
      "target_size": 1024,
      "upscale_method": "lanczos",
      "crop_method": "center",
      "weight1": 1.0,
      "weight2": 1.0,
      "weight3": 1.0,
      "vl_size": 384,
      "instruction": "Match the body pose shown in Picture 1 (3D body mesh). Picture 2 is the character to draw. Picture 3 shows the skeleton overlay. Replicate the exact pose, limb positions, and body orientation from Picture 1 while maintaining the character's identity, clothing, and appearance.",
      "qwen_2511": true
    }
  },
  "10": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["8", 0],
      "seed": 42,
      "steps": 4,
      "cfg": 1.0,
      "sampler_name": "euler",
      "scheduler": "simple",
      "positive": ["9", 0],
      "negative": ["9", 1],
      "latent_image": ["9", 2],
      "denoise": 1.0
    }
  },
  "11": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["10", 0],
      "vae": ["6", 0]
    }
  },
  "13": {
    "class_type": "VNCCS_RMBG2",
    "inputs": {
      "image": ["11", 0],
      "model": "BEN2",
      "sensitivity": 1.0,
      "process_res": 1024,
      "mask_blur": 4,
      "mask_offset": 0,
      "invert_output": false,
      "refine_foreground": true,
      "background": "Alpha"
    }
  },
  "12": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["13", 0],
      "filename_prefix": "PoseDirector/sprite"
    }
  }
}
```

### What NOT to change

These parameters are critical for VNCCS consistency. Changing them will produce wrong results:

- **Node 9 image ordering**: image1=mesh (node 2), image2=character (node 1), image3=openpose (node 3). VNCCS reads image2 as the character reference.
- **Node 9 instruction**: The text tells VNCCS which picture is what. If image1/image2 are swapped without changing the instruction, you get a different character.
- **Node 9 qwen_2511**: Must be `true`. Without it, VNCCS uses the old encoder which has worse consistency.
- **Nodes 7-8 LoRAs**: Both Lightning 4-step and VNCCS PoseStudio LoRAs are required. Missing either produces garbage.
- **Node 10 steps**: Must be 4 with the Lightning LoRA. More steps = worse results, not better.
- **Node 13 VNCCS_RMBG2**: BEN2 model with Alpha background for transparent output.

### Step 3: Poll for Completion

```bash
# Get prompt_id from the /prompt response, then poll:
curl http://100.86.69.57:18800/comfyui/history/{prompt_id}
```

When `status.completed` is true, check `outputs` for image filenames. Each frame takes ~30 seconds.

### Step 4: Download Output

```bash
curl "http://100.86.69.57:18800/comfyui/view?filename=PoseDirector_sprite_01.png&type=output" \
  -o sprite.png
```

## Pose Reference (rotations_json)

Joints use Euler angles in degrees `[x, y, z]` from rest pose. BodyMeshRenderer uses the Anny 3D body model.

### Available Joints

| Joint | X-axis | Y-axis | Z-axis |
|-------|--------|--------|--------|
| `spine` | Lean forward/back | Twist torso | Side bend |
| `right_shoulder` | Forward/back swing | Rotate arm | Raise/lower (**+90 = arm UP**) |
| `left_shoulder` | Forward/back swing | Rotate arm | Raise/lower (**-90 = arm UP**) |
| `right_elbow` | — | — | **-90 = bend** |
| `left_elbow` | — | — | **+90 = bend** |
| `right_hip` | Leg forward/back | Leg outward | — |
| `left_hip` | Leg forward/back | Leg outward | — |
| `right_knee` | **+90 = bend backward** | — | — |
| `left_knee` | **+90 = bend backward** | — | — |
| `right_ankle` | Foot point/flex | — | — |
| `left_ankle` | Foot point/flex | — | — |
| `right_wrist` | — | — | Hand rotation |
| `left_wrist` | — | — | Hand rotation |
| `neck` | Nod | Turn head | Tilt |
| `head` | Nod | Turn | Tilt |

### Example Poses

**T-Pose** (arms outstretched):
```json
{"right_shoulder": [0, 0, 90], "left_shoulder": [0, 0, -90]}
```

**Idle stance** (arms relaxed at sides):
```json
{"right_shoulder": [0, 0, -20], "left_shoulder": [0, 0, 20], "right_elbow": [0, 0, -10], "left_elbow": [0, 0, 10]}
```

**Wave** (right arm raised and bent):
```json
{"right_shoulder": [0, 0, 150], "right_elbow": [0, 0, -90]}
```

**Walking** (mid-stride):
```json
{"right_shoulder": [0, 0, -30], "left_shoulder": [0, 0, 30], "right_knee": [20, 0, 0], "left_hip": [15, 0, 0]}
```

**Combat stance** (guard position):
```json
{"spine": [5, 0, 0], "right_shoulder": [0, 0, 60], "left_shoulder": [0, 0, -40], "right_elbow": [0, 0, -90], "left_elbow": [0, 0, -80], "right_knee": [15, 0, 0], "left_knee": [10, 0, 0]}
```

### Viewing Angles (model_rotation_y)

| Value | Direction |
|-------|-----------|
| 0 | Front |
| 90 | Right side |
| 180 | Back |
| 270 | Left side |
| 45 | Front-right |
| 315 | Front-left |

## Animation Frame Strategy

For animation sequences (walk, idle, attack), generate one frame per pose at a time. Each frame is an independent workflow execution with a different `rotations_json`.

**Consistency between frames**: Use the same `seed` for all frames in an animation. The VNCCS reference latents are deterministic given the same character image, so the same seed produces the most consistent rendering across frames.

**Frame ordering**: For a walk cycle, define 4-8 keyframe poses and generate them sequentially. The evenly-spread approach works for looping animations.

## Python Usage (Recommended)

The pipeline module loads the tested workflow JSON and patches only what changes — no risk of misconfiguration:

```python
import requests, json, time, sys

COMFYUI = "http://100.86.69.57:18800/comfyui"

# Import the pipeline module from the comfyui-pose-director project
sys.path.insert(0, "/path/to/comfyui-pose-director")
from pipeline.ui_to_api import load_ui_workflow, patch_ui_workflow, convert_ui_to_api

def upload_image(image_path: str, name: str) -> str:
    resp = requests.post(f"{COMFYUI}/upload/image",
        files={"image": (name, open(image_path, "rb"), "image/png")},
        data={"subfolder": "PoseDirector", "overwrite": "true"})
    return resp.json()["name"]

def generate_sprite(character_image: str, pose: dict, direction: float = 0.0,
                    seed: int = 42, prefix: str = "sprite") -> str:
    """Generate a posed sprite with transparent background."""
    wf = load_ui_workflow("workflows/05_sprite_bodymesh_edit.json")
    patch_ui_workflow(wf, {
        "LoadImage": {"image": character_image},
        "BodyMeshRenderer": {
            "rotations_json": json.dumps(pose),
            "model_rotation_y": direction,
        },
        "KSampler": {"seed": seed},
        "SaveImage": {"filename_prefix": f"PoseDirector/{prefix}"},
    })
    api = convert_ui_to_api(wf)

    resp = requests.post(f"{COMFYUI}/prompt", json={"prompt": api}, timeout=60)
    prompt_id = resp.json()["prompt_id"]

    while True:
        time.sleep(3)
        hist = requests.get(f"{COMFYUI}/history/{prompt_id}").json()
        if prompt_id in hist and hist[prompt_id].get("status", {}).get("completed"):
            break

    for nid, out in hist[prompt_id]["outputs"].items():
        for img in out.get("images", []):
            return img["filename"]
    raise RuntimeError("No output images produced")

# Example: 4-direction idle sprite sheet
character = upload_image("luna.png", "luna.png")
idle_pose = {"right_shoulder": [0, 0, -20], "left_shoulder": [0, 0, 20]}

for direction, label in [(0, "front"), (90, "right"), (180, "back"), (270, "left")]:
    filename = generate_sprite(character, idle_pose, direction=direction, prefix=f"luna_idle_{label}")
    print(f"Generated: {filename} ({label})")
```

## Batch Generation with CLI Scripts

```bash
# Generate sprite sheet for one animation in all 4 directions
uv run python scripts/run_sprite_batch.py \
    --character luna.png \
    --animations idle \
    --directions front right back left \
    --seed 42

# Backgrounds are now removed in the workflow (VNCCS_RMBG2)
# The --remove-bg flag still works as a fallback via Python rembg
uv run python scripts/run_sprite_batch.py \
    --character luna.png \
    --animations idle wave combat \
    --directions front right back left

# Export to Godot project
uv run python scripts/export_godot.py manifest.json \
    --name luna \
    --project ~/game \
    --resize 512
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Different character per frame | Wrong image ordering in VNCCS | Use the exact JSON above — image1=mesh, image2=character |
| No transparency | Missing VNCCS_RMBG2 node | Verify node 13 is in the workflow |
| Hallucinated backgrounds | Wrong VNCCS instruction | Use the exact instruction text from the JSON |
| "No Qwen-Image-Edit GGUF" | Model file missing | Check `unet/` dir for GGUF file |
| "Invalid image file" | Upload failed | Re-upload the character PNG |
| Timeout after 300s | GPU loading models | Cold start takes ~5 min, retry |
| Blank output | BodyMeshRenderer returned empty | Check `rotations_json` is valid JSON |

## Model Files Required

| File | Location | Size |
|------|----------|------|
| `Qwen-Image-Edit-2511-Q5_0.gguf` | `unet/` | ~5.3GB |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `clip/` | ~7.8GB |
| `qwen_image_vae.safetensors` | `vae/` | ~230MB |
| `qwen/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | `loras/` | ~45MB |
| `qwen/VNCCS/VNCCS_PoseStudioQIE2511_V2.safetensors` | `loras/` | ~45MB |

All paths are relative to the ComfyUI `models/` directory.
