# ComfyUI Pose Director

AI-driven skeleton manipulation nodes for ComfyUI. Uses an LLM (via OpenAI-compatible API) to generate and refine OpenPose keyframes for character manipulation, then feeds them into Qwen Image Edit and LTX Video.

## ComfyUI Environment Setup

There is a standalone setup script at `~/Documents/programs/img/comfyui-setup/` that installs a complete ComfyUI environment from scratch. It can be run from anywhere.

### `setup_comfyui.sh` — Full Installer

```bash
bash setup_comfyui.sh --models /home/ubuntu/Documents/programs/shared-docker-infra/models/image-gen/comfyui [install_dir]
```

- `install_dir` defaults to `./ComfyUI` (relative to where you run the script)
- `--models` is optional — if provided, writes `extra_model_paths.yaml` so ComfyUI can find checkpoints, loras, vae, etc. from a shared model storage directory
- On this machine the shared models path is: `/home/ubuntu/Documents/programs/shared-docker-infra/models/image-gen/comfyui`
- Idempotent: re-running pulls latest git changes instead of re-cloning

**What it does (8 steps):**
1. Clone/pull ComfyUI from GitHub
2. Clone 11 custom nodes into `custom_nodes/` (Manager, Comfyroll, controlnet_aux, GGUF, Impact Pack, Inspire, VideoHelperSuite, QwenEditUtils, VNCCS, VNCCS_Utils)
3. Create Python 3.12 venv via `uv` (system Python is 3.14, too bleeding edge)
4. Install PyTorch 2.10.0+cu130 from PyTorch's cu130 index
5. Install flash-attn prebuilt wheel + `nvidia-cuda-runtime-cu12`
6. Install ComfyUI requirements (strips torch lines to avoid conflicts)
7. Install extras (triton, setuptools, wheel, ninja, sageattention, onnx) + best-effort custom node deps
8. Write `extra_model_paths.yaml` if `--models` was provided

### `run.sh` — Launcher

```bash
bash run.sh
```

cds into the ComfyUI directory next to the script and runs:
```
PYTHONWARNINGS="ignore::FutureWarning,ignore::SyntaxWarning" uv run main.py --port 8465 --use-flash-attention --dont-upcast-attention
```

### Key Technical Details

**GPU & CUDA:**
- RTX 4090 (24GB), CUDA toolkit 13.1, Ubuntu, 62GB RAM
- PyTorch cu130 works fine on CUDA 13.1 (backward compatible)
- cu130 is preferred over cu128 — avoids ComfyUI's "no CUDA 13" warning and enables comfy_kitchen CUDA backend optimizations

**Flash Attention:**
- NEVER compile flash-attn from source (OOMs due to systemd-oomd killing processes at 50% memory pressure)
- Use the prebuilt wheel: `flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` from the flash-attention v2.8.1 GitHub release
- Requires `nvidia-cuda-runtime-cu12` alongside cu130 PyTorch — provides `libcudart.so.12` that the wheel expects

**ComfyUI Flags:**
- `--use-flash-attention` — uses the flash_attn package for attention
- `--dont-upcast-attention` — prevents unnecessary fp32 cast
- `--port 8465` — user's preferred port
- `PYTHONWARNINGS="ignore::FutureWarning,ignore::SyntaxWarning"` — suppresses pynvml/groundingdino noise

**Package Manager:** `uv` throughout. `uv venv --python 3.12` to create venv, `uv pip install` for packages, `uv run` to execute. No venv activation needed.

### Broken Custom Nodes (upstream bugs)

- ComfyUI-MagCache — REMOVED, broken import, ComfyUI API changed
- ComfyUI-QwenVL — REMOVED, incompatible with transformers 5.x
- comfyui-rmbg/SDMatte — bad relative imports (loads 44 nodes fine, 2 sub-modules fail)
- comfyui-segment-anything-2 — pyproject.toml build conflicts, skipped in setup

---

## Pose Director Architecture

### End-to-End Pipeline

The full anime scene creation pipeline has 6 stages. Our extension handles step C.

```
A. Character Concept    Z-Image → initial character design
B. Character Sheet      Qwen Image Edit → full-body turnaround sheet
C. Pose Manipulation    OpenposePreprocessor → PoseManipulator/MotionDirector → skeleton keyframes  ← THIS EXTENSION
D. Video Generation     LTX Video 2.3 with keyframe guides → animated clips
E. Post-Processing      LTX EditAnything IC-LoRA → fix artifacts, add/remove elements
F. Lip Sync + Final     (future) → final composited scene
```

### Static Pose Workflow (single keyframe)
```
LoadImage → OpenposePreprocessor → PoseManipulator(Gemini) → skeleton IMAGE
                                                      ↓
CheckpointLoader + CLIPLoader(qwenedit) + VAELoader(qwenedit-vae)
                                                      ↓
VNCCS_QWEN_Encoder(image1=character, image2=skeleton, prompt, instruction)
                                                      ↓
KSampler → VAEDecode → SaveImage (pose-edited character)
```

### Motion Keyframe Workflow (multiple keyframes)
```
LoadImage → OpenposePreprocessor → MotionDirector(Gemini, N=4 keyframes) → IMAGE batch
                                                                ↓
ImpactImageBatchToImageList (split batch into individual images)
                                                                ↓
For EACH keyframe:
  VNCCS_QWEN_Encoder(image1=character, image2=keyframe_skeleton) → KSampler → VAEDecode → SaveImage
                                                                ↓
For LTX Video:
  Load each saved keyframe → LTXVAddGuideAdvanced(frame_idx=0,8,16,24)
  → LTXVSampler → VAEDecode → VideoCombine
```

### LTX EditAnything Post-Processing (Step E)
After generating video, use LTX EditAnything IC-LoRA for fixes:
- **LoRA model**: `ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors`
- **Source**: Alissonerdx/LTX-LoRAs on HuggingFace
- **Load via**: `LTXICLoRALoaderModelOnly` node
- **Prompt patterns**: "Add [object]", "Remove [object]", "Replace [X] with [Y]", "Convert [X] to [Y]"
- **Use for**: fixing artifacts, adding/removing scene elements, style adjustments

### Nodes
1. **PoseManipulator** — Takes an existing OpenPose skeleton + text prompt → LLM recalculates joint positions → outputs modified skeleton image.
2. **MotionDirector** — Takes a starting pose + text description of movement → LLM generates keyframes (not every frame, just anchors at evenly spaced frame numbers) → outputs batch of OpenPose skeleton images.
3. **PoseRenderer** — Converts OpenPose keypoints JSON into a skeleton image. Standalone renderer, also used internally by the other two nodes.
4. **PoseExtractor** — Bridge node: extracts body keypoints from OpenposePreprocessor's POSE_KEYPOINT output into flat JSON. Handles coordinate denormalization (0-1 → pixel).

### Key Design Decisions
- **Keyframes only, not full animation.** LTX 2.3 handles interpolation between anchor frames natively. The LLM just needs to produce 3-5 key poses.
- **OpenAI-compatible API.** Uses the `openai` Python package. Works with any provider (Ollama, LM Studio, LiteLLM proxy, OpenAI). Each node takes `base_url`, `api_key`, and `model` as inputs.
- **POSE_KEYPOINT compatibility.** Nodes accept both STRING (JSON) and POSE_KEYPOINT (from OpenposePreprocessor) as input. The `extract_keypoints_from_dwpose()` utility handles all output formats and auto-converts normalized coordinates (0-1 range from OpenposePreprocessor) to pixel coordinates.
- **Use OpenposePreprocessor, NOT DWPreprocessor.** DWPose's YOLOX person detector fails on anime images. The regular OpenposePreprocessor detects 18/18 keypoints on anime characters. OpenposePreprocessor returns normalized (0-1) coordinates; PoseExtractor auto-denormalizes them to pixel coordinates using canvas dimensions.
- **Portrait dimensions for full-body.** Z-Image character generation should use 768x1024 (portrait) to produce full-body characters. Square 1024x1024 produces bust crops.
- **Robust LLM parsing.** `parse_llm_json()` in `utils/json_parser.py` handles direct JSON, markdown-wrapped responses, trailing commas, and embedded JSON extraction.
- **Retry logic.** API client retries on timeout, connection errors, and rate limits (429/503) with exponential backoff.

### LLM Infrastructure
- **LiteLLM proxy**: `http://litellm.local/v1`, key `sk-litellm-e2e-test-master-key`
- **llama.cpp**: `http://llama.local/v1` (via Traefik)
- **Services managed by**: `~/Documents/programs/shared-docker-infra/` (Taskfile: `task up`)
- **Available models**: qwen-35-9b, qwen-35-27-vision, gemma-4-26b, etc. (see LiteLLM config.yaml)

## Development

### Setup
```bash
# Symlink into ComfyUI
ln -s /path/to/comfyui-pose-director /path/to/ComfyUI/custom_nodes/comfyui-pose-director
```

### Testing
```bash
# Start ComfyUI
bash /path/to/comfyui-setup/run.sh

# Trigger a workflow via API
curl -X POST http://127.0.0.1:8465/prompt -d @test_workflow.json
```

### Node Structure
Each node is a Python class with:
- `INPUT_TYPES` — class method defining inputs
- `RETURN_TYPES` — tuple of output types
- `FUNCTION` — string name of the execute method
- `CATEGORY` — where it appears in the ComfyUI menu (`pose_director`)

### Adding a New Node
1. Create `nodes/your_node.py` with a class following the ComfyUI pattern
2. Import it in `__init__.py` and add to `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`
3. Any shared utilities go in `utils/`

## OpenPose Format
Standard COCO-18, 18 keypoints per person as flat array `[x, y, confidence, ...]`:
```
[0]  Nose          [1]  Neck
[2]  R Shoulder    [5]  L Shoulder
[3]  R Elbow       [6]  L Elbow
[4]  R Wrist       [7]  L Wrist
[8]  R Hip         [11] L Hip
[9]  R Knee        [12] L Knee
[10] R Ankle       [13] L Ankle
[14] R Eye         [15] L Eye
[16] R Ear         [17] L Ear
```

## Project Structure
```
comfyui-pose-director/
├── __init__.py              # Registers nodes with ComfyUI
├── pyproject.toml           # Dependencies: openai>=1.0
├── nodes/
│   ├── pose_manipulator.py  # Static pose manipulation via LLM
│   ├── motion_director.py   # Keyframe generation via LLM
│   └── pose_renderer.py     # OpenPose JSON → color-coded skeleton image
├── utils/
│   ├── openpose.py          # Keypoint names, bone pairs, body part groups, colors
│   ├── api_client.py        # OpenAI-compatible API client with retry logic
│   └── json_parser.py       # Robust JSON extraction from LLM responses
├── ARCHITECTURE.md          # Full pipeline architecture and implementation plan
└── CLAUDE.md
```

### Impact Pack Batch Conversion
The MotionDirector outputs a batch of skeleton images, but VNCCS_QWEN_Encoder processes one image at a time. Use Impact Pack nodes for conversion:
- `ImpactImageBatchToImageList` — splits batch tensor into list of single images (`OUTPUT_IS_LIST = (True,)`)
- `ImageListToImageBatch` — merges list back into batch
- Located in: `comfyui-impact-pack/modules/impact/util_nodes.py`

### VNCCS Reference Latent Pattern (Critical)
Character consistency comes from the **reference latent** approach, NOT ControlNet:
- VNCCS_QWEN_Encoder VAE-encodes both the character image and pose skeleton as reference latents
- Injects at timestep zero via `reference_latents_method = "index_timestep_zero"`
- Uses quadratic weighting: `(weight^2 * latent)` per image
- Settings: `image1=character (weight 1.0)`, `image2=pose_skeleton (weight 1.0)`, `qwen_2511=True`
- Credit: AHEKOT/MiuProject, https://github.com/AHEKOT/ComfyUI_VNCCS, MIT license

## External Extensions (installed in ComfyUI)
- `comfyui_controlnet_aux` — OpenPose extraction (use `OpenposePreprocessor`, NOT `DWPreprocessor` — DWPose fails on anime), returns `(IMAGE, POSE_KEYPOINT)`
- `qweneditutils` — Qwen Image Edit nodes, takes CLIP + VAE + images + prompt
- `ComfyUI-LTXVideo@nightly` — LTX Video 2.3 (**ENABLED**)
- `comfyui-videohelpersuite` — Video combine/encode
- `vnccs` / `vnccs-utils` — Visual Novel Character Creator Suite with QWEN encoder
- `comfyui-impact-pack` — Batch/list conversion nodes (`ImpactImageBatchToImageList`)

## LTX EditAnything IC-LoRA (Post-Processing)
- **Model**: `ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors`
- **Author**: Alissonerdx
- **Source**: https://huggingface.co/Alissonerdx/LTX-LoRAs
- **Load via**: `LTXICLoRALoaderModelOnly` node
- **Prompt patterns**: "Add [X]", "Remove [X]", "Replace [X] with [Y]", "Convert [X] to [Y]"
- **Purpose**: Fix video artifacts, add/remove scene elements, style adjustments
- **Place in**: `shared-docker-infra/models/image-gen/comfyui/loras/`

## Reference Extensions (cloned for study)
- `/tmp/comfyui-openpose-studio/` — Visual pose editor, POSE_KEYPOINT patterns
- `/tmp/comfyui-llm-party/` — LLM API integration patterns
