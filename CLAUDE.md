# ComfyUI Pose Director

AI-driven skeleton manipulation nodes for ComfyUI. Uses an LLM (via OpenAI-compatible API) to generate and refine OpenPose keyframes for character manipulation, then feeds them into Qwen Image Edit and LTX Video.

## ComfyUI Environment Setup

There is a standalone setup script at `~/Documents/programs/img/comfyui-setup/` that installs a complete ComfyUI environment from scratch. It can be run from anywhere.

### `setup_comfyui.sh` — Full Installer

```bash
bash setup_comfyui.sh [--models /path/to/shared/models] [install_dir]
```

- `install_dir` defaults to `./ComfyUI` (relative to where you run the script)
- `--models` is optional — if provided, writes `extra_model_paths.yaml` so ComfyUI can find checkpoints, loras, vae, etc. from a shared model storage directory
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

### Pipeline
```
Base Image → DWPose (extract skeleton)
                  ↓
         LLM Pose Director (this extension)
                  ↓
         OpenPose keyframes (static or motion)
                  ↓
         Qwen Image Edit (apply poses to character)
                  ↓
         LTX Video Sequencer (animate between keyframes)
```

### Nodes
1. **PoseManipulator** — Takes an existing OpenPose skeleton + text prompt → LLM recalculates joint positions → outputs modified skeleton image.
2. **MotionDirector** — Takes a starting pose + text description of movement → LLM generates keyframes (not every frame, just anchors at evenly spaced frame numbers) → outputs batch of OpenPose skeleton images.
3. **PoseRenderer** — Converts OpenPose keypoints JSON into a skeleton image. Standalone renderer, also used internally by the other two nodes.

### Key Design Decisions
- **Keyframes only, not full animation.** LTX 2.3 handles interpolation between anchor frames natively. The LLM just needs to produce 3-5 key poses.
- **OpenAI-compatible API.** Works with any provider (Ollama, LM Studio, Gemini via proxy, OpenAI). Each node takes `base_url`, `api_key`, and `model` as inputs.
- **No extra dependencies beyond `openai`.** The API client uses `urllib` directly. Flash attention and torch come from the ComfyUI environment.

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
├── nodes/
│   ├── pose_manipulator.py  # Static pose manipulation via LLM
│   ├── motion_director.py   # Keyframe generation via LLM
│   └── pose_renderer.py     # OpenPose JSON → image rendering
├── utils/
│   ├── openpose.py          # Keypoint names, bone pairs, body part groups
│   └── api_client.py        # OpenAI-compatible API client (urllib)
├── requirements.txt         # openai
└── CLAUDE.md
```
