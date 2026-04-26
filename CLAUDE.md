# ComfyUI Pose Director

AI-driven skeleton manipulation nodes for ComfyUI. Uses an LLM (via OpenAI-compatible API) to generate and refine OpenPose keyframes for character manipulation, then feeds them into Qwen Image Edit and LTX Video.

## Autonomous Iteration Workflow

This project has a closed-loop AI workflow for continuous pipeline improvement:

```
┌─────────────────────────────────────────────────────────┐
│  1. RUN     Execute pipeline (full or partial stage)    │
│     ↓        via ComfyUI MCP → execute_workflow         │
│  2. ANALYZE  Vision-assess output images                │
│     ↓        via analyze_image, compare_outputs         │
│  3. RESEARCH  If issues found, research solutions       │
│     ↓        via MCP web tools, discover, CivitAI       │
│  4. MODIFY   Apply changes to workflows/code            │
│     ↓        via apply_workflow_patch, Edit, Write      │
│  5. RECORD   Log what worked / didn't / why             │
│     └──────→ memory files, record_outcome, add_note     │
│                  then loop back to step 1                │
└─────────────────────────────────────────────────────────┘
```

### Step-by-Step Pattern

**1. Run Pipeline Stage**
- Use MCP tools: `load_workflow` → `validate_before_execute` → `execute_workflow`
- Or run E2E test: `RUN_E2E=1 uv run pytest tests/e2e/ -v -s`
- Partial stages: load individual workflow from `workflows/` directory
- Always check ComfyUI is running first: `comfyui_agent_ping`

**2. Analyze Output Images**
- `analyze_image` — quality score (0-1), artifacts, prompt adherence, suggestions
- `compare_outputs` — A/B comparison after a workflow change
- `hash_compare_images` — fast pre-check before detailed vision analysis
- Output images land in: `/home/ubuntu/Documents/programs/img/comfyui/output/e2e/`
- Get absolute paths: `get_output_path` or `get_history`

**3. Research Solutions**
- Spawn a research subagent for web searches when vision analysis reveals issues
- `discover` — search for models/nodes on CivitAI, HuggingFace, registry
- `get_node_info` — check exact parameter schemas before modifying workflows
- `get_editable_fields` — see what can be changed without rewiring
- Use the `research` MCP tools (search, scrape, crawl) for online solutions

**4. Modify and Test**
- `apply_workflow_patch` — surgical parameter changes to loaded workflows
- `set_input` / `connect_nodes` / `add_node` — programmatic workflow editing
- `preview_workflow_patch` — preview changes before applying
- `undo_workflow_patch` — rollback if change doesn't help
- For code changes: Edit tool → rsync to ComfyUI → restart ComfyUI

**5. Record and Iterate**
- `record_outcome` — log parameters, quality scores, what worked
- `add_note` — session notes (preference, observation, decision, tip)
- `save_session` — persist state across conversations
- `get_learned_patterns` — query past outcomes for recommendations
- Update `memory/MEMORY.md` with stable findings

### Known Quality Issues (from Vision Analysis)

| Issue | Severity | Suggested Fix |
|-------|----------|---------------|
| Hand/foot missing from skeletons | High | Add hand detector or switch to DWPose for non-anime |
| Facial distortion in VNCCS edits | Medium | Increase steps 30→40, or add inpainting pass |
| Background warping near character | Medium | Increase padding or use larger canvas |
| Q4 softness in LTX video output | Low | Higher quantization or post-sharpen |
| Hallucinated text on buildings | Low | Add negative prompt targeting text/signs |

---

## MCP Tools (Comfy-Cozy)

The ComfyUI Agent MCP server is configured in `.mcp.json` at project root. It provides 114 tools for autonomous ComfyUI workflow management.

### Quick Reference

| Task | Tool | Notes |
|------|------|-------|
| Check connection | `comfyui_agent_ping` | Always call first |
| Load workflow | `load_workflow` | Accepts absolute path |
| See what you can change | `get_editable_fields` | Groups by node type |
| Change a value | `set_input` or `apply_workflow_patch` | Patch supports bulk changes |
| Add a node | `add_node` + `connect_nodes` | Two-step process |
| Validate before run | `validate_before_execute` | Catches errors early |
| Run workflow | `execute_workflow` | Returns output filenames |
| Run with progress | `execute_with_progress` | Real-time updates, better for long renders |
| Get output path | `get_output_path` | Resolve filename → absolute path |
| Analyze image quality | `analyze_image` | Returns quality score + artifacts + suggestions |
| Compare two outputs | `compare_outputs` | A/B diff with change description |
| Find models/nodes | `discover` | Searches local + CivitAI + HuggingFace |
| Get node schema | `get_node_info` | Exact inputs/outputs/defaults |
| Check GPU/VRAM | `get_system_stats` | GPU name, VRAM usage, loaded models |
| Record what happened | `record_outcome` | Logs params + quality for learning |
| Save session state | `save_session` | Persists across conversations |

### ComfyUI Connection
- ComfyUI runs on `127.0.0.1:8465`
- Start: `bash ~/Documents/programs/img/comfyui-setup/run.sh`
- MCP config in `.mcp.json` (project-level) and `~/.claude/settings.json` (global fallback)

---

## End-to-End Pipeline

```
A. Character Concept    Z-Image Turbo (768x1024) → character.png          [00_z_image_character.json]
B. Character Sheet      VNCCS QWEN Edit → turnaround sheet (not yet implemented)
C. Pose Manipulation    OpenposePreprocessor → MotionDirector → Anny body mesh keyframes  [01_motion_keyframes_rotation.json]
D. Pose → Character     VNCCS_QWEN_Encoder(image1=char, image2=mesh) → edited keyframes   [02_vnccs_edit_keyframe.json]
E. Video Assembly       LTX Video 2.3 with keyframe guides → MP4                         [04_ltx_video_assembly.json]
F. Post-Processing      LTX EditAnything IC-LoRA → fix artifacts                          [06_ltx_edit_anything_postprocess.json]
G. Lip Sync             InfiniteTalk (not yet installed)                                 [07_lip_sync.json]
```

### Workflow Files (Source of Truth)

All workflows are in **ComfyUI UI format** — load them directly in ComfyUI's browser to inspect, edit, and debug. The AIO script and tests convert them to API format on the fly via `pipeline/ui_to_api.py`.

| File | Stage | Description |
|------|-------|-------------|
| `00_z_image_character.json` | A | Z-Image Turbo character generation |
| `01_motion_keyframes_rotation.json` | C | OpenPose → MotionDirector (rotation mode → Anny 3D mesh) |
| `02_vnccs_edit_keyframe.json` | D | Single-keyframe VNCCS QWEN editing (character + keyframe → edited) |
| `02_vnccs_edit_mesh.json` | C+D | Static pose pipeline: OpenPose → PoseManipulator → VNCCS edit |
| `03_motion_keyframes_qwen_edit_mesh.json` | C+D | Motion keyframes with VNCCS editing (batch split version) |
| `04_ltx_video_assembly.json` | E | LTX Video 2.3 with keyframe guides |
| `06_ltx_edit_anything_postprocess.json` | F | Post-processing with IC-LoRA |
| `07_lip_sync.json` | G | Lip sync (placeholder) |

**No more parallel API-format JSONs.** The `pipeline/ui_to_api.py` module handles conversion:
- `load_ui_workflow(path)` — load UI JSON
- `patch_ui_workflow(workflow, patches)` — batch-patch widget values by node type
- `patch_widget_value(workflow, node_type, input_name, value, node_index=0)` — single value patch
- `patch_connection(workflow, ...)` — add/rewire connections
- `convert_ui_to_api(workflow)` → API-format dict ready for `/prompt`

The AIO runner (`scripts/run_full_pipeline.py`) loads these files, patches runtime values (image paths, seeds, prompts), converts, submits, and saves per-stage artifacts to `output/run_YYYYMMDD_HHMMSS/{stage_a,stage_c,stage_d,stage_e}/` alongside the patched workflow JSONs.

---

## Architecture

### Nodes (8 registered)
1. **PoseManipulator** — OpenPose skeleton + text prompt → LLM recalculates joints → modified skeleton IMAGE
2. **MotionDirector** — Starting pose + movement description → LLM generates N keyframes → IMAGE batch
3. **PoseRenderer** — OpenPose keypoints JSON → color-coded skeleton image
4. **PoseExtractor** — POSE_KEYPOINT from OpenposePreprocessor → flat JSON (handles denormalization)
5. **KeyframeReviewer** — IMAGE batch → vision LLM review (consistency score, passed/failed)
6. **BodyMeshRenderer** — Named joint rotations → Anny forward pass → 3D mesh render
7. **AnimationExporter** — Keyframe rotations → SLERP interpolation → BVH file + MP4/GIF preview
8. **PromptEnhancer** — Simple prompt → detailed Z-Image prompt

### Key Design Decisions
- **SLERP interpolation for smooth animation.** scipy quaternion SLERP between keyframes, exported as BVH (game engines) and MP4/GIF (preview).
- **BVH uses independent 15-bone hierarchy.** Not tied to Anny's 163 bones — game engines re-target anyway.
- **OpenAI-compatible API.** Works with any provider (llama.cpp, LiteLLM, OpenAI). Each node takes `base_url`, `api_key`, `model`.
- **Use OpenposePreprocessor, NOT DWPreprocessor.** DWPose's YOLOX fails on anime.
- **Portrait for full-body.** Z-Image at 768x1024 for head-to-toe characters.
- **Robust LLM parsing.** `parse_llm_json()` handles direct JSON, markdown-wrapped, trailing commas, embedded extraction.

### LLM Setup
- **Primary**: Qwen3.6-35B-A3B via llama.cpp (`http://127.0.0.1:8001/v1`)
  - Docker container `orchestrator-llama`, model `qwen3.6-35b-a3b`
  - MoE 35B total / 3B active, ~86 tok/s, thinking mode enabled
  - API key: `sk-no-key-required`
- **Fallback**: Gemini 2.5 Flash (free tier, 20 req/day)
- **Vision**: `call_llm_vision()` in api_client.py — base64 PNG via OpenAI chat format

### VNCCS Pattern (Critical)
Character consistency uses **reference latents**, NOT ControlNet:
- VNCCS_QWEN_Encoder VAE-encodes character + skeleton as reference latents
- Injects at timestep zero (`reference_latents_method = "index_timestep_zero"`)
- Quadratic weighting: `(weight^2 * latent)` per image
- Key params: `image1_name`, `image2_name`, `weight1`, `weight2`, `target_size`, `vl_size`, `instruction`, `qwen_2511`
- **Always verify params against live schema**: `get_node_info("VNCCS_QWEN_Encoder")`

### LTX Video 2.3 Setup
- **UNET**: `ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf` (16.4GB) via UnetLoaderGGUF
- **Text Encoder**: Gemma 3 12B via LTXVGemmaCLIPModelLoader
  - Also needs: `ltx-2.3_text_projection_bf16.safetensors` + `ltx-2.3-connectors-textproj.safetensors`
- **VAE (full)**: `LTX23_video_vae_bf16.safetensors` (1.45GB) — required for LTXVAddGuideAdvanced
- **VAE (tiny)**: `taeltx2_3.safetensors` (TAEHV) — fast but INCOMPATIBLE with guide injection
- **Latent**: `EmptyLTXVLatentVideo` (5D tensor, NOT EmptyLatentImage)
- **Keyframe injection**: `LTXVAddGuideAdvanced` chains at frame_idx (must be divisible by 8)

---

## Development

### Paths
- **Source**: `/home/ubuntu/Documents/programs/img/comfyui-pose-director`
- **Installed**: `/home/ubuntu/Documents/programs/img/comfyui/custom_nodes/comfyui-pose-director` (full copy, rsync from source)
- **ComfyUI**: `/home/ubuntu/Documents/programs/img/comfyui` (port 8465)
- **Shared models**: `/home/ubuntu/Documents/programs/shared-docker-infra/models/image-gen/comfyui`
- **Output**: `/home/ubuntu/Documents/programs/img/comfyui/output/e2e/`

### After Code Changes
```bash
rsync -av --delete --exclude='__pycache__' /home/ubuntu/Documents/programs/img/comfyui-pose-director/ /home/ubuntu/Documents/programs/img/comfyui/custom_nodes/comfyui-pose-director/
# Then restart ComfyUI
```

### Testing
```bash
# Unit tests (no ComfyUI needed)
uv run pytest tests/unit/ -v

# Integration tests (ComfyUI + LLM needed)
uv run pytest tests/integration/ -v

# E2E full pipeline (~17 min, requires ComfyUI + llama.cpp + all models)
RUN_E2E=1 uv run pytest tests/e2e/ -v -s
```

### ComfyUI App Builder
Generates `.app.json` files from extension workflows for ComfyUI's native App Builder (Apps sidebar tab).
```bash
# Generate all apps (needs ComfyUI running for best results)
uv run python scripts/build_comfyui_apps.py

# Offline mode (uses heuristics only)
uv run python scripts/build_comfyui_apps.py --offline

# Single extension
uv run python scripts/build_comfyui_apps.py --source vnccs

# Preview without writing files
uv run python scripts/build_comfyui_apps.py --dry-run
```
Covers: VNCCS, VNCCS-Utils, HY-Motion, Pose Director (26 apps across 7 categories).
Output: `{comfyui_dir}/Workflows/apps/{category}/` + `docs/workflow_catalog.md`

### Environment
- RTX 4090 (24GB VRAM), CUDA 13.1, Ubuntu, 62GB RAM
- Python 3.12 via `uv` (system Python is 3.14, too new)
- PyTorch 2.10.0+cu130
- Flash Attention via prebuilt wheel (NEVER compile from source — systemd-oomd kills at 50% memory)

### Setup Script
`~/Documents/programs/img/comfyui-setup/setup_comfyui.sh` installs a complete ComfyUI environment:
```bash
bash setup_comfyui.sh --models /home/ubuntu/Documents/programs/shared-docker-infra/models/image-gen/comfyui [install_dir]
```
Launch with: `bash ~/Documents/programs/img/comfyui-setup/run.sh`

---

## OpenPose Format
COCO-18, 18 keypoints as flat array `[x, y, confidence, ...]`:
```
[0] Nose  [1] Neck  [2] R Shoulder  [5] L Shoulder
[3] R Elbow  [6] L Elbow  [4] R Wrist  [7] L Wrist
[8] R Hip  [11] L Hip  [9] R Knee  [12] L Knee
[10] R Ankle  [13] L Ankle  [14] R Eye  [15] L Eye
[16] R Ear  [17] L Ear
```

## External Extensions
- `comfyui_controlnet_aux` — OpenPose extraction (`OpenposePreprocessor` only)
- `ComfyUI-GGUF` — GGUF model loading (UNET, CLIP)
- `ComfyUI-LTXVideo@nightly` — LTX Video 2.3
- `vnccs` / `vnccs-utils` — VNCCS QWEN encoder for character consistency
- `qweneditutils` — Qwen Image Edit utilities
- `comfyui-impact-pack` — Batch/list conversion (`ImpactImageBatchToImageList`)
- `comfyui-videohelpersuite` — Video combine/encode

## Broken Custom Nodes (upstream bugs)
- ComfyUI-MagCache — REMOVED, broken import
- ComfyUI-QwenVL — REMOVED, incompatible with transformers 5.x
- comfyui-rmbg/SDMatte — bad relative imports (44 nodes load, 2 sub-modules fail)
- comfyui-segment-anything-2 — pyproject.toml build conflicts
