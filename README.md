# ComfyUI Pose Director

AI-driven skeleton manipulation nodes for ComfyUI. Uses an LLM (via OpenAI-compatible API) to generate and refine OpenPose keyframes for character manipulation, then feeds them into Qwen Image Edit and LTX Video.

## Nodes

| Node | Description |
|------|-------------|
| **Pose Manipulator (LLM)** | Takes an OpenPose skeleton + text instruction, uses an LLM to recalculate joint positions, outputs a modified skeleton image |
| **Motion Director (LLM)** | Takes a starting pose + motion description, generates N keyframe poses at evenly-spaced frame numbers |
| **Pose Renderer** | Converts OpenPose keypoints JSON into a color-coded skeleton image (matches DWPose output style) |
| **Pose Extractor** | Bridge node: extracts body keypoints from OpenposePreprocessor's POSE_KEYPOINT output into flat JSON |

## Pipeline

```
Base Image → DWPose (extract skeleton)
                  ↓
         Pose Manipulator / Motion Director (LLM generates poses)
                  ↓
         Qwen Image Edit (reference-latent pose transfer)
                  ↓
         LTX Video 2.3 (animate between keyframes)
```

## Key Credits

### Qwen Image Edit Reference Latent Pattern

The near-perfect character consistency in the image editing step is achieved through a **reference latent** approach, not ControlNet. This pattern was pioneered by the **VNCCS (Visual Novel Character Creation Suite)** project:

- **Author**: [AHEKOT / MiuProject](https://github.com/AHEKOT)
- **Repository**: [https://github.com/AHEKOT/ComfyUI_VNCCS](https://github.com/AHEKOT/ComfyUI_VNCCS)
- **License**: MIT

The technique works by VAE-encoding both the original character image and a pose skeleton as **reference latents**, then injecting them at timestep zero (`reference_latents_method = "index_timestep_zero"`) with quadratic weighting (`weight^2 * latent`). This gives the model direct access to the original character's appearance while guiding it toward the target pose.

The `VNCCS_QWEN_Encoder` node handles this wiring:
```
image1 = original character (weight: 1.0)  → preserves identity
image2 = pose skeleton guide (weight: 1.0) → provides target pose
qwen_2511 = True                            → enables timestep-zero injection
```

Without this insight, achieving character-consistent pose editing with Qwen Image Edit would not be practical.

## LLM Configuration

Works with any OpenAI-compatible API:

| Provider | base_url | Notes |
|----------|----------|-------|
| **Gemini Flash** (default) | `https://generativelanguage.googleapis.com/v1beta/openai/` | Free tier, zero VRAM |
| Ollama | `http://localhost:11434/v1` | Local, uses GPU VRAM |
| LiteLLM proxy | `http://litellm.local/v1` | Routes to any backend |
| OpenAI | `https://api.openai.com/v1` | Paid API |

Set `base_url`, `api_key`, and `model` on each node. No code changes needed.

## Installation

```bash
# Symlink into ComfyUI
ln -s /path/to/comfyui-pose-director /path/to/ComfyUI/custom_nodes/comfyui-pose-director

# Install dependency (openai package)
cd /path/to/ComfyUI
uv pip install openai
```

## External Dependencies

These ComfyUI extensions must also be installed:

- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) — OpenPose extraction (use OpenposePreprocessor, not DWPreprocessor — DWPose fails on anime)
- [qweneditutils](https://github.com/lrzjason/qweneditutils) or [VNCCS](https://github.com/AHEKOT/ComfyUI_VNCCS) — Qwen Image Edit nodes
- [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) — LTX Video 2.3 (for motion pipeline)

## Workflows — Full Scene Maker Pipeline

Follow these workflows in order to create a complete animated scene from scratch.

| # | File | Description |
|---|------|-------------|
| 0 | `00_z_image_character.json` | Generate character image from text (Z-Image Turbo) |
| 1 | — | *(optional)* Use VNCCS `VN_Step1_QWEN_CharSheetGenerator` to make a character sheet |
| 2 | *(our nodes handle this)* | DWPose extracts skeleton, LLM generates keyframe poses |
| 3 | `03_static_pose_qwen_edit.json` | Apply a single pose to character (VNCCS_QWEN_Encoder) |
| 4 | `04_motion_keyframes_qwen_edit.json` | Apply multiple keyframe poses (batch → per-keyframe Qwen edits) |
| 5 | `05_ltx_video_assembly.json` | Stitch keyframe images into video (LTX Video 2.3) |
| 6 | `06_ltx_edit_anything_postprocess.json` | Fix artifacts, add/remove elements (EditAnything IC-LoRA) |
| 7 | `07_lip_sync.json` | Add voice/lip sync (InfiniteTalk via WanVideoWrapper) |

Basic node tests are in `workflows/tests/`.

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
