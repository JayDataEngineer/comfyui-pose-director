# Pose Director — Architecture & Implementation Plan

## Pipeline Overview

```
Base Image
    ↓
DWPose Preprocessor (controlnet_aux)
    ↓ outputs (IMAGE, POSE_KEYPOINT)
PoseManipulator / MotionDirector (this extension)
    ↓ LLM recalculates joint positions
    ↓ outputs (IMAGE batch of skeletons, STRING keypoints JSON, POSE_KEYPOINT)
Qwen Image Edit (qweneditutils)
    ↓ takes base image + pose image + text prompt → edited image
    ↓ runs once per keyframe pose
LTX Video 2.3 (ComfyUI-LTXVideo)
    ↓ receives multiple edited images as guide frames
    ↓ interpolates between keyframes
    ↓ outputs final video
```

## External Dependencies (already installed)

| Extension | Directory | Purpose |
|-----------|-----------|---------|
| controlnet_aux | `comfyui_controlnet_aux` | DWPose/OpenPose extraction from images |
| qweneditutils | `qweneditutils` | Qwen Image Edit nodes |
| LTX Video | `ComfyUI-LTXVideo@nightly` | Video generation with keyframe guides (**ENABLED**) |
| VideoHelperSuite | `comfyui-videohelpersuite` | Video combine/encode |

## Data Formats

### ComfyUI Tensor Format
- Shape: `(batch, height, width, channels)` — float32, 0.0–1.0
- Convert with `tensor.movedim(-1, 1)` for PyTorch `(B, C, H, W)`

### OpenPose JSON from DWPose/OpenPose
```json
{
  "people": [
    {
      "pose_keypoints_2d": [x, y, conf, x, y, conf, ...],   // 18 body keypoints × 3
      "face_keypoints_2d": [...],                              // 70 face keypoints × 3
      "hand_left_keypoints_2d": [...],                         // 21 hand keypoints × 3
      "hand_right_keypoints_2d": [...]                         // 21 hand keypoints × 3
    }
  ],
  "canvas_height": 512,
  "canvas_width": 512
}
```

### Our Internal Format (COCO-18, body only)
Flat array of 54 floats: `[x, y, conf, x, y, conf, ...]` for 18 keypoints.
This is what `pose_keypoints_2d` contains — we extract just the body part.

### LLM Interface
- **OpenAI-compatible API** via the `openai` Python package
- Works with: LiteLLM proxy (`http://litellm.local/v1`), llama.cpp, Ollama, any OpenAI-compatible endpoint
- Auth: `api_key` param (empty string for local models)

## Node Inventory

### Completed Nodes

#### 1. PoseRenderer (`nodes/pose_renderer.py`)
- **Input**: keypoints JSON string, width, height, line_width, point_radius
- **Output**: IMAGE tensor
- **Status**: Done. Reference-matched gradient colors (17 bone colors + 18 keypoint colors). Multi-person support.

#### 2. PoseManipulator (`nodes/pose_manipulator.py`)
- **Input**: keypoints STRING + text instruction + LLM config + optional POSE_KEYPOINT
- **Output**: IMAGE + STRING (keypoints JSON) + POSE_KEYPOINT
- **Status**: Done. Tested with Gemini 2.5 Flash. COCO-17 compatible input. Retry logic. Robust JSON parsing.

#### 3. MotionDirector (`nodes/motion_director.py`)
- **Input**: keypoints STRING + motion description + keyframe count + total frames + LLM config + optional POSE_KEYPOINT
- **Output**: IMAGE batch + STRING (keyframes JSON) + POSE_KEYPOINT (per-frame)
- **Status**: Done. Generates N keyframes at evenly-spaced frame numbers. COCO-17 compatible. Batch output for direct consumption.

#### 4. PoseExtractor (`nodes/pose_extractor.py`)
- **Input**: POSE_KEYPOINT (from DWPose) + optional person_index
- **Output**: STRING (keypoints JSON) + INT (canvas_height) + INT (canvas_width)
- **Status**: Done. Bridge node for DWPose output. COCO-17 auto-conversion.

### Reference Extensions (cloned for study)
- `/tmp/comfyui-openpose-studio/` — Visual pose editor, POSE_KEYPOINT handling patterns
- `/tmp/comfyui-llm-party/` — LLM API integration patterns, config hierarchy

## LLM Integration Details

### API Client (`utils/api_client.py`)
- Uses `openai` package with retry logic (timeout, 429/503, exponential backoff)
- Default: `base_url="https://generativelanguage.googleapis.com/v1beta/openai/"`, `model="models/gemini-2.5-flash"`
- Zero VRAM — LLM runs on Google's servers (free tier)
- Also works with: LiteLLM, llama.cpp, Ollama, any OpenAI-compatible endpoint

### LLM Prompt Strategy
The LLM receives:
1. Current keypoint coordinates (flat array with labels)
2. Image dimensions (for coordinate scale context)
3. Text instruction describing the desired pose/motion change

The LLM must return ONLY JSON — no explanations. We strip markdown wrappers as fallback.

### Response Parsing
```python
# Strategy: try direct parse, then strip markdown, then regex extract JSON
def parse_llm_json(response: str) -> dict:
    response = response.strip()
    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    # Strip markdown code block
    if "```" in response:
        block = response.split("```")[1]
        if block.startswith("json"):
            block = block[4:]
        return json.loads(block.strip())
    # Try to find JSON object/array in text
    import re
    match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f"Could not extract JSON from LLM response")
```

## Qwen Image Edit Integration

### Key Node: `TextEncodeQwenImageEditPlusAdvance_lrzjason`
- **Inputs**: CLIP, prompt, VAE, up to 6 images, instruction
- **Outputs**: CONDITIONING, LATENT, processed images, pad_info
- **How it works**:
  1. Images are resized and VAE-encoded into reference latents
  2. Images are also resized for vision-language (VL) processing
  3. CLIP tokenizes prompt with image placeholders
  4. Returns conditioning with reference latents attached

### How to feed our skeletons:
- `image1` = original base image (the character)
- `image2` = our rendered OpenPose skeleton (the target pose)
- `prompt` = description of the desired edit
- `instruction` = system prompt telling Qwen how to interpret the edit

### This means:
Our PoseManipulator outputs a skeleton IMAGE that can plug directly into
Qwen's image2 input. The Qwen node handles VAE encoding internally.

## LTX Video Integration

### Key Node: `LTXVAddGuideAdvanced`
- **Inputs**: conditioning, latent, image, frame_idx, strength
- **Outputs**: CONDITIONING, CONDITIONING, LATENT
- Adds a reference image at a specific frame index

### How to feed keyframes:
```
For each keyframe pose edited by Qwen:
    LTXVAddGuideAdvanced(image=qwen_output, frame_idx=N, ...)
Chain them: output → next node's latent input
```

### Frame numbering:
- Frame 0 = first frame
- For 9+ frame videos, frame_idx must be divisible by 8
- Typical: frame 0, 8, 16, 24, etc.

### IC-LoRA for pose control:
- `ltx-2-19b-ic-lora-pose-control.safetensors` (if available)
- `ltx-2-19b-ic-lora-union-ref0.5.safetensors` (combined control)
- Loaded via `LTXICLoRALoaderModelOnly`

## Implementation Phases

### Phase 1: Core Hardening — DONE
1. ✅ pyproject.toml migration (no more requirements.txt)
2. ✅ openai package integration with retry logic
3. ✅ Reference-matched color-coded renderer (17 bone + 18 keypoint gradient)
4. ✅ Package structure (__init__.py in nodes/ and utils/)
5. ✅ Robust JSON parsing with 4 fallback strategies
6. ✅ Accept POSE_KEYPOINT type directly + STRING
7. ✅ Extract body keypoints from DWPose nested format
8. ✅ COCO-17 auto-detection and conversion to COCO-18
9. ✅ POSE_KEYPOINT output on all manipulation nodes
10. ✅ PoseExtractor bridge node for explicit type conversion
11. ✅ Tested with Gemini 2.5 Flash (zero VRAM)
12. ✅ Example workflow JSONs in workflows/
13. ✅ LTX Video extension enabled
14. ✅ Reference repos cloned (OpenPose Studio, LLM Party, VNCCS studied)

### Phase 2: Integration Testing — NEXT
1. 🔲 Start ComfyUI
2. 🔲 Test DWPose → PoseManipulator → Preview via API
3. 🔲 Test DWPose → MotionDirector → Preview via API
4. 🔲 Verify POSE_KEYPOINT passthrough to ControlNet nodes

### Phase 3: Qwen Pipeline
1. 🔲 Wire PoseManipulator skeleton → Qwen Image Edit
2. 🔲 Determine optimal Qwen instruction prompt for pose editing
3. 🔲 Test single-pose manipulation end-to-end
4. 🔲 Test multi-person scenarios

### Phase 4: Motion Pipeline
1. 🔲 MotionDirector → multiple Qwen edits → LTX guides
2. 🔲 Frame number calculation (divisible by 8)
3. 🔲 Test keyframe interpolation quality
4. 🔲 Tune LLM prompts for motion generation

### Phase 5: Polish
1. 🔲 Workflow documentation
2. 🔲 Multi-person support refinement
3. 🔲 Community-ready release

## Testing Approach

### ComfyUI API Testing
```bash
# Start ComfyUI
bash ~/Documents/programs/img/comfyui-setup/run.sh

# Trigger workflow
curl -X POST http://127.0.0.1:8465/prompt \
  -H "Content-Type: application/json" \
  -d @test_workflow.json

# Check status
curl http://127.0.0.1:8465/history/<prompt_id>
```

### Workflow JSON Format

**VNCCS workflows**: Pre-exported API format from ComfyUI browser ("Save (API Format)"). Stored in `workflows/api/`. Each node is keyed by ID with `class_type` and `inputs` — patch directly, no converter needed.

**Our custom workflows**: UI format, converted at runtime via `pipeline/ui_to_api.py`.
