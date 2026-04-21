# ComfyUI Pose Director

AI-driven skeleton manipulation nodes for ComfyUI. Uses an LLM (via OpenAI-compatible API) to generate and refine OpenPose keyframes for character manipulation, then feeds them into Qwen Image Edit and LTX Video.

## Architecture

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
1. **LLMConnector** — OpenAI-compatible API client. Plugs into any provider (Ollama, LM Studio, Gemini via proxy, OpenAI).
2. **PoseManipulator** — Takes an existing OpenPose skeleton + text prompt → LLM recalculates joint positions → outputs modified skeleton image.
3. **MotionDirector** — Takes a starting pose + text description of movement → LLM generates keyframes (not every frame, just anchors) → outputs batch of OpenPose images.
4. **PoseRenderer** — Converts OpenPose JSON/coordinates into properly formatted skeleton images for ControlNet consumption.

### Key Design Decisions
- **Keyframes only, not full animation.** LTX 2.3 handles interpolation between anchor frames natively. The LLM just needs to produce 3-5 key poses.
- **OpenAI-compatible API.** Works with any provider. The node takes `base_url` and `api_key` as inputs.
- **Interactive mode.** The LLM Director node can pause for user feedback before finalizing poses. ComfyUI's `NODE_CLASS_MAPPINGS` supports nodes that return partial results.
- **Multi-person support.** OpenPose JSON naturally supports multiple people via separate keypoint arrays.

## Development

### Setup
```bash
# Symlink into ComfyUI (adjust paths to your setup)
ln -s /path/to/comfyui-pose-director /path/to/ComfyUI/custom_nodes/comfyui-pose-director
```

### Testing
Use ComfyUI's API mode to automate the test loop:
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
- `CATEGORY` — where it appears in the ComfyUI menu

### Reference Extensions (for learning patterns)
- `comfyui_controlnet_aux` — OpenPose/DWPose extraction, skeleton rendering
- `ComfyUI-LTXVideo` — multi-keyframe sequencing, guide insertion
- `ComfyUI-Manager` — API integration patterns

## OpenPose Format
Standard COCO + foot format, 18 keypoints per person:
```
[0]  Nose
[1]  Neck
[2]  Right Shoulder  [5]  Left Shoulder
[3]  Right Elbow     [6]  Left Elbow
[4]  Right Wrist     [7]  Left Wrist
[8]  Right Hip       [11] Left Hip
[9]  Right Knee      [12] Left Knee
[10] Right Ankle     [13] Left Ankle
[14] Right Eye       [15] Left Eye
[16] Right Ear       [17] Left Ear
```
Each keypoint is `[x, y, confidence]`. Skeleton bones connect specific keypoint pairs.

## Project Structure
```
comfyui-pose-director/
├── __init__.py              # Registers nodes with ComfyUI
├── nodes/
│   ├── llm_connector.py     # OpenAI-compatible API client node
│   ├── pose_manipulator.py  # Static pose manipulation via LLM
│   ├── motion_director.py   # Keyframe generation via LLM
│   └── pose_renderer.py     # OpenPose JSON → image rendering
├── utils/
│   ├── openpose.py          # OpenPose data structures and constants
│   └── api_client.py        # Shared API client logic
├── requirements.txt
├── CLAUDE.md
└── README.md
```
