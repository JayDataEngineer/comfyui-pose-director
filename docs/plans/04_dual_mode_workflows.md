# Plan 4: Dual-Mode Workflows

## Status: Designed

## Concept

Every pipeline stage supports two paths: **manual** and **automatic**.
The user chooses at runtime via CLI flag or ComfyUI workflow toggle.

## Current State

We already have separate workflow files for manual vs automatic paths:

| Stage | Manual (Browser) | Automatic (CLI) |
|-------|-----------------|-----------------|
| Pose/Body | `05_sprite_pose_edit.json` (VNCCS_PoseStudio) | `05_sprite_bodymesh_edit.json` (BodyMeshRenderer) |
| Motion | Manual pose sliders in Pose Studio | HY-Motion text-to-motion |
| Character | `00_z_image_character.json` (manual prompt) | PromptEnhancer auto-generates prompt |

The pipeline scripts already pick the right workflow:
- `run_sprite_pipeline.py` → `05_sprite_bodymesh_edit.json` (auto)
- `run_sprite_batch.py` → `05_sprite_bodymesh_edit.json` (auto)
- Browser user → `05_sprite_pose_edit.json` (manual)

## Implementation

### Pipeline-Level Mode Switch

Rather than modifying workflows (which would require ComfyUI-specific switching nodes),
we handle mode selection at the **pipeline runner level**:

```python
# In pipeline runner scripts:
WORKFLOWS = {
    "manual": {
        "pose": "05_sprite_pose_edit.json",    # VNCCS_PoseStudio (Three.js)
        "motion": "manual_slider_input",
    },
    "automatic": {
        "pose": "05_sprite_bodymesh_edit.json", # BodyMeshRenderer (Anny)
        "motion": "01_hymotion_keyframes.json", # HY-Motion DiT
    },
}

mode = os.environ.get("POSE_DIRECTOR_MODE", "automatic")
workflow_file = WORKFLOWS[mode]["pose"]
```

### CLI Integration

```bash
# Automatic mode (default): LLM/HY-Motion generates poses
python scripts/run_sprite_batch.py --animations idle --directions front

# Manual mode: load poses from JSON (user provides or from PoseLibrary)
python scripts/run_sprite_batch.py --poses Hi exercises1 --directions front

# HY-Motion mode: trained DiT generates poses
python scripts/run_sprite_batch.py --hymotion output/wave.npz --directions front
```

### Future: Combined Workflow with Toggle

If `comfyui-logic` or `ComfyUI-Impact-Pack` switching nodes are installed,
a single workflow could support both modes via a boolean widget:

```
[Mode Switch: automatic]
    ├── [automatic] → BodyMeshRenderer → OpenposePreprocessor → VNCCS_QWEN_Encoder → KSampler
    └── [manual]    → VNCCS_PoseStudio → VNCCS_QWEN_Encoder → KSampler
```

This requires:
1. Installing `comfyui-logic` (`cd custom_nodes && git clone https://github.com/EllangoK/ComfyUI-Logic`)
2. Creating a combined workflow JSON with both paths
3. Adding a boolean widget to switch between them

### Benefits of Dual-Mode

| Aspect | Manual | Automatic |
|--------|--------|-----------|
| Speed | Slow (hand-tune per pose) | Fast (batch, unattended) |
| Quality | Higher for hero characters | Good enough for NPCs |
| Control | Pixel-perfect | Limited to text description |
| Best for | Main character, bosses | Guards, villagers, crowds |
| Iteration | Visual feedback in browser | Vision evaluation + retry |

## Relation to Other Plans

- **Plan 1 (sprites)**: Both modes produce the same sprite output format
- **Plan 2 (TRELLIS)**: Manual = TRELLIS GUI, Automatic = Ray SDK
- **Plan 3 (batch)**: Always uses automatic mode, benefits from retry logic
