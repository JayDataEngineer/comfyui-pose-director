# Plan 4: Dual-Mode Workflows

## Status: Implemented

## Concept

Every pipeline stage supports two paths: **manual** and **automatic**.
The user chooses at runtime via CLI flag or by setting the ImpactSwitch `select` value.

## Implementation

### Combined Workflow: `05_sprite_dual_mode_edit.json`

A single workflow file with both paths, using `ImpactSwitch` from comfyui-impact-pack:

```
LoadImage(1) → character photo

BodyMeshRenderer(13) → IMAGE ──┐
                                ├── ImpactSwitch(15) → OpenposePreprocessor(3) → IMAGE
VNCCS_PoseStudio(14) → IMAGE ──┘                              │
         select=1 = auto        select=2 = manual              │
                                                               ↓
                                              VNCCS_QWEN_Encoder(9).image1 ← ImpactSwitch
                                              VNCCS_QWEN_Encoder(9).image2 ← LoadImage
                                              VNCCS_QWEN_Encoder(9).image3 ← OpenposePreprocessor
                                                        ↓
                                              KSampler(10) → VAEDecode(11) → SaveImage(12)
```

**Mode switching**:
- `select=1` → BodyMeshRenderer (automatic: Anny 3D mesh, rotations from PoseLibrary/HY-Motion)
- `select=2` → VNCCS_PoseStudio (manual: Three.js browser-based pose editor)

**Shared nodes** (4-12): UnetLoaderGGUF, CLIPLoader, VAELoader, LoRA x2, VNCCS_QWEN_Encoder, KSampler, VAEDecode, SaveImage

### CLI Integration

```bash
# Automatic mode (default): BodyMeshRenderer + PoseLibrary/HY-Motion
python scripts/run_sprite_batch.py --animations idle --directions front

# Explicit auto mode
python scripts/run_sprite_batch.py --animations idle --directions front --mode auto

# Manual mode: VNCCS_PoseStudio (load pose from JSON)
python scripts/run_sprite_batch.py --poses Hi exercises1 --directions front --mode manual

# HY-Motion mode: trained DiT generates poses
python scripts/run_sprite_batch.py --hymotion output/wave.npz --directions front
```

### Pipeline Script Changes

`scripts/run_sprite_batch.py`:
- `run_single_frame()` now takes `mode` param ("auto" or "manual")
- Loads `05_sprite_dual_mode_edit.json` instead of `05_sprite_bodymesh_edit.json`
- Sets `ImpactSwitch.select` via `patch_ui_workflow`
- `--mode` CLI flag (default: "auto")

### UI-to-API Converter

`pipeline/ui_to_api.py`:
- Added `ImpactSwitch` to `NODE_SCHEMAS`: `["select", "sel_mode"]`

### Workflow Files

| File | Description |
|------|-------------|
| `05_sprite_dual_mode_edit.json` | **Combined** workflow with ImpactSwitch mode toggle |
| `05_sprite_bodymesh_edit.json` | Auto-only (BodyMeshRenderer) — kept for backward compat |
| `05_sprite_pose_edit.json` | Manual-only (VNCCS_PoseStudio) — kept for backward compat |

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
