# TRELLIS Alternative Pipeline Design

## Vision

Instead of the current fragile chain (LLM → 2D mesh → VNCCS image edit → LTX video),
build a character once as a 3D model, then render it from any angle/pose directly.

**One-and-done character creation. Infinite reuse.**

## Pipeline Overview

```
Stage A: Character Concept    Z-Image Turbo → character portrait
Stage B: 3D Model Creation    TRELLIS (single image → 3D GLB mesh with textures)
Stage C: Motion Data           HY-Motion → joint rotations + translation (rot6d, 22 joints)
Stage D: 3D Rendering          Apply motion to mesh → render keyframes from any camera angle
Stage E: Video                 Stitch rendered frames → MP4
```

## Key Differences from Current Pipeline

| Current Pipeline | TRELLIS Pipeline |
|-----------------|------------------|
| 2D mesh keyframes (gray silhouettes) | Full-color 3D character model |
| VNCCS tries to match pose from mesh | Direct 3D rendering, no "matching" needed |
| LTX video assembly (guided diffusion) | Direct frame rendering, no diffusion needed |
| ~20 min per run | Potentially much faster for re-renders |
| Quality depends on VNCCS edit accuracy | Pixel-perfect consistency |
| Character looks different in each frame | Same 3D model, guaranteed consistency |

## Stage Details

### Stage A: Character Concept (unchanged)
- Z-Image Turbo → 768x1024 character portrait
- Workflow: `00_z_image_character.json`
- ~50s

### Stage B: 3D Model Creation (NEW — one-time)
- **Input**: Single character portrait image
- **Tool**: TRELLIS 2.0 (`Trellis2ImageTo3DPipeline`)
- **Model**: `microsoft/TRELLIS.2-4B` (already downloaded at `~/Documents/models/3d/trellis/`)
- **Output**: GLB file with textured mesh (~1M faces, 4K texture)
- **Implementation options**:
  1. Python script calling TRELLIS directly (standalone, outside ComfyUI)
  2. ComfyUI custom node wrapping TRELLIS (requires `ComfyUI_TRELLIS` extension)
- **VRAM**: ~8-12GB for TRELLIS inference
- **Time**: ~30-60s per model
- **One-and-done**: Create once, reuse across all animations

### Stage C: Motion Data (unchanged)
- HY-Motion 1.0 → text-to-motion DiT
- Output: rot6d (22 joints) + translation per frame
- Workflow: `01_hymotion_keyframes.json`
- ~380s

### Stage D: 3D Rendering (NEW — replaces VNCCS edit)
- **Input**: TRELLIS GLB mesh + HY-Motion joint rotations
- **Challenge**: TRELLIS mesh has NO skeleton/rigging
  - Option 1: Auto-rig (Mixamo, AccuRIG) → then apply HY-Motion joints
  - Option 2: Approximate deformation using nearest-vertex-to-joint mapping
  - Option 3: Skip posing, just rotate camera around static mesh (turntable)
- **Output**: Keyframe images from any camera angle
- **Rendering**: pyrender / nvdiffrast (TRELLIS includes nvdiffrast)

### Stage E: Video (simplified)
- Stitch rendered frames → MP4
- No diffusion needed — direct rendering
- Could use ffmpeg directly or VHS_VideoCombine

## Implementation Complexity

### Easy (turntable mode)
- Just render the TRELLIS mesh from rotating camera angles
- No posing needed
- Result: character rotating in 3D, like a figurine
- Good for: showcases, character reveals

### Medium (camera-only animation)
- Use HY-Motion translation data to move the camera
- Mesh stays static, camera follows motion path
- Result: cinematic camera movement around character
- Good for: simple scenes, orbiting shots

### Hard (full skeletal animation)
- Need to rig the TRELLIS mesh with a skeleton
- Map HY-Motion's 22 joints to the rig
- Forward kinematics → mesh deformation → render
- Tools: AccuRIG (free, command-line), Mixamo (web), or custom weight painting
- Result: fully animated character with correct poses
- This is the gold standard but most complex

## TRELLIS Installation Status

- **TRELLIS 2.0**: Installed at `~/Documents/programs/TRELLIS.2/` (or `~/Documents/programs/ray/infra/repos/TRELLIS.2/`)
- **Model weights**: `~/Documents/models/3d/trellis/TRELLIS.2-4B/`
- **ComfyUI extension**: NOT installed yet (`smthemex/ComfyUI_TRELLIS`)
- **Dependencies**: needs `o_voxel`, `utils3d`, `nvdiffrast`, custom CUDA ops
- **Ray deployment**: Already deployed at `/3d/trellis` via `services/creative/trellis.py`
- **Ray SDK**: `client.generate_3d(image_bytes)` → GLB bytes
- **Job API**: `POST /jobs/trellis` for async generation

## VNCCS Character Sheet Integration

VNCCS has a character sheet workflow (`VN_Step1_QWEN_CharSheetGenerator_v1.json`)
that generates multi-angle views. Could feed these to TRELLIS for better 3D results,
but TRELLIS currently takes single-image input. The multi-angle LoRA
(`qwen-image-edit-2511-multiple-angles-lora.safetensors`) could generate
consistent multi-angle views from a character portrait, then we pick the best one
for TRELLIS input.

## Next Steps

1. ~~Install `ComfyUI_TRELLIS` extension OR run TRELLIS standalone~~ — Ray deployment already handles this
2. Build Stage B: character image → 3D GLB via Ray SDK (`client.generate_3d()`)
3. Build Stage D: render GLB from multiple camera angles (turntable first)
4. Replace current pipeline's Stage D+E with TRELLIS rendering
5. (Future) Add skeletal rigging for full animation

### Ray Integration Path

Since TRELLIS is already deployed as a Ray Serve endpoint:

```python
from ray.sdk.client import RayClient

client = RayClient()
glb_bytes = await client.generate_3d(image_bytes)
```

Or via job API for async:
```bash
curl -X POST http://localhost:8000/jobs/trellis -F "image=character.png"
```

The Pose Director TRELLIS script should:
1. Use the Ray SDK client for 3D generation
2. Use pyrender for turntable rendering (no ComfyUI needed for this stage)
3. Output rendered keyframes in the same format as the sprite pipeline
