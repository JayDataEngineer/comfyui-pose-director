# Declarative Build System — Profile & Documentation

## Architecture

**Script**: `scripts/build_character.py` (~1100 lines)
**Character YAMLs**: `characters/{name}.yaml`
**Output**: `output/{name}/` directory tree

### Pipeline Stages (in order)

| Stage | Description | ComfyUI Workflow | Time/Step |
|-------|-------------|------------------|-----------|
| generation | Z-Image (turbo/base) or import | `00_z_image_character.json` | ~50s (warm), ~5min (cold) |
| trellis | 3D model via TRELLIS (T2+ only) | subprocess fallback | ~2min if available, skipped if not |
| portraits | VNCCS edit with emotion instruction | `05_sprite_bodymesh_edit.json` | ~30-40s per emotion |
| sprites | Motion NPZ → BodyMesh → VNCCS edit per frame | `05_sprite_bodymesh_edit.json` | ~15-20s per frame |
| outfit | VNCCS outfit ref + re-run sprites | `05_sprite_bodymesh_edit.json` | ~30s ref + ~15s per frame |
| state | VNCCS edit with state description | `05_sprite_bodymesh_edit.json` | ~30-40s per variant |

### Key Workflows Used

1. **`00_z_image_character.json`** — Character generation
   - UNETLoader(z_image_turbo_bf16, ~12GB) + CLIPLoader(qwen_3_4b, ~7.5GB) + VAELoader(ae)
   - KSampler: 8 steps, seed fixed, euler
   - Output: 768x1024 character image

2. **`01_hymotion_keyframes.json`** — Motion NPZ generation
   - HYMotionLoadLLM → EncodeText → Generate → SaveNPZ
   - NOT AVAILABLE ON REMOTE — causes failures for `motion: custom`
   - NPZ saved to ComfyUI output dir, must be copied to `motions/custom/`

3. **`05_sprite_bodymesh_edit.json`** — Sprite frame generation (THE WORKHORSE)
   - BodyMeshRenderer(Anny 3D, rotations_json) → mesh render image
   - VNCCS_QWEN_Encoder(character + mesh + openpose, Lightning 4-step)
   - KSampler: 4 steps, cfg=1, euler, simple
   - SaveImage → output frame
   - Each call = ~15-20s per frame

### Hardware Requirements

| Component | VRAM | Notes |
|-----------|------|-------|
| Z-Image (character gen) | ~15GB | 12GB UNET + 7.5GB CLIP (overlapped) |
| VNCCS edit (sprites/portraits) | ~20GB | 14GB UNET GGUF + 8.8GB CLIP + VAE + LoRAs |
| HY-Motion (motion gen) | ~11GB | 4GB motion model + 1.7GB CLIP + Qwen3 CPU |
| BodyMeshRenderer (Anny) | ~0.5GB | CPU + pyrender, minimal VRAM |

**Local RTX 2070 Super (8GB)**: Cannot run ANY stage. Must use remote.
**Remote RTX 4090 (24GB)**: Runs all stages. ComfyUI started standalone on port 18465.

### Remote Setup

```
ComfyUI: /home/user/Documents/programs/ray/infra/repos/ComfyUI/
Models:  /home/user/Documents/models/image-gen/comfyui/
Custom nodes: ComfyUI/custom_nodes/comfyui-pose-director/
```

Start ComfyUI on remote:
```bash
ssh user@100.86.69.57
cd ~/Documents/programs/ray/infra/repos/ComfyUI
nohup .venv/bin/python main.py --port 18465 --listen 0.0.0.0 > /tmp/comfyui_remote.log 2>&1 &
```

Access: `http://100.86.69.57:18465` (Tailscale)

**IMPORTANT**: Ray Serve proxy (`:18800/comfyui`) returns 500 errors. Use direct ComfyUI instead.

**IMPORTANT**: llama.cpp Docker (`orchestrator-llama`) must be stopped before ComfyUI on local:
```bash
docker update --restart=no orchestrator-llama && docker stop orchestrator-llama
```

### Build Command

```bash
# From ComfyUI root on remote:
COMFYUI_URL=http://127.0.0.1:18465 .venv/bin/python \
  custom_nodes/comfyui-pose-director/scripts/build_character.py \
  custom_nodes/comfyui-pose-director/characters/{name}.yaml

# Options:
--dry-run    # Show plan without executing
--clean      # Delete output dir and rebuild from scratch
--seed N     # Override seed
--rebuild sprites.casual.normal.idle  # Force rebuild specific target
```

### Build State

File: `output/{name}/build_state.json`

Tracks: yaml_hash, generation status/portraits/sprites/outfits/states with per-item completion.

Incremental builds: only rebuilds items not marked `status: complete`.

### Bugs Found & Fixed

1. **`file_hash()` infinite loop** — `lambda: open(path, "rb").read(8192)` opens new file handle each iteration, always reads from position 0. Never reaches EOF. **Fix**: open file once with `with` statement.

2. **Import image path resolution** — `Path(details["image"])` resolved relative to CWD, not YAML file. **Fix**: resolve relative to `_yaml_path` stored in data dict.

3. **Shared motion fallback** — Non-standard animation names (attack, die, run) had no shared NPZ files. **Fix**: fallback to `idle.npz` when specific file not found.

4. **Outfit `die()` kills build** — `die()` on outfit reference failure terminates entire build. **Fix**: changed to `raise RuntimeError()` caught by main loop's try/except.

5. **Outfit reference regenerated per animation** — Each outfit animation step regenerates the reference. **Fix**: check if outfit ref already exists, reuse it.

6. **Rebuild flag dot/slash mismatch** — `sprites.casual.normal.idle` vs `sprites/casual/normal/idle`. **Fix**: normalize with `prefix.replace(".", "/")`.

### Performance Benchmarks

| Character | Steps | Total Time | PNGs | Notes |
|-----------|-------|------------|------|-------|
| shopkeeper (T1) | 5/5 | 396s (6.6min) | 15 | 1 gen + 2 portraits + 2 sprites (front, 6 frames each) |
| shadow_beast (T1) | 6/6 | ~22min | 75 | 1 import + 2 portraits + 3 sprites (4 dirs x 6 frames each) |
| marty (T2) | 31/31* | ~45min total | 238 | 1 gen + 10 portraits + 4x4 sprites + 4x4 outfits + 10+1 states |
| | | | | *TRELLIS skipped (not available on remote) |

**Bottleneck**: VNCCS edits (~15-20s per frame). Each frame = BodyMesh render + VNCCS 4-step edit.

**Speed optimization**: Could batch all frames for same direction into single KSampler call if VNCCS supports batched latents.

### Motion Library

| Motion | File | Size | Source |
|--------|------|------|--------|
| idle | `motions/shared/idle.npz` | 73KB | HY-Motion generated locally |
| walk | `motions/shared/walk.npz` | 108KB | HY-Motion generated locally |

**Missing motions**: run, attack, die, and any custom motions → fallback to idle.npz

**Custom motions**: Require HY-Motion (not on remote). Need to either:
1. Install HY-Motion on remote
2. Generate locally and rsync
3. Pre-generate and store in `motions/shared/`

### Port Forwarding (Local Development)

To run builds from local machine against remote ComfyUI:
```bash
ssh -L 18465:localhost:18465 user@100.86.69.57
COMFYUI_URL=http://127.0.0.1:18465 uv run python scripts/build_character.py characters/shopkeeper.yaml
```

### Sync Commands

```bash
# Sync code to remote
rsync -avz --delete \
  --exclude='__pycache__' --exclude='.venv' --exclude='.git' \
  --exclude='output' --exclude='.env' --exclude='.claude' \
  --exclude='.github' --exclude='.mcp.json' --exclude='docs' \
  --exclude='docs-site' --exclude='conv*.txt' \
  /home/ubuntu/Documents/programs/img/comfyui-pose-director/ \
  user@100.86.69.57:~/Documents/programs/ray/infra/repos/ComfyUI/custom_nodes/comfyui-pose-director/

# Copy outputs back
rsync -avz user@100.86.69.57:~/Documents/programs/ray/infra/repos/ComfyUI/custom_nodes/comfyui-pose-director/output/{name}/ \
  /home/ubuntu/Documents/programs/img/comfyui-pose-director/output/{name}/
```
