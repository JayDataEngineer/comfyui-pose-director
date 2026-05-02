# Plan 3: Autonomous Batch Pipeline

## Status: Implemented

## Ray Integration

The batch pipeline supports running through the Tech Noir Ray infrastructure
(`~/Documents/programs/ray/`), enabling cloud access instead of local-only.

```bash
# Local mode (default): talks directly to ComfyUI
python scripts/run_autonomous_batch.py --npcs npcs/example_npcs.yaml

# Ray mode: talks to Ray gateway proxy
COMFYUI_URL=http://localhost:8000/comfyui python scripts/run_autonomous_batch.py --npcs npcs/example_npcs.yaml

# Remote Ray: from any machine with network access
COMFYUI_URL=https://your-ray-host:8000/comfyui python scripts/run_autonomous_batch.py --npcs npcs/example_npcs.yaml
```

The `pipeline/comfyui_client.py` module reads `COMFYUI_URL` from env var.
All scripts that import it automatically support both modes.

## Concept

A vLLM agent orchestrates the entire pipeline autonomously for batch NPC generation.
Give it a brief ("make an animation suite for a dark mage NPC, animations: idle, walk,
attack, hurt, death") and it runs every stage, evaluates quality, and retries or adjusts
as needed. No human in the loop.

## Why This Is Different From Current Pipeline

Current `run_full_pipeline.py` is a linear script — it runs stages A→B→C→D→E no matter what.
If Stage A produces a bad character, it still proceeds. No evaluation, no retry, no adjustment.

The autonomous agent adds:
- **Vision evaluation between stages** — does this character look right? Is the pose correct?
- **Autonomous retry** — bad output? Adjust a parameter and try again
- **Batch processing** — generate 20 NPCs with animation suites in one job
- **Self-correction** — agent sees artifacts and adjusts prompts/steps/params

## Pipeline Flow

```
Input: "Make animation suite for a dark mage NPC"
       Animations: idle, walk, attack, hurt, death
       Optional: character portrait / character sheet

                ┌─────────────────────────────────┐
                │         vLLM Agent Loop          │
                │                                  │
                │  1. Z-Image → character portrait │
                │     ↓ evaluate (vision)          │
                │     retry? adjust prompt/seed    │
                │                                  │
                │  2. HY-Motion → poses per anim   │
                │     ↓ evaluate (vision)          │
                │     retry? adjust description    │
                │                                  │
                │  3. VNCCS Edit → character+pose  │
                │     ↓ evaluate (vision)          │
                │     retry? adjust instruction    │
                │                                  │
                │  4. Sprite sheet assembly        │
                │     ↓ evaluate (completeness)    │
                │                                  │
                │  5. Next NPC or finish           │
                └─────────────────────────────────┘
```

## Key Design Points

### Agent Framework Options

1. **Raw** (recommended) — Python script with vision evaluation between stages.
   We already have `run_full_pipeline.py` + `utils/pose_evaluator.py`. Extend the
   existing pattern. No new dependencies.

2. **Pi-Mono** — if we need more sophisticated agent behavior (tool use, planning)

3. **OpenClaw** — hype, probably overkill for a linear pipeline with retry logic

The pipeline is fundamentally linear (A→B→C→D) with retry loops. A simple
while-loop with vision checks is probably sufficient. No need for a full agent
framework unless we want the agent to discover and install new ComfyUI nodes
mid-run (we don't).

### Vision Evaluation Per Stage

Each stage gets a quality check:

| Stage | Evaluation | Pass criteria |
|-------|-----------|---------------|
| A: Character | Does it match the description? Anatomically coherent? | Score > 0.7 |
| B: HY-Motion | Are the poses physically plausible? Correct motion type? | Score > 0.7 |
| C: VNCCS Edit | Character identity maintained? Pose matched? Artifacts? | Score > 0.7 |
| D: Sprite Sheet | All frames present? Consistent sizing? | Completeness check |

### Retry Strategy

```python
for attempt in range(max_retries):
    result = run_stage(stage, params)
    score = evaluate(result)
    if score > threshold:
        break
    params = agent.adjust(params, score, attempt)
```

The agent adjusts:
- Stage A: seed, prompt wording, steps
- Stage B: motion description, seed
- Stage C: VNCCS instruction, weights, seed
- Stage D: N/A (deterministic assembly)

### Batch Processing

```python
npcs = [
    {"name": "dark_mage", "description": "hooded figure, dark robes, glowing eyes"},
    {"name": "guard", "description": "armored knight, broad shoulders, helmet"},
    {"name": "villager", "description": "peasant woman, simple dress, brown hair"},
]

for npc in npcs:
    character = generate_character(npc["description"])
    for anim in ["idle", "walk", "attack", "hurt", "death"]:
        poses = generate_poses(anim)
        sprites = render_sprites(character, poses)
        pack_sprite_sheet(sprites, npc["name"], anim)
```

### Character Sheet as Optional Input

If a character portrait or sheet is provided (from VNCCS CharSheet generator or manual
illustration), skip Stage A and go straight to pose generation. This lets artists
provide their own character art and still use the batch pipeline for animation.

## Implementation Notes

### What We Already Have
- `run_full_pipeline.py` — linear pipeline runner
- `utils/pose_evaluator.py` — vision evaluation with Gemini Flash
- `pipeline/comfyui_client.py` — ComfyUI API submission
- `pipeline/ui_to_api.py` — workflow conversion
- `.env` with API keys
- **ComfyCozy MCP server** — 114 tools for autonomous ComfyUI control
  - The agent should use MCP tools: `load_workflow`, `execute_workflow`, `analyze_image`
  - Vision evaluation via `analyze_image` (quality scores, artifact detection)
  - Workflow patching via `apply_workflow_patch`, `set_input`
  - Progress tracking via `execute_with_progress`
  - This gives the agent full ComfyUI control without wrapping the API separately

### What Needs Building
- Stage-level retry loop with parameter adjustment
- Vision evaluation for character quality (not just pose quality)
- Batch character definition format (YAML/JSON with NPC descriptions)
- Sprite sheet assembly (use VNCCS SheetManager)
- Progress reporting for batch jobs

### VRAM / Concurrency
- Z-Image: ~6GB VRAM
- HY-Motion: ~11GB VRAM (must stop llama.cpp first)
- VNCCS Edit: ~8GB VRAM (QIE2511 GGUF + LoRAs)
- Cannot run stages in parallel on single GPU — sequential is fine for batch

## Relation to Other Plans

- Uses Plan 1 (sprite pipeline) as the output format
- Plan 4 (dual-mode) means the agent picks the automatic path
- Could optionally use Plan 2 (TRELLIS) if 3D models are needed instead of sprites
