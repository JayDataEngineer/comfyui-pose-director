# Plan 4: Dual-Mode Workflows

## Status: Tentative

## Concept

Every workflow supports two paths: **manual** and **automatic**. The user chooses
at runtime. Manual mode gives full control (VNCCS Pose Studio sliders, hand-tuned
parameters). Automatic mode lets HY-Motion and the pipeline handle everything.

## Why

Right now our workflows are single-path. If you want to pose a character, you either:
- Use VNCCS Pose Studio (manual, slow, precise) — good for hero characters
- Use HY-Motion → VNCCS edit (automatic, fast, less control) — good for NPCs

These should be the same workflow with a mode switch, not two separate workflows.

## Implementation

### ComfyUI Branching

Use `comfyui-logic` or `comfyui-impact-pack` nodes for conditional execution:

```
                    ┌─ Mode Switch ─────────────────────┐
                    │                                    │
              [Manual]                            [Automatic]
                    │                                    │
                    ▼                                    ▼
         VNCCS_PoseStudio                    HY-Motion Generate
         (user adjusts sliders)              (text → trained DiT)
                    │                                    │
                    ▼                                    ▼
              VNCCS_PoseStudio                      Convert to VNCCS
              mesh render                           pose_data format
                    │                                    │
                    └──────────┬─────────────────────────┘
                               ▼
                     VNCCS_QWEN_Encoder
                     (3-image conditioning)
                               ▼
                         KSampler (4-step)
                               ▼
                          VAEDecode
                               ▼
                         SaveImage
```

### Mode Switch Options

1. **comfyui-logic `If` node** — boolean widget toggles between paths
2. **comfyui-impact-pack `ImageBatch`** — merge both paths, use only one
3. **ComfyUI native `mode` property** — set node `mode: 4` (bypass) on the unused path
4. **Workflow-level switch** — two workflow files, pipeline script picks the right one

Option 4 is simplest and what we already do (separate workflow files). But having
a single combined workflow with a toggle is better for the ComfyUI UI experience.

### Where Dual-Mode Applies

| Workflow | Manual Path | Automatic Path |
|----------|------------|----------------|
| Character (Stage A) | Manual prompt, hand-tuned params | PromptEnhancer generates prompt |
| Pose/Motion (Stage B) | VNCCS Pose Studio sliders | HY-Motion text-to-motion |
| Edit (Stage C) | Manual instruction tweaking | Default VNCCS instruction |
| Video (Stage E) | Manual keyframe placement | Auto keyframe at regular intervals |

### Use Cases

**Manual mode** (hero characters, key animations):
- Main character idle/walk cycle — needs to be perfect
- Boss attack animations — signature moves
- Cutscene poses — dramatic moments
- When the user wants pixel-perfect control

**Automatic mode** (NPCs, batch generation):
- Guard idle/walk — generic but consistent
- Villager animations — don't need to be unique
- Batch generation for Plan 3 (autonomous batch)
- Quick prototyping before manual refinement

### Hybrid Approach

Start in automatic mode, then switch to manual for refinement:
1. HY-Motion generates initial poses
2. User sees the preview in ComfyUI
3. Adjusts specific bones in VNCCS Pose Studio
4. Re-runs only the VNCCS edit stage

This is the natural workflow: automatic draft → manual polish.

## Technical Notes

### comfyui-logic Installation
- Already installed? Need to check
- Provides: `If`, `Switch`, `Compare` nodes for conditional execution

### comfyui-impact-pack
- Already installed (used for `ImpactImageBatchToImageList`)
- Provides additional routing/switching nodes

### Workflow Design Pattern
```json
{
  "nodes": [
    {"type": "ModeSwitch", "widgets_values": ["automatic"]},
    {"type": "VNCCS_PoseStudio", "mode": 4, "id": 100},
    {"type": "HYMotionGenerate", "mode": 0, "id": 200},
    ...
  ]
}
```

Setting `mode: 4` on a node bypasses it (ComfyUI native feature). The mode switch
would toggle mode between 0 (active) and 4 (bypass) on the appropriate path nodes.

## Relation to Other Plans

- Plan 1 (sprites): Dual-mode means manual OR automatic sprite generation
- Plan 2 (TRELLIS): Manual camera placement OR automatic camera paths
- Plan 3 (batch agent): Always uses automatic mode, but benefits from the same workflow structure
