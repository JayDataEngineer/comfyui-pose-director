# Pipeline Plans — Overview

Pose Director has four pipeline plans. Each targets a different use case.
All share the same foundation: Z-Image (character) + HY-Motion (poses) + VNCCS (editing).

## The Plans

| # | Name | Status | One-liner |
|---|------|--------|-----------|
| 1 | **Sprite Pipeline** | COMPLETE | Game sprite sheets via VNCCS, AI-driven poses |
| 2 | **TRELLIS Pipeline** | Implemented (turntable) | 3D character model, render from any angle |
| 3 | **Autonomous Batch** | Implemented | Vision-evaluated batch NPC sprite generation |
| 4 | **Dual-Mode Workflows** | Implemented | ImpactSwitch toggles manual/automatic paths |

## Shared Foundation

```
Character:  Z-Image Turbo → portrait (all plans start here)
Motion:     HY-Motion 1.0 → trained DiT, NOT LLM guessing
Editing:    VNCCS QWEN Edit 2511 + Lightning 4-step + PoseStudio LoRA
Direction:  VNCCS modelRotation for multi-angle views
Output:     Depends on plan (sprites / 3D / video)
```

## How the Plans Relate

```
                ┌─────────────────────────┐
                │   Plan 4: Dual-Mode      │
                │   (manual OR automatic)  │
                │   Applies to ALL plans   │
                └────────┬────────────────┘
                         │
            ┌────────────┼────────────────┐
            ▼            ▼                ▼
   ┌────────────┐ ┌──────────────┐ ┌──────────────┐
   │ Plan 1:    │ │ Plan 2:      │ │ Plan 3:      │
   │ Sprites    │ │ TRELLIS      │ │ Batch Agent  │
   │            │ │              │ │              │
   │ Game ready │ │ 3D model     │ │ Autonomous   │
   │ 2D frames  │ │ One & done   │ │ NPC factory  │
   └────────────┘ └──────────────┘ └──────────────┘
```

Plan 4 (dual-mode) is a cross-cutting concern — it modifies how Plans 1-3 are
executed, not what they produce. Plan 3 (batch agent) orchestrates Plans 1 or 2
autonomously at scale.

## Files

| File | Description |
|------|-------------|
| `01_sprite_pipeline.md` | Game sprite sheets via VNCCS |
| `02_trellis_pipeline.md` | 3D character model from image |
| `03_autonomous_batch.md` | vLLM agent for batch NPC generation |
| `04_dual_mode_workflows.md` | Manual + automatic paths |
