# Codebase Audit — Rolling Tracker

Last audit: 2026-05-05

## Dead Code

| # | Item | Status | Location |
|---|------|--------|----------|
| D1 | `workflows/VNCCS_Utils Pose Studio.json` | ✅ Deleted | Unused, not imported anywhere |
| D2 | `docs/alpha_workflows/` | ⚪ Low priority | Old planning docs, reference-only |

## Stale References

| # | Item | Status | Location |
|---|------|--------|----------|
| S1 | MEMORY.md stale refs | ✅ Fixed | Already correct |

## DRY Violations

| # | Duplication | Count | Status | Fix |
|---|-------------|-------|--------|-----|
| R1 | `vnccs_char_name = "None"` workaround | 6 stages | ✅ | `VNCCS_CHAR_NAME` constant on context |
| R2 | `_RMBG_MODEL = "BEN2"` override | 4 stages | ✅ | `DEPLOYMENT_RMBG_MODEL` constant |
| R3 | `ctx.character_data.get("packages", {}).get(X, {})` pattern | 9 occurrences | ✅ | `ctx.pkg()` helper on BuildContext |
| R4 | Artifact saving loop (enumerate + write_bytes + relative_to) | 5 stages | ✅ | `_save_artifacts()` on base Stage |
| R5 | Try/except → `StageResult(status="failed")` wrapper | 12 occurrences | ⚪ Low priority | Each has unique error messages, not worth abstracting |
| R6 | `checkpoint_override="sd_xl_base_1.0.safetensors"` | 3 stages | ✅ | `DEPLOYMENT_CHECKPOINT` constant |

## Godfiles

| # | File | Lines | Status | Fix |
|---|------|-------|--------|-----|
| G1 | `tests/unit/test_build_v2.py` | 757 | ✅ | Split into 4 files: dag, core, stages, strategy |

## Logic Separation

| # | Issue | Status | Location |
|---|-------|--------|----------|
| L1 | validators.py mixes validation + tier inference + stage enablement | ⚪ Low priority | Cohesive enough for current scale |
| L2 | Deployment overrides hardcoded in stage bodies | ✅ (covered by R2/R6) | Now constants on context |

## Data Access Consistency

| # | Issue | Status | Location |
|---|-------|--------|----------|
| A1 | `sprites_animated` used `character_data.get("sprites")` (top-level only) in 5 places while `sprites_static` correctly used `ctx.pkg("sprites")` | ✅ Fixed | All 5 locations now use `ctx.pkg("sprites")` |

## Strategy Composition

| # | Concern | Status | Notes |
|---|---------|--------|-------|
| C1 | `body_reference` only consumed by `sprites_static`, not `sprites_animated` | ❌ Gap | Both stages render a body reference — TRELLIS creates skeletons too. Animated stage should respect body_reference. |
| C2 | `motion_strategy` only consumed by `sprites_animated`, not `sprites_static` | ❌ Gap | Static sprites also go through motion generation (pose render → QWEN edit). Both need motion data. |
| C3 | `face_refinement` cascade to downstream stages | ✅ Natural | Sheet post-processing cascades through DAG automatically |
| C4 | `generation.source` encapsulation | ✅ Clean | All downstream stages only depend on `approved_image` existing |
| C5 | `video` × `sprites_animated` independence | ✅ Clean | Independent output pipelines (LTX video vs game engine frames) |
