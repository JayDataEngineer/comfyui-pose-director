"""Build orchestrator — DAG construction, tier filtering, execution loop."""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from build_v2.context import BuildContext
from build_v2.dag import build_adjacency, topological_sort
from build_v2.state import (
    compute_yaml_hash,
    load_build_state,
    save_build_state,
    stage_needs_rebuild,
    update_stage_state,
)
from build_v2.validators import get_enabled_stages, infer_tier, validate_character
from build_v2.workflow_executor import WorkflowExecutor

# Import all stages
from build_v2.stages.generate import GenerateStage
from build_v2.stages.sprites_animated import SpritesAnimatedStage


def log(msg: str):
    print(f"[build] {msg}", flush=True)


# Registry of all available stages (name -> class)
ALL_STAGE_CLASSES = [
    GenerateStage,
    SpritesAnimatedStage,
    # More stages added in Phase 2-3:
    # SheetStage, EmotionsStage, SpritesStaticStage,
    # TrellisStage, OutfitStage, StateStage, VideoStage, GodotStage,
]


class BuildOrchestrator:
    """Executes a character build according to the DAG."""

    def __init__(self, ctx: BuildContext):
        self.ctx = ctx
        self.executor = WorkflowExecutor(ctx.comfyui_url)
        self.state = load_build_state(ctx.output_dir)

    def _create_stages(self) -> dict[str, object]:
        """Instantiate stages filtered by enabled packages and tier."""
        enabled = get_enabled_stages(self.ctx.character_data, self.ctx.tier)
        stages = {}
        for cls in ALL_STAGE_CLASSES:
            if cls.name in enabled:
                stage = cls(self.ctx)
                stage.executor = self.executor  # Inject executor
                stages[stage.name] = stage
        return stages

    def _needs_rebuild(self, stage) -> bool:
        """Check if a stage needs to run."""
        # Forced rebuild?
        yaml_hash = compute_yaml_hash(self.ctx.character_data)
        if stage_needs_rebuild(
            self.state, stage.name, yaml_hash, self.ctx.rebuild_targets
        ):
            return True
        # Already complete and artifacts exist?
        if stage.is_complete(self.state) and stage.artifacts_exist(self.state):
            return False
        return True

    def build(self):
        """Execute the build DAG."""
        ctx = self.ctx

        # Handle --clean
        if ctx.dry_run and "--clean" in sys.argv:
            pass  # Don't actually clean in dry-run

        # Handle --clean
        if not ctx.dry_run and ctx.output_dir.exists():
            # Check if clean was requested via rebuild_targets
            if "all" in ctx.rebuild_targets:
                shutil.rmtree(ctx.output_dir, ignore_errors=True)
                log(f"Cleaned: {ctx.output_dir}")

        # Ensure output dir exists
        ctx.output_dir.mkdir(parents=True, exist_ok=True)

        # Create stages
        stages = self._create_stages()
        if not stages:
            log("No stages to run")
            return

        # Build DAG and sort
        graph = build_adjacency(list(stages.values()))
        # Only include stages that are in our filtered set
        filtered_graph = {
            name: [d for d in deps if d in stages]
            for name, deps in graph.items()
            if name in stages
        }
        order = topological_sort(filtered_graph)

        # Determine what needs building
        todo = []
        for name in order:
            if name not in stages:
                continue
            stage = stages[name]
            if self._needs_rebuild(stage):
                todo.append(stage)
            else:
                log(f"[{name}] up to date, skipping")

        if not todo:
            log("Nothing to build — all assets up to date")
            return

        log(f"Build plan: {len(todo)} stages")
        if ctx.dry_run:
            for stage in todo:
                log(f"  [{stage.name}] would run (tier={stage.tier_required})")
            return

        # Update yaml hash
        yaml_hash = compute_yaml_hash(ctx.character_data)
        self.state["yaml_hash"] = yaml_hash
        self.state["character"] = ctx.character_name

        # Execute in order
        start = time.time()
        failed = []

        for stage in todo:
            # Check prerequisites
            prereqs = stage.check_prerequisites(self.state)
            if prereqs:
                log(f"[{stage.name}] skipping — prerequisites not met: {prereqs}")
                update_stage_state(self.state, stage.name, "skipped", error=f"Missing prerequisites: {prereqs}")
                continue

            log(f"[{stage.name}] running...")
            stage_start = time.time()
            try:
                result = stage.run(self.state)
                duration = time.time() - stage_start

                update_stage_state(
                    self.state,
                    stage.name,
                    result.status,
                    artifacts=result.artifacts,
                    input_hash=yaml_hash,
                    duration_s=duration,
                    meta=result.meta,
                )
                log(f"[{stage.name}] {result.status} ({duration:.1f}s)")

            except Exception as e:
                duration = time.time() - stage_start
                log(f"[{stage.name}] FAILED: {e}")
                failed.append(stage.name)
                update_stage_state(
                    self.state,
                    stage.name,
                    "failed",
                    duration_s=duration,
                    error=str(e),
                )

            # Save state after each stage
            save_build_state(ctx.output_dir, self.state)

        elapsed = time.time() - start
        succeeded = len(todo) - len(failed)
        log(f"Build complete: {succeeded}/{len(todo)} stages in {elapsed:.1f}s")
        if failed:
            log(f"  Failed: {failed}")
