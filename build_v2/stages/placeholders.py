"""Placeholder stages for not-yet-implemented pipeline steps.

These stages exist so the DAG resolves and the build system runs without error.
They report "skipped" with an explanation rather than failing.
"""

from __future__ import annotations

from build_v2.stages.base import Stage, StageResult


class _Placeholder(Stage):
    """Stage that reports skipped without error."""
    reason: str = "not yet implemented"

    def is_complete(self, state: dict) -> bool:
        return False

    def run(self, state: dict) -> StageResult:
        return StageResult(status="skipped", error=self.reason)


class TrellisStage(_Placeholder):
    name = "trellis"
    tier_required = "t2"
    depends_on = ["generate"]
    reason = "TRELLIS 3D model generation not yet implemented"


class StateStage(_Placeholder):
    name = "state"
    tier_required = "t1"
    depends_on = ["sprites_static"]
    reason = "State condition re-edit not yet implemented"


class VideoStage(_Placeholder):
    name = "video"
    tier_required = "t1.5"
    depends_on = ["sprites_animated"]
    reason = "LTX Video assembly not yet implemented"


class GodotStage(_Placeholder):
    name = "godot"
    tier_required = "t1"
    depends_on = ["sprites_static", "sprites_animated"]
    reason = "Godot export not yet implemented"
