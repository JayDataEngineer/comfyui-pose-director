"""Stage base class and result type for the DAG build system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class StageResult:
    """Outcome of a single stage execution."""

    status: str = "pending"  # "complete" | "failed" | "skipped"
    artifacts: list[str] = field(default_factory=list)  # relative paths from output root
    duration_s: float = 0.0
    error: str = ""
    meta: dict = field(default_factory=dict)  # stage-specific data (seed, params, etc.)


class Stage(ABC):
    """Base class for all pipeline stages.

    Each stage declares its tier requirement and dependencies via class attributes.
    The orchestrator instantiates stages, filters by tier, and runs them in DAG order.
    """

    name: str = ""
    tier_required: str = "t0"  # minimum tier to enable this stage
    depends_on: list[str] = []  # names of stages that must complete first

    def __init__(self, ctx: "BuildContext"):  # noqa: F821
        self.ctx = ctx

    @abstractmethod
    def is_complete(self, state: dict) -> bool:
        """Check if this stage's artifacts are already built."""

    @abstractmethod
    def run(self, state: dict) -> StageResult:
        """Execute the stage. Returns result with artifact paths."""

    def check_prerequisites(self, state: dict) -> list[str]:
        """Return list of missing prerequisites. Empty = ready to run."""
        missing = []
        for dep_name in self.depends_on:
            dep_state = state.get("stages", {}).get(dep_name, {})
            if dep_state.get("status") != "complete":
                missing.append(dep_name)
        return missing

    def get_artifact_paths(self, state: dict) -> list[str]:
        """Return expected output artifact paths (relative to output root)."""
        stage_state = state.get("stages", {}).get(self.name, {})
        return stage_state.get("artifacts", [])

    def artifacts_exist(self, state: dict) -> bool:
        """Check if all declared artifacts exist on disk."""
        for rel_path in self.get_artifact_paths(state):
            full_path = self.ctx.output_dir / rel_path
            if not full_path.exists():
                return False
        return True
