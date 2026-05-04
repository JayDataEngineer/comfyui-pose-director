"""Build state management — load, save, hash checking for incremental builds."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


def compute_yaml_hash(data: dict) -> str:
    """Hash the YAML content for change detection (excludes internal keys)."""
    clean = {k: v for k, v in data.items() if not k.startswith("_")}
    content = json.dumps(clean, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def file_hash(path: Path) -> str:
    """SHA256 of a file (first 16 hex chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:16]}"


def load_build_state(output_dir: Path) -> dict:
    """Load build state from character output directory."""
    state_file = output_dir / "build_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {
        "character": "",
        "yaml_hash": "",
        "last_build": "",
        "stages": {},
    }


def save_build_state(output_dir: Path, state: dict):
    """Save build state atomically."""
    state["last_build"] = datetime.now().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_file = output_dir / "build_state.json"
    tmp_file = output_dir / ".build_state.tmp.json"

    with open(tmp_file, "w") as f:
        json.dump(state, f, indent=2)
    tmp_file.rename(state_file)


def update_stage_state(
    state: dict,
    stage_name: str,
    status: str,
    artifacts: list[str] | None = None,
    input_hash: str = "",
    duration_s: float = 0.0,
    error: str = "",
    meta: dict | None = None,
):
    """Update a single stage's entry in the build state."""
    if "stages" not in state:
        state["stages"] = {}

    state["stages"][stage_name] = {
        "status": status,
        "started_at": state.get("stages", {})
        .get(stage_name, {})
        .get("started_at", datetime.now().isoformat()),
        "completed_at": datetime.now().isoformat() if status in ("complete", "failed", "skipped") else "",
        "input_hash": input_hash,
        "artifacts": artifacts or [],
        "duration_s": duration_s,
        "error": error,
        "meta": meta or {},
    }


def stage_needs_rebuild(
    state: dict,
    stage_name: str,
    current_input_hash: str,
    rebuild_targets: list[str],
) -> bool:
    """Check if a stage needs to be (re)built.

    Returns True if:
    - Stage not in state or not complete
    - Input hash changed (YAML changed)
    - Stage is in rebuild_targets
    """
    # Check rebuild targets
    if rebuild_targets:
        for target in rebuild_targets:
            if target == "all" or target == stage_name:
                return True

    stage_state = state.get("stages", {}).get(stage_name, {})

    # Not complete yet
    if stage_state.get("status") != "complete":
        return True

    # Input hash changed
    old_hash = stage_state.get("input_hash", "")
    if old_hash and current_input_hash and old_hash != current_input_hash:
        return True

    return False
