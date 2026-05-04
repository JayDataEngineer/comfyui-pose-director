#!/usr/bin/env python3
"""Build V2 — DAG-based declarative character asset build system.

Usage:
    uv run python scripts/build_v2.py characters/marty.yaml
    uv run python scripts/build_v2.py characters/marty.yaml --dry-run
    uv run python scripts/build_v2.py characters/marty.yaml --rebuild generate
    uv run python scripts/build_v2.py characters/marty.yaml --clean
    uv run python scripts/build_v2.py characters/*.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Run: uv pip install pyyaml")
    sys.exit(1)

from build_v2.context import BuildContext
from build_v2.orchestrator import BuildOrchestrator, log
from build_v2.validators import infer_tier, validate_character
from pipeline.comfyui_client import COMFYUI_URL


def load_character_yaml(path: Path) -> dict:
    """Load and validate a character YAML manifest."""
    if not path.exists():
        log(f"ERROR: Character YAML not found: {path}")
        sys.exit(1)
    path = path.resolve()
    with open(path) as f:
        data = yaml.safe_load(f)
    data["_yaml_path"] = str(path)
    validate_character(data, path)

    # Auto-infer tier if not declared
    char = data.get("character", {})
    if "tier" not in char:
        inferred = infer_tier(data)
        char["tier"] = inferred
        log(f"  Inferred tier: {inferred}")

    return data


def check_comfyui(url: str):
    """Verify ComfyUI is reachable."""
    import requests
    try:
        r = requests.get(f"{url}/system_stats", timeout=5)
        assert r.status_code == 200
    except Exception as e:
        log(f"ERROR: ComfyUI not reachable at {url}: {e}")
        sys.exit(1)
    log(f"ComfyUI ready at {url}")


def build_character(
    yaml_path: Path,
    rebuild_targets: list[str] | None = None,
    seed_override: int | None = None,
    dry_run: bool = False,
    clean: bool = False,
):
    """Build a character from its YAML manifest."""
    data = load_character_yaml(yaml_path)
    char = data["character"]
    name = char["name"]
    tier = char.get("tier", "t1")

    log(f"Character: {name} (tier={tier})")

    comfyui_url = os.environ.get("COMFYUI_URL", COMFYUI_URL)

    # Handle --clean
    if clean:
        import shutil
        output_dir = PROJECT_ROOT / "output" / name
        if output_dir.exists():
            shutil.rmtree(output_dir)
            log(f"  Cleaned: {output_dir}")

    ctx = BuildContext.from_yaml(
        data=data,
        yaml_path=str(yaml_path),
        project_root=PROJECT_ROOT,
        comfyui_url=comfyui_url,
        seed_override=seed_override,
        dry_run=dry_run,
        rebuild_targets=rebuild_targets or [],
    )

    orchestrator = BuildOrchestrator(ctx)

    if not dry_run:
        check_comfyui(comfyui_url)

    orchestrator.build()


def main():
    parser = argparse.ArgumentParser(
        description="Build V2: DAG-based declarative character asset build system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("yaml_files", nargs="+", help="Character YAML manifest(s)")
    parser.add_argument("--rebuild", nargs="*", default=None,
                       help="Force rebuild specific stages (e.g. generate, sprites_animated, all)")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--clean", action="store_true", help="Delete outputs and rebuild")

    args = parser.parse_args()

    for yaml_file in args.yaml_files:
        build_character(
            yaml_path=Path(yaml_file),
            rebuild_targets=args.rebuild,
            seed_override=args.seed,
            dry_run=args.dry_run,
            clean=args.clean,
        )


if __name__ == "__main__":
    main()
