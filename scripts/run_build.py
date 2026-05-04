#!/usr/bin/env python3
"""Build system CLI entry point.

Usage:
    uv run python scripts/build_v2.py characters/t1_npc.yaml
    uv run python scripts/build_v2.py characters/t1_npc.yaml --dry-run
    uv run python scripts/build_v2.py characters/t1_npc.yaml --rebuild sheet
    uv run python scripts/build_v2.py characters/t1_npc.yaml --clean
    uv run python scripts/build_v2.py characters/t1_npc.yaml --comfyui-url http://100.86.69.57:18465
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_v2.context import BuildContext
from build_v2.orchestrator import BuildOrchestrator
from build_v2.validators import validate_character


def main():
    parser = argparse.ArgumentParser(description="DAG-based character build system")
    parser.add_argument("yaml_path", type=Path, help="Character YAML manifest")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--rebuild", action="append", default=[], dest="rebuild_targets",
                        help="Force rebuild specific stages (repeatable, or 'all')")
    parser.add_argument("--clean", action="store_true", help="Delete output and rebuild from scratch")
    parser.add_argument("--seed", type=int, help="Override seed for this run")
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8465",
                        help="ComfyUI URL (default: http://127.0.0.1:8465)")
    args = parser.parse_args()

    yaml_path = args.yaml_path.resolve()
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found")
        sys.exit(1)

    data = yaml.safe_load(yaml_path.read_text())
    validate_character(data, yaml_path)

    if args.clean:
        args.rebuild_targets = ["all"]

    ctx = BuildContext.from_yaml(
        data=data,
        yaml_path=str(yaml_path),
        project_root=Path(__file__).parent.parent,
        comfyui_url=args.comfyui_url,
        seed_override=args.seed,
        dry_run=args.dry_run,
        rebuild_targets=args.rebuild_targets,
    )

    orch = BuildOrchestrator(ctx)
    orch.build()


if __name__ == "__main__":
    main()
