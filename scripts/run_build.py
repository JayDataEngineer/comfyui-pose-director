#!/usr/bin/env python3
"""Build system CLI entry point.

Usage:
    # Standalone character YAML (existing)
    uv run python scripts/run_build.py characters/t1_npc.yaml
    uv run python scripts/run_build.py characters/t1_npc.yaml --dry-run
    uv run python scripts/run_build.py characters/t1_npc.yaml --rebuild sheet

    # Floor YAML extraction (new)
    uv run python scripts/run_build.py --floor floors/club.yaml --character bartender
    uv run python scripts/run_build.py --floor floors/club.yaml --character bartender --dry-run

    # Other options
    uv run python scripts/run_build.py characters/t1_npc.yaml --clean
    uv run python scripts/run_build.py characters/t1_npc.yaml --comfyui-url http://100.86.69.57:18465
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_v2.context import BuildContext
from build_v2.floor_adapter import extract_character_from_floor, list_floor_characters
from build_v2.orchestrator import BuildOrchestrator
from build_v2.validators import validate_character


def main():
    parser = argparse.ArgumentParser(description="DAG-based character build system")

    # Mutually exclusive: standalone YAML or floor extraction
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("yaml_path", nargs="?", type=Path,
                       help="Standalone character YAML manifest")
    group.add_argument("--floor", type=Path,
                       help="Floor YAML to extract character from")

    parser.add_argument("--character",
                        help="Character name within floor YAML (required with --floor)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")
    parser.add_argument("--rebuild", action="append", default=[], dest="rebuild_targets",
                        help="Force rebuild specific stages (repeatable, or 'all')")
    parser.add_argument("--clean", action="store_true",
                        help="Delete output and rebuild from scratch")
    parser.add_argument("--seed", type=int, help="Override seed for this run")
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8465",
                        help="ComfyUI URL (default: http://127.0.0.1:8465)")
    args = parser.parse_args()

    if args.floor:
        floor_path = args.floor.resolve()
        if not floor_path.exists():
            print(f"ERROR: {floor_path} not found")
            sys.exit(1)

        floor_data = yaml.safe_load(floor_path.read_text())

        if not args.character:
            names = list_floor_characters(floor_data)
            parser.error(f"--character required with --floor. Available: {names}")

        data = extract_character_from_floor(floor_data, args.character)
        validate_character(data, floor_path)
        yaml_ref = str(floor_path)
    else:
        if args.yaml_path is None:
            parser.error("yaml_path is required when not using --floor")

        yaml_path = args.yaml_path.resolve()
        if not yaml_path.exists():
            print(f"ERROR: {yaml_path} not found")
            sys.exit(1)

        data = yaml.safe_load(yaml_path.read_text())
        validate_character(data, yaml_path)
        yaml_ref = str(yaml_path)

    if args.clean:
        args.rebuild_targets = ["all"]

    ctx = BuildContext.from_yaml(
        data=data,
        yaml_path=yaml_ref,
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
