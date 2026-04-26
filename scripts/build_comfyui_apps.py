#!/usr/bin/env python3
"""Scan ComfyUI custom_nodes workflows and generate .app.json files.

ComfyUI's native App Builder converts workflows into clean webapps with
exposed inputs/outputs. This script automates that conversion by:
  1. Discovering workflow JSONs across installed extensions
  2. Analyzing node structures to find user-facing inputs and outputs
  3. Generating .app.json files with proper extra.linearData
  4. Optionally writing a Markdown catalog

Generated .app.json files appear in ComfyUI's "Apps" sidebar tab automatically.

Usage:
    uv run python scripts/build_comfyui_apps.py
    uv run python scripts/build_comfyui_apps.py --offline
    uv run python scripts/build_comfyui_apps.py --source vnccs
    uv run python scripts/build_comfyui_apps.py --dry-run
"""

import argparse
import json
import logging
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent.parent
COMFYUI_DIR = Path(
    os.environ.get("COMFYUI_DIR", "/home/ubuntu/Documents/programs/img/comfyui")
) if (os := __import__("os")) else Path("/home/ubuntu/Documents/programs/img/comfyui")
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://localhost:8000/comfyui") if (os := __import__("os")) else "http://localhost:8000/comfyui"

# Directories to scan for workflows (relative to ComfyUI or absolute)
SOURCES: list[tuple[str, Path]] = [
    ("pose-director", PROJECT_DIR / "workflows"),
    ("vnccs", COMFYUI_DIR / "custom_nodes" / "vnccs" / "workflows"),
    ("vnccs-utils", COMFYUI_DIR / "custom_nodes" / "vnccs-utils" / "workflows"),
    ("hy-motion", COMFYUI_DIR / "custom_nodes" / "ComfyUI-HY-Motion1" / "workflows"),
]

# Subdirectory names to skip during discovery
SKIP_DIRS = {"old_workflows", "legacy", "legacy_stick_figure", "tests"}

# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

# These node types produce final output — their node IDs become app outputs
OUTPUT_NODE_TYPES = {"SaveImage", "PreviewImage", "VHS_VideoCombine", "HYMotionExportFBX"}

# These node types are internal config — never expose their widgets
SKIP_NODE_TYPES = {
    "UNETLoader", "CLIPLoader", "VAELoader", "UnetLoaderGGUF",
    "DualCLIPLoaderGGUF", "CheckpointLoaderSimple",
    "LTXVGemmaCLIPModelLoader", "LTXICLoRALoaderModelOnly",
    "HYMotionLoadLLM", "HYMotionLoadLLMGGUF", "HYMotionLoadNetwork",
    "LoraLoaderModelOnly", "ModelSamplingAuraFlow", "ConditioningZeroOut",
    "VAEDecode", "VAEEncode", "ImpactImageBatchToImageList",
    "CLIPVisionLoader", "OpenposePreprocessor",
}

# Widgets to expose per node type: {node_type: [widget_names]}
EXPOSE_RULES: dict[str, list[str]] = {
    "LoadImage": ["image"],
    "CLIPTextEncode": ["text"],
    "KSampler": ["seed", "steps", "cfg", "denoise"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "EmptyLTXVLatentVideo": ["width", "height", "length", "batch_size"],
    "HYMotionEncodeText": ["text"],
    "HYMotionGenerate": ["duration", "seed", "cfg_scale", "num_samples"],
    "HYMotionPreview": ["sample_index", "frame_step", "image_size"],
    "MotionDirector": ["motion_description", "num_keyframes", "total_frames"],
    "PoseManipulator": ["instruction"],
    "VNCCS_QWEN_Encoder": ["instruction", "weight1", "weight2", "target_size"],
    "CharacterCreator": ["sex", "age", "race", "eyes", "hair", "seed", "new_character_name"],
    "VNCCS_Pipe": ["seed_int", "sample_steps", "cfg", "denoise"],
    "VNCCS_PoseGenerator": ["pose_data", "line_thickness", "safe_zone"],
    "PromptEnhancer": ["prompt", "variant"],
    "AnimationExporter": ["total_frames", "fps", "export_bvh", "export_video"],
    "BodyMeshRenderer": ["width", "height"],
}

# Node schemas: ordered widget names (from pipeline/ui_to_api.py)
NODE_SCHEMAS: dict[str, list[str]] = {
    "LoadImage": ["image", "upload"],
    "KSampler": ["model", "seed", "steps", "cfg", "sampler_name", "scheduler",
                  "positive", "negative", "latent_image", "denoise"],
    "CLIPTextEncode": ["text", "clip"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "EmptyLTXVLatentVideo": ["width", "height", "length", "batch_size"],
    "SaveImage": ["images", "filename_prefix"],
    "PreviewImage": ["images", "filename_prefix"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "UnetLoaderGGUF": ["unet_name"],
    "HYMotionEncodeText": ["llm", "text"],
    "HYMotionGenerate": ["network", "conditioning", "duration", "seed",
                         "cfg_scale", "num_samples"],
    "HYMotionPreview": ["motion_data", "sample_index", "frame_step", "image_size"],
    "MotionDirector": ["motion_description", "num_keyframes", "total_frames",
                       "base_url", "model", "width", "height", "max_retries",
                       "vision_feedback_rounds"],
    "PoseManipulator": ["keypoints", "instruction", "base_url", "api_key", "model",
                        "width", "height", "pose_keypoint", "base_pose"],
    "VNCCS_QWEN_Encoder": ["clip", "prompt", "vae", "latent_image_index",
                           "image1", "image2", "image3", "image1_name",
                           "image2_name", "image3_name", "target_size",
                           "upscale_method", "crop_method", "weight1", "weight2",
                           "weight3", "vl_size", "instruction", "qwen_2511"],
    "PromptEnhancer": ["prompt", "variant", "base_url", "model"],
    "AnimationExporter": ["rotations_json", "keyframes_json", "total_frames", "fps",
                         "export_bvh", "export_video", "width", "height",
                         "filename_prefix", "video_format", "backend"],
    "BodyMeshRenderer": ["rotations_json", "width", "height", "backend"],
    # VNCCS custom nodes
    "VNCCS_Pipe": ["model", "clip", "vae", "pos", "neg", "pipe", "seed_int",
                   "sample_steps", "cfg", "denoise", "sampler_name", "scheduler"],
    "VNCCS_PoseGenerator": ["pose_data", "line_thickness", "safe_zone"],
    "VNCCS_Resize": ["image", "width", "height", "method"],
    "CharacterCreator": ["existing_character", "background_color", "aesthetics", "nsfw",
                         "sex", "age", "race", "eyes", "hair", "face", "body",
                         "skin_color", "additional_details", "seed", "negative_prompt",
                         "lora_prompt", "new_character_name"],
}

# UI-only widgets that appear in widgets_values but aren't real inputs
UI_ONLY_WIDGETS: dict[str, dict[int, str]] = {
    "KSampler": {1: "control_after_generate"},
    "HYMotionGenerate": {2: "control_after_generate"},
}

# ---------------------------------------------------------------------------
# Category assignment
# ---------------------------------------------------------------------------

CATEGORY_RULES: list[tuple[str, str]] = [
    # VNCCS steps
    (r"vnccs.*Step1", "Character Generation"),
    (r"vnccs.*Step2", "Character Editing"),
    (r"vnccs.*Step3", "Character Editing"),
    (r"vnccs.*Step4", "Animation"),
    (r"vnccs.*Step5", "Utility"),
    (r"vnccs.*Clone", "Character Generation"),
    # VNCCS-Utils
    (r"vnccs-utils.*Pose", "Pose & Motion"),
    (r"vnccs-utils.*camera", "Character Editing"),
    (r"vnccs-utils.*Model", "Utility"),
    (r"vnccs-utils.*Detail", "Character Editing"),
    # HY-Motion
    (r"hy-motion", "Pose & Motion"),
    # Pose Director pipeline stages
    (r"pose-director.*/00_", "Character Generation"),
    (r"pose-director.*/01_", "Pose & Motion"),
    (r"pose-director.*/02_", "Character Editing"),
    (r"pose-director.*/03_", "Character Editing"),
    (r"pose-director.*/04_", "Video"),
    (r"pose-director.*/06_", "Post-Processing"),
    (r"pose-director.*/07_", "Utility"),
    # Catch-all
    (r"vnccs", "Character Editing"),
    (r"pose-director", "Utility"),
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WorkflowApp:
    """Analysis result for one workflow."""
    source_path: Path
    extension: str
    category: str
    display_name: str
    inputs: list[tuple[str, str]] = field(default_factory=list)  # [(node_id, widget_name)]
    outputs: list[str] = field(default_factory=list)  # [node_id]
    node_count: int = 0
    node_types: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class BuildReport:
    total_scanned: int = 0
    apps_generated: int = 0
    apps_skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_workflows(sources: list[tuple[str, Path]] | None = None) -> list[tuple[Path, str]]:
    """Find all workflow JSONs across configured sources.

    Returns [(path, extension_name), ...] sorted by extension then filename.
    """
    if sources is None:
        sources = SOURCES

    results: list[tuple[Path, str]] = []

    for ext_name, base_dir in sources:
        if not base_dir.is_dir():
            logging.warning("Source directory not found: %s", base_dir)
            continue

        for json_file in sorted(base_dir.rglob("*.json")):
            # Skip unwanted subdirs
            parts = json_file.relative_to(base_dir).parts
            if any(p in SKIP_DIRS for p in parts):
                continue

            # Skip API-format files (they have a "Prompt" or top-level node IDs)
            # We detect these later during load; for now just collect paths
            results.append((json_file, ext_name))

    return results


# ---------------------------------------------------------------------------
# Widget name resolution
# ---------------------------------------------------------------------------


def get_widget_names_for_node(
    node: dict,
    object_info: dict[str, dict] | None = None,
) -> list[str]:
    """Get ordered widget names for a node.

    Priority:
      1. NODE_SCHEMAS (known types)
      2. object_info from ComfyUI API
      3. Heuristic from node's inputs array
    """
    node_type = _get_node_type(node)

    # 1. Known schemas
    if node_type in NODE_SCHEMAS:
        schema = NODE_SCHEMAS[node_type]
        # Filter to only widget-type params (skip connection params)
        ui_inputs = {inp["name"] for inp in node.get("inputs", [])}
        return [name for name in schema if name not in ui_inputs or _is_widget_input(node, name)]

    # 2. object_info from API
    if object_info and node_type in object_info:
        return _widget_names_from_object_info(node_type, object_info)

    # 3. Heuristic: inputs with link=None are widgets
    return _widget_names_from_inputs(node)


def _get_node_type(node: dict) -> str:
    """Get the real node type, handling UUID-type nodes."""
    t = node.get("type", "")
    # Some VNCCS nodes use UUIDs as type; check properties for real name
    if re.match(r"^[0-9a-f]{8}-", t):
        return node.get("properties", {}).get("Node name for S&R", t)
    return t


def _is_widget_input(node: dict, name: str) -> bool:
    """Check if a named input is a widget (has no connection link)."""
    for inp in node.get("inputs", []):
        if inp.get("name") == name:
            return inp.get("link") is None
    return True  # Not in inputs array = hidden widget


def _widget_names_from_object_info(node_type: str, object_info: dict) -> list[str]:
    """Extract widget names from ComfyUI's /object_info API response."""
    info = object_info.get(node_type, {})
    result = []
    for section in ("required", "optional"):
        for name, type_info in info.get("input", {}).get(section, {}).items():
            if isinstance(type_info, (list, tuple)):
                # COMBO type like ["euler", "dpmpp_2m", ...] or [STRING, {"default": ""}]
                result.append(name)
            elif isinstance(type_info, str) and type_info in {"STRING", "INT", "FLOAT", "BOOLEAN"}:
                result.append(name)
    return result


def _widget_names_from_inputs(node: dict) -> list[str]:
    """Fallback: extract widget names from node's own inputs array."""
    result = []
    for inp in node.get("inputs", []):
        if inp.get("link") is None:
            name = inp.get("name", "")
            typ = inp.get("type", "")
            if typ in {"STRING", "INT", "FLOAT", "BOOLEAN"} or isinstance(typ, list):
                result.append(name)
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def load_workflow(path: Path) -> dict | None:
    """Load and validate a workflow JSON. Returns None if not a valid UI workflow."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logging.warning("Failed to parse %s: %s", path.name, e)
        return None

    # Detect API format (has numeric string keys at top level)
    if not isinstance(data, dict):
        return None
    if "nodes" not in data:
        return None  # API format or not a workflow

    return data


def analyze_workflow(
    workflow: dict,
    source_path: Path,
    extension: str,
    object_info: dict[str, dict] | None = None,
) -> WorkflowApp:
    """Analyze a workflow and determine its app inputs/outputs."""
    app = WorkflowApp(
        source_path=source_path,
        extension=extension,
        category=determine_category(source_path, extension),
        display_name=extract_display_name(source_path, extension),
    )

    nodes = workflow.get("nodes", [])
    if not nodes:
        app.skipped = True
        app.skip_reason = "No nodes found"
        return app

    app.node_count = len(nodes)
    seen_types: set[str] = set()

    for node in nodes:
        node_type = _get_node_type(node)
        seen_types.add(node_type)
        node_id = str(node.get("id", ""))

        # Check for output nodes
        if node_type in OUTPUT_NODE_TYPES:
            app.outputs.append(node_id)
            continue

        # Skip internal config nodes
        if node_type in SKIP_NODE_TYPES:
            continue

        # Check expose rules
        if node_type in EXPOSE_RULES:
            widget_names = get_widget_names_for_node(node, object_info)
            ui_only = UI_ONLY_WIDGETS.get(node_type, {})
            # Map widget names to widgets_values positions
            widget_idx = 0
            exposed_names = EXPOSE_RULES[node_type]
            name_to_idx: dict[str, int] = {}
            for name in widget_names:
                while widget_idx in ui_only:
                    widget_idx += 1
                name_to_idx[name] = widget_idx
                widget_idx += 1

            for expose_name in exposed_names:
                if expose_name in name_to_idx:
                    app.inputs.append((node_id, expose_name))
                elif expose_name in widget_names:
                    # Widget exists but not in positional map — use name directly
                    app.inputs.append((node_id, expose_name))

    app.node_types = sorted(seen_types)

    if not app.outputs:
        app.skipped = True
        app.skip_reason = "No output nodes (SaveImage/PreviewImage) found"

    return app


def determine_category(source_path: Path, extension: str) -> str:
    """Assign a category based on source path patterns."""
    path_str = f"{extension}/{source_path.name}"
    for pattern, category in CATEGORY_RULES:
        if re.search(pattern, path_str, re.IGNORECASE):
            return category
    return "Utility"


def extract_display_name(source_path: Path, extension: str) -> str:
    """Generate a human-readable display name from the filename."""
    stem = source_path.stem
    # Remove common prefixes
    for prefix in ("VN_Step1_QWEN_", "VN_Step2_QWEN_", "VN_Step3_QWEN_",
                   "VN_Step4_", "VN_Step5_", "VN_Step1.1_QWEN_",
                   "VNCCS_Utils ", "workflow"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    # Remove trailing version tags
    stem = re.sub(r"[_-]v?\d+(\.\d+)*$", "", stem)
    # Replace underscores/hyphens with spaces
    stem = stem.replace("_", " ").replace("-", " ")
    # Title case
    stem = stem.strip()
    if not stem or stem == extension:
        stem = source_path.stem
    return stem.strip()


# ---------------------------------------------------------------------------
# .app.json generation
# ---------------------------------------------------------------------------


def generate_app_json(workflow: dict, app: WorkflowApp) -> dict:
    """Generate a .app.json by adding extra.linearData to the workflow."""
    result = deepcopy(workflow)

    # Preserve existing extra fields
    extra = result.get("extra", {})
    extra.update({
        "linearMode": True,
        "linearData": {
            "inputs": [[nid, wname] for nid, wname in app.inputs],
            "outputs": app.outputs,
        },
        "ds": extra.get("ds", {"scale": 0.6, "offset": [0, 0]}),
    })
    result["extra"] = extra

    return result


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text)
    return text[:60].rstrip("_")


# ---------------------------------------------------------------------------
# Catalog generation
# ---------------------------------------------------------------------------


def generate_catalog(apps: list[WorkflowApp]) -> str:
    """Generate Markdown catalog from all analyzed workflows."""
    lines: list[str] = [
        "# ComfyUI Workflow Catalog",
        "",
        f"Auto-generated from {len(apps)} workflows.",
        "",
    ]

    # Group by category
    by_category: dict[str, list[WorkflowApp]] = {}
    for app in apps:
        cat = app.category
        by_category.setdefault(cat, []).append(app)

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    for cat in sorted(by_category):
        count = len(by_category[cat])
        lines.append(f"| {cat} | {count} |")
    lines.append("")

    # Per-category details
    for cat in sorted(by_category):
        lines.append(f"## {cat}")
        lines.append("")
        lines.append("| Workflow | Extension | Nodes | Inputs | Outputs |")
        lines.append("|----------|-----------|-------|--------|---------|")
        for app in by_category[cat]:
            status = "SKIP" if app.skipped else "OK"
            lines.append(
                f"| {app.display_name} | {app.extension} | "
                f"{app.node_count} | {len(app.inputs)} | "
                f"{len(app.outputs)} {status} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ComfyUI API
# ---------------------------------------------------------------------------


def fetch_object_info() -> dict[str, dict] | None:
    """Fetch /object_info from ComfyUI. Returns None if unreachable."""
    if not HAS_REQUESTS:
        return None
    try:
        resp = requests.get(f"{COMFYUI_URL}/object_info", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.info("ComfyUI API unavailable (%s), using heuristics", e)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_apps(
    sources: list[tuple[str, Path]] | None = None,
    output_dir: Path | None = None,
    offline: bool = False,
    dry_run: bool = False,
    catalog_path: Path | None = None,
) -> BuildReport:
    """Main build pipeline."""
    report = BuildReport()

    # Fetch object_info for better widget resolution
    object_info = None if offline else fetch_object_info()

    # Discover workflows
    discovered = discover_workflows(sources)
    report.total_scanned = len(discovered)
    logging.info("Discovered %d workflows", len(discovered))

    apps: list[WorkflowApp] = []

    for path, ext in discovered:
        workflow = load_workflow(path)
        if workflow is None:
            report.errors.append(f"Not a valid workflow: {path.name}")
            continue

        app = analyze_workflow(workflow, path, ext, object_info)
        apps.append(app)

        if app.skipped:
            report.apps_skipped += 1
            logging.info("SKIP %-40s [%s] — %s", path.name, ext, app.skip_reason)
            continue

        report.apps_generated += 1
        logging.info("OK   %-40s [%s] → %d inputs, %d outputs",
                     path.name, ext, len(app.inputs), len(app.outputs))

        if dry_run:
            continue

        # Generate .app.json
        app_json = generate_app_json(workflow, app)
        if output_dir is None:
            output_dir = COMFYUI_DIR / "Workflows" / "apps"

        cat_dir = output_dir / slugify(app.category)
        cat_dir.mkdir(parents=True, exist_ok=True)

        out_path = cat_dir / f"{slugify(app.display_name)}.app.json"
        out_path.write_text(json.dumps(app_json, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("  → %s", out_path)

    # Generate catalog
    catalog_content = generate_catalog(apps)
    if catalog_path is None:
        catalog_path = PROJECT_DIR / "docs" / "workflow_catalog.md"

    if not dry_run:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(catalog_content, encoding="utf-8")
        logging.info("Catalog written to %s", catalog_path)
    else:
        logging.info("Catalog preview:\n%s", catalog_content)

    return report


def main():
    parser = argparse.ArgumentParser(description="Build ComfyUI .app.json files from extension workflows")
    parser.add_argument("--comfyui-dir", type=Path, default=COMFYUI_DIR,
                        help="ComfyUI installation directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for .app.json files (default: {comfyui-dir}/Workflows/apps/)")
    parser.add_argument("--offline", action="store_true",
                        help="Skip ComfyUI /object_info fetch (use heuristics only)")
    parser.add_argument("--source", type=str, default=None,
                        choices=["pose-director", "vnccs", "vnccs-utils", "hy-motion", "all"],
                        help="Only process one extension source")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be generated without writing files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed analysis per workflow")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Filter sources
    if args.source and args.source != "all":
        sources = [(name, path) for name, path in SOURCES if name == args.source]
    else:
        sources = SOURCES

    report = build_apps(
        sources=sources,
        output_dir=args.output_dir,
        offline=args.offline,
        dry_run=args.dry_run,
    )

    print(f"\nDone: {report.apps_generated} apps generated, "
          f"{report.apps_skipped} skipped, "
          f"{report.total_scanned} scanned")
    if report.errors:
        print(f"Errors: {len(report.errors)}")
        for err in report.errors:
            print(f"  - {err}")

    return 0 if not report.errors else 1


if __name__ == "__main__":
    sys.exit(main())
