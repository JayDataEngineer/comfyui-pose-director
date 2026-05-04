"""Export VNCCS UI-format workflows to API format.

Converts workflows/VN_Step*.json → workflows/api/vnccs_stepN.json
using subgraph expansion + ui_to_api conversion.

Usage:
    uv run python scripts/export_api_workflows.py
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.ui_to_api import load_ui_workflow, convert_ui_to_api

# Mapping from UI filename → API filename
WORKFLOW_MAP = {
    # VNCCS main (Step 1-5)
    "VN_Step1_QWEN_CharSheetGenerator_v1.json": "vnccs_step1_char_sheet.json",
    "VN_Step1.1_QWEN_Clone_Existing_Character_v1.json": "vnccs_step11_clone_existing.json",
    "VN_Step2_QWEN_ClothesGenerator_v1.json": "vnccs_step2_clothes.json",
    "VN_Step3_QWEN_EmotionStudio_V1.json": "vnccs_step3_emotions.json",
    "VN_Step4_CharSpriteCreatorV5.json": "vnccs_step4_sprites.json",
    "VN_Step5_LoraDataSetGeneratorV5.json": "vnccs_step5_lora_dataset.json",
    # VNCCS-Utils (Pose Studio — we replace manual poses with Anny/HY-Motion)
    "VNCCS_Utils Pose Studio.json": "vnccs_utils_pose_studio.json",
}

workflows_dir = project_root / "workflows"
api_dir = workflows_dir / "api"
api_dir.mkdir(exist_ok=True)


def main():
    for ui_name, api_name in WORKFLOW_MAP.items():
        ui_path = workflows_dir / ui_name
        api_path = api_dir / api_name

        if not ui_path.exists():
            print(f"SKIP {ui_name} (not found)")
            continue

        print(f"Converting {ui_name} → api/{api_name}...", end=" ", flush=True)

        try:
            ui = load_ui_workflow(str(ui_path))
            api = convert_ui_to_api(ui)
            node_count = len(api)
            with open(api_path, "w") as f:
                json.dump(api, f, indent=2)
            print(f"OK ({node_count} nodes)")
        except Exception as e:
            print(f"FAILED: {e}")
            if api_path.exists():
                api_path.unlink()

    print("\nDone.")


if __name__ == "__main__":
    main()
