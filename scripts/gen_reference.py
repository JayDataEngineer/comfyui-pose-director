"""Quick script to generate a reference image for a character using Z-Image."""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.comfyui_client import submit_workflow, get_output_images
from pipeline.ui_to_api import load_ui_workflow, patch_widget_value, convert_ui_to_api


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "90s OVA anime cel of a large quadruped shadow creature, glowing red eyes, wispy dark form"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 999
    output_path = sys.argv[3] if len(sys.argv) > 3 else "reference/shadow_beast_concept.png"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wf_path = os.path.join(base_dir, "workflows", "00_z_image_character.json")

    print(f"Loading workflow: {wf_path}")
    wf = load_ui_workflow(wf_path)
    wf = patch_widget_value(wf, "CLIPTextEncode", "text", f"Full body portrait, wide shot. {prompt}")
    wf = patch_widget_value(wf, "KSampler", "seed", seed)

    api_wf = convert_ui_to_api(wf)
    print("Submitting Z-Image workflow...")
    entry = submit_workflow(api_wf)
    print(f"Workflow complete, output nodes: {list(entry.get('outputs', {}).keys())}")

    images = get_output_images(entry["outputs"], 10)
    if images:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(images[0])
        print(f"Saved: {output_path} ({len(images[0])} bytes)")
    else:
        print("ERROR: No images returned")


if __name__ == "__main__":
    main()
