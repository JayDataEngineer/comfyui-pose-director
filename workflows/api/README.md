# Pre-Exported API-Format Workflows

VNCCS workflows contain UUID subgraph nodes that our `ui_to_api.py` converter
cannot expand. Instead, we pre-export them as API-format JSONs from the
ComfyUI browser.

## How to Export

1. Open ComfyUI browser (local: `http://127.0.0.1:8465`, remote: `http://100.86.69.57:18465`)
2. Load the VNCCS workflow via the menu (e.g. `VN_Step1.1_QWEN_Clone_Existing_Character_v1.json`)
3. Click the menu button → **Save (API Format)**
4. Save the file here with the naming convention below

## Naming Convention

```
vnccs_step1_char_sheet.json      # Step 1: CharacterCreator (prompt → sheet)
vnccs_step11_clone_existing.json # Step 1.1: Clone existing character (image → sheet)
vnccs_step2_clothes.json         # Step 2: Clothes generator (sheet + outfit → new sheet)
vnccs_step3_emotions.json        # Step 3: Emotion studio (sheet → face emotion variants)
vnccs_step4_sprites.json         # Step 4: Sprite creator (sheet + poses → sprite frames)
vnccs_step5_lora_dataset.json    # Step 5: LoRA training dataset generator
```

## Patching

API-format workflows are trivial to patch — just load the JSON and change
`inputs` values by node ID:

```python
import json
with open("workflows/api/vnccs_step11_clone_existing.json") as f:
    api = json.load(f)

# Change seed on node 123
api["123"]["inputs"]["seed"] = 42

# Change prompt on node 456
api["456"]["inputs"]["prompt"] = "anime character..."

# Submit to ComfyUI
requests.post("http://127.0.0.1:8465/prompt", json={"prompt": api})
```

See `build_v2/api_patcher.py` for the programmatic patcher used by build stages.

## Source Workflows

The original UI-format VNCCS workflows are in the parent directory:
- `workflows/VN_Step1_QWEN_CharSheetGenerator_v1.json`
- `workflows/VN_Step1.1_QWEN_Clone_Existing_Character_v1.json`
- `workflows/VN_Step2_QWEN_ClothesGenerator_v1.json`
- `workflows/VN_Step3_QWEN_EmotionStudio_V1.json`
- `workflows/VN_Step4_CharSpriteCreatorV5.json`
- `workflows/VN_Step5_LoraDataSetGeneratorV5.json`
