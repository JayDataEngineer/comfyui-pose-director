# ComfyUI Workflow Catalog

Auto-generated from 27 workflows.

## Summary

| Category | Count |
|----------|-------|
| Animation | 1 |
| Character Editing | 10 |
| Character Generation | 4 |
| Pose & Motion | 7 |
| Post-Processing | 1 |
| Utility | 3 |
| Video | 1 |

## Animation

| Workflow | Extension | Nodes | Inputs | Outputs |
|----------|-----------|-------|--------|---------|
| CharSpriteCreatorV5 | vnccs | 3 | 0 | 1 OK |

## Character Editing

| Workflow | Extension | Nodes | Inputs | Outputs |
|----------|-----------|-------|--------|---------|
| 02 vnccs edit keyframe | pose-director | 9 | 10 | 1 OK |
| 02 vnccs edit mesh | pose-director | 11 | 10 | 2 OK |
| 03 motion keyframes qwen edit mesh | pose-director | 13 | 12 | 3 OK |
| Pose Studio | pose-director | 10 | 1 | 1 OK |
| VN Step2 ClothesChanger | vnccs | 45 | 36 | 5 OK |
| VN Step3 CharEmotionGeneratorV6 | vnccs | 21 | 9 | 1 OK |
| ClothesGenerator | vnccs | 18 | 8 | 1 OK |
| EmotionStudio V1 | vnccs | 16 | 9 | 1 OK |
| QwenDetailer ChangeEmotion | vnccs-utils | 5 | 1 | 1 OK |
| Visual camera control node for Qwen Image Edit 2511 Multiple Angles LoRa | vnccs-utils | 10 | 1 | 2 OK |

## Character Generation

| Workflow | Extension | Nodes | Inputs | Outputs |
|----------|-----------|-------|--------|---------|
| 00 z image character | pose-director | 10 | 8 | 1 OK |
| VN Step1 CharSheetGenerator | vnccs | 37 | 33 | 5 OK |
| Clone Existing Character | vnccs | 51 | 44 | 5 OK |
| CharSheetGenerator | vnccs | 32 | 31 | 4 OK |

## Pose & Motion

| Workflow | Extension | Nodes | Inputs | Outputs |
|----------|-----------|-------|--------|---------|
| 01 hymotion keyframes | pose-director | 7 | 5 | 1 OK |
| Pose Studio | vnccs-utils | 10 | 1 | 1 OK |
| gguf | hy-motion | 8 | 8 | 2 OK |
| glb | hy-motion | 7 | 8 | 1 OK |
| mixamo | hy-motion | 8 | 5 | 1 OK |
| rewrite prompt | hy-motion | 13 | 6 | 2 OK |
| workflow | hy-motion | 8 | 8 | 2 OK |

## Post-Processing

| Workflow | Extension | Nodes | Inputs | Outputs |
|----------|-----------|-------|--------|---------|
| 06 ltx edit anything postprocess | pose-director | 10 | 10 | 1 OK |

## Utility

| Workflow | Extension | Nodes | Inputs | Outputs |
|----------|-----------|-------|--------|---------|
| 07 lip sync | pose-director | 8 | 1 | 1 OK |
| LoraDataSetGeneratorV5 | vnccs | 2 | 0 | 0 SKIP |
| Model Loader ShowCase | vnccs-utils | 9 | 1 | 1 OK |

## Video

| Workflow | Extension | Nodes | Inputs | Outputs |
|----------|-----------|-------|--------|---------|
| 04 ltx video assembly | pose-director | 16 | 13 | 1 OK |
