# VNCCS Capability Reference

Complete inventory of everything VNCCS and VNCCS-Utils provides. This is the platform we build on — use these tools before building our own.

**Source locations:**
- VNCCS: `/home/ubuntu/Documents/programs/img/comfyui/custom_nodes/vnccs/`
- VNCCS-Utils: `/home/ubuntu/Documents/programs/img/comfyui/custom_nodes/vnccs-utils/`

---

## Node Inventory (22 total)

### VNCCS Extension (15 nodes)

| # | Node Class | Display Name | Category | Purpose |
|---|-----------|-------------|----------|---------|
| 1 | `CharacterCreator` | VNCCS Character Creator | VNCCS | Define character attributes, generate prompts, create directory structure |
| 2 | `CharacterPreview` | VNCCS Character Preview | VNCCS | Quick character creation without file management |
| 3 | `CharacterAssetSelector` | VNCCS Asset Selector | VNCCS | Select existing character + costume + accessories |
| 4 | `EmotionGenerator` | VNCCS Emotion Generator | VNCCS | **Batch emotion generation** — 100+ emotions across 6 groups |
| 5 | `EmotionGeneratorV2` | VNCCS Emotion Generator V2 | VNCCS | Enhanced version with web API, JSON config, costume/costume selection |
| 6 | `VNCCSSheetManager` | VNCCS Sheet Manager | VNCCS | Split/compose 2x6 sprite grids |
| 7 | `CharacterSheetCropper` | VNCCS Character Sheet Cropper | VNCCS/Util | Contour-based auto-crop from sheets |
| 8 | `SpriteGenerator` | VNCCS Sprite Generator | VNCCS | Batch sprite output from character sheets (12-frame sequences) |
| 9 | `VNCCS_QWEN_Encoder` | VNCCS QWEN Encoder | VNCCS/encoding | **Core node** — 3-image reference latents, timestep-zero injection |
| 10 | `VNCCS_Pipe` | VNCCS Pipe | VNCCS | Aggregation pipe — chains model/clip/vae/seed/steps/cfg through workflow |
| 11 | `VNCCSSamplerSchedulerPicker` | VNCCS Sampler Scheduler Picker | VNCCS | Dynamic sampler/scheduler enumeration |
| 12 | `DatasetGenerator` | VNCCS Dataset Generator | VNCCS | **Build captioned training datasets** for LoRA fine-tuning |
| 13 | `PoseGenerator` | VNCCS Pose Generator | VNCCS | Generate OpenPose images (512x1536) with visual editor |
| 14 | `VNCCS_RMBG2` | VNCCS RMBG2 | VNCCS | **Background removal** — BEN2/BEN/RMBG-2.0/InspyreNet models |
| 15 | Common nodes (5) | Integer/Float/String/MultilineText/PromptConcat | VNCCS | Passthrough utilities |

### VNCCS-Utils Extension (7 nodes)

| # | Node Class | Display Name | Category | Purpose |
|---|-----------|-------------|----------|---------|
| 16 | `VNCCS_PositionControl` | VNCCS Position Control | VNCCS | Generate camera prompts from azimuth/elevation angles |
| 17 | `VNCCS_VisualPositionControl` | VNCCS Visual Camera Control | VNCCS | Same as PositionControl but accepts JSON from browser |
| 18 | `VNCCS_QWEN_Detailer` | VNCCS QWEN Detailer | VNCCS | AI face/region enhancement with bbox detection + inpainting |
| 19 | `VNCCS_BBox_Extractor` | VNCCS BBox Extractor | VNCCS | Extract bounding boxes from detected objects |
| 20 | `VNCCS_ModelManager` | VNCCS Model Manager | VNCCS/manager | Download/manage VNCCS models from HuggingFace |
| 21 | `VNCCS_ModelSelector` | VNCCS Model Selector | VNCCS/manager | Select specific model versions |
| 22 | `VNCCS_PoseStudio` | VNCCS Pose Studio | VNCCS/pose | Browser-based 3D pose editor (MakeHuman + Three.js) |

---

## Detailed Node Reference

### 1. CharacterCreator

**What it does:** Defines a new character with physical attributes. Generates positive/negative prompts. Creates directory structure: `{character}/Sheets/`, `{character}/Faces/`, `{character}/Sprites/` with subdirectories per costume and emotion.

**Inputs:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| existing_character | COMBO | first found | Select or create character name |
| background_color | STRING | "green" | Solid background for training consistency |
| aesthetics | STRING | "masterpiece,best quality,amazing quality" | Quality tags |
| nsfw | BOOLEAN | false | Enable nude base model |
| sex | COMBO | "female" | Character sex |
| age | INT | 18 | Character age (drives age LoRA strength) |
| race | STRING | "human" | Species/race |
| eyes | STRING | "blue eyes" | Eye description |
| hair | STRING | "black long" | Hair description |
| face | STRING | "freckles" | Face features |
| body | STRING | "medium breasts" | Body type |
| skin_color | STRING | "white" | Skin tone |
| additional_details | STRING | "" | Extra tags |
| seed | INT | 0 | Reproducibility |
| negative_prompt | STRING | "bad quality,worst quality" | What to avoid |
| lora_prompt | STRING | "" | Additional LoRA trigger tokens |
| new_character_name | STRING | "" | Rename character |

**Outputs:** `positive_prompt`, `seed`, `negative_prompt`, `age_lora_strength`, `sheets_path`, `faces_path`, `face_details`

**Character directory structure:**
```
output/VN_CharacterCreatorSuit/{character_name}/
├── {character_name}_config.json
├── Sheets/
│   └── {costume}/
│       └── {emotion}/
│           └── sheet_{emotion}_*.png
├── Faces/
│   └── {costume}/
│       └── {emotion}/
│           └── face_{emotion}_*.png
├── Sprites/
│   └── {costume}/
│       └── {emotion}/
│           └── sprite_{emotion}_*.png
└── lora/              ← created by DatasetGenerator
    └── {captioned images + .txt files}
```

---

### 4. EmotionGenerator (HIGH VALUE — WE ARE NOT USING THIS)

**What it does:** Takes a character and a comma-separated list of emotions. For each costume the character has, generates prompt text for each emotion. Returns images, emotion prompts, output paths, and masks ready for batch processing.

**100+ built-in emotions across 6 groups:**

| Group | Examples |
|-------|---------|
| **Emotions** (60+) | angry, sad, happy, confused, determined, disgusted, drunk, excited, exhausted, flustered, frustrated, lonely, nervous, pout, scared, serious, surprised, thinking |
| **Sexual** (8) | afterglow, aroused, seductive smile |
| **Smile** (13) | grin, evil grin, light smile, sad smile, forced smile |
| **Smug** (4) | doyagao, smirk, smug, troll face |
| **Surprised/Scared/Sad** (10) | horrified, screaming, sobbing, traumatized, despair |
| **Emotes** (50+) | ;), :d, xd, uwu, owo, :3, heart-shaped eyes, diamond mouth |

**Inputs:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| character | COMBO | first found | Select from existing characters |
| emotions | STRING | "angry,shy,smile,pout,sad,neutral" | Comma-separated emotion list |
| emotion_selector | COMBO | "happy" | Single emotion picker (full list) |

**Outputs (OUTPUT_IS_LIST = True):**
- `image` (batch) — character neutral sheet images (one per costume×emotion combo)
- `emotion` (list) — emotion prompt strings with character details + costume + emotion tags
- `face_output_path` (list, 12 per emotion) — save paths for face crops
- `sheet_output_path` (list) — save paths for full sheets
- `positive_prompt` — character's positive prompt
- `negative_prompt` — character's negative prompt + water drop exclusions
- `seed` — generation seed
- `mask` (list) — character masks

**Critical behavior:** It loads the character's neutral sheet for each costume, prepends the emotion tag and character description, and outputs everything needed for a batch VNCCS_Pipe → KSampler generation pass. The downstream workflow handles actual image generation — EmotionGenerator just prepares the data.

---

### 5. EmotionGeneratorV2

Enhanced version with:
- Web API endpoints (`/vnccs/get_emotions`, `/vnccs/get_character_costumes`, `/vnccs/get_character_sheet_preview`)
- JSON config file for emotions (`emotions-config/emotions.json`)
- Costume-aware generation
- Preview capability in browser

---

### 6. VNCCSSheetManager

**What it does:** Two modes:

**Split mode:** Divides a sheet image into 12 parts (2 rows × 6 columns). Each part is resized to `target_width × target_height`. Used to extract individual frames from a character sheet.

**Compose mode:** Arranges up to 12 images into a 2×6 grid. Creates `target_height × target_height` square sheet.

**Inputs:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| mode | COMBO | "split" | "split" or "compose" |
| images | IMAGE | — | Input image(s) |
| target_width | INT | 1024 | Width per split part |
| target_height | INT | 3072 | Height per split part / total compose height |
| safe_margin | BOOLEAN | false | Padding during composition |

**Also includes:** Built-in RMBG background removal (BEN2, RMBG-2.0, InspyreNet models) and the full VNCCS_RMBG2 node.

---

### 7. CharacterSheetCropper

**What it does:** Uses OpenCV contour detection on masks to auto-crop individual characters from sheets. Returns cropped images and masks.

**Inputs:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| image | IMAGE | — | Sheet image |
| mask | MASK | — | Binary mask |
| min_size | INT | 64 | Minimum crop dimension (filters noise) |

**Outputs:** `cropped_images` (list), `cropped_masks` (list) — each contour becomes a separate output.

---

### 8. SpriteGenerator

**What it does:** Scans a character's Sheets directory (all costumes, all emotions). Loads each sheet, outputs images + masks + file paths for batch saving. Generates 12 save paths per sheet (expects downstream to save 12 frames).

**Inputs:**
| Param | Type | Default |
|-------|------|---------|
| character | COMBO | first found |

**Outputs (all lists):** `images`, `file_paths` (12 per sheet), `masks`

**Behavior:** Iterates `{character}/Sheets/{costume}/{emotion}/`, loads sheets, outputs with save paths to `{character}/Sprites/{costume}/{emotion}/sprite_{emotion}_`.

---

### 9. VNCCS_QWEN_Encoder (CORE NODE)

**What it does:** The heart of VNCCS. Takes up to 3 images, VAE-encodes them as reference latents, injects at timestep zero with quadratic weighting. NOT ControlNet — this is latent injection.

**Inputs:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| clip | CLIP | required | Text encoder |
| prompt | STRING | required | Text prompt |
| vae | VAE | required | VAE decoder |
| image1 | IMAGE | optional | Reference image 1 |
| image2 | IMAGE | optional | Reference image 2 |
| image3 | IMAGE | optional | Reference image 3 |
| image1_name | STRING | "Picture 1" | Name for prompt reference |
| image2_name | STRING | "Picture 2" | Name for prompt reference |
| image3_name | STRING | "Picture 3" | Name for prompt reference |
| target_size | COMBO | 1024 | Output resolution (512-2048) |
| weight1 | FLOAT | 1.0 | Image 1 influence (0.0-2.0, quadratic) |
| weight2 | FLOAT | 1.0 | Image 2 influence (0.0-2.0, quadratic) |
| weight3 | FLOAT | 1.0 | Image 3 influence (0.0-2.0, quadratic) |
| vl_size | INT | 384 | Vision-language token resolution |
| instruction | STRING | (long default) | System instruction for QWEN |
| qwen_2511 | BOOLEAN | true | Use Qwen Image Edit 2511 mode |
| latent_image_index | INT | 1 | Which image provides latent shape |

**Outputs:** `positive` (CONDITIONING), `negative` (CONDITIONING), `latent` (LATENT)

**Critical parameters we MUST get right:**
- Image ordering: image1=mesh/pose guide, image2=character reference, image3=openpose
- Weights: all 1.0 default, can tune per image
- `qwen_2511=true` — REQUIRED for our model
- `instruction` — tells the model how to combine images. Default works, but "Draw character from image2" style instructions are proven
- Target size 1024 for sprite frames

---

### 12. DatasetGenerator (HIGH VALUE — FOR LORA TRAINING)

**What it does:** Scans character's Faces/ and Sprites/ directories. Copies all images to a `lora/` subdirectory with auto-generated caption `.txt` files. Ready for LoRA training tools (kohya, one-trainer, etc.).

**Inputs:**
| Param | Type | Default |
|-------|------|---------|
| character | COMBO | first found |
| game_name | STRING | "VN" |
| additional_caption | STRING | "" |

**Output:** `dataset_path` (STRING) — path to `{character}/lora/` directory

**Caption format example:**
```
TechNoir_Marty, 1boy, 18yo, portrait, human, black short hair, blue eyes, freckles, sweatdrop, happy, wear casual suit, simple background, green background, solid background
```

**For sprites**, caption includes: `multiple views, same character in different poses, 2boys/2girls`

**This is our LoRA training pipeline:**
1. CharacterCreator → define character
2. EmotionGenerator → generate emotion sheets
3. SpriteGenerator → extract sprites
4. **DatasetGenerator → build captioned training set**
5. Train LoRA with external tool (kohya_ss, etc.)
6. Use LoRA in generation for intrinsic character identity

---

### 14. VNCCS_RMBG2 (WE ARE USING THIS)

**What it does:** Background removal with 4 model options. Outputs transparent image + mask.

**Inputs:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| image | IMAGE | required | Input |
| model | COMBO | — | BEN2, BEN, RMBG-2.0, InspyreNet |
| sensitivity | FLOAT | 1.0 | Mask detection strength (0-1) |
| process_res | INT | 1024 | Processing resolution |
| mask_blur | INT | 0 | Edge blur radius |
| mask_offset | INT | 0 | Expand/shrink mask |
| invert_output | BOOLEAN | false | Invert mask |
| refine_foreground | BOOLEAN | false | Fast foreground color estimation |
| background | COMBO | "Alpha" | Alpha/Green/Blue |

**Outputs:** `IMAGE` (with alpha), `MASK`, `MASK_IMAGE`

**Best model:** BEN2 — highest quality. RMBG-2.0 is faster but less accurate.

---

### 16-17. PositionControl / VisualPositionControl

**What it does:** Generates camera position prompts from azimuth/elevation angles. These prompts go into the VNCCS_QWEN_Encoder to control viewing angle.

**Inputs:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| azimuth | FLOAT | 0 | 0°=front, 90°=right, 180°=back, 270°=left |
| elevation | FLOAT | 0 | -30° to 60° |
| distance | STRING | — | close-up/medium/wide |
| include_trigger | BOOLEAN | — | Prepend `<sks>` trigger token |

**Output:** STRING prompt like `<sks> front view eye-level shot medium shot`

**This replaces our manual model_rotation_y approach** when working inside VNCCS. The VNCCS model was trained with these position prompts.

---

### 18. VNCCS_QWEN_Detailer

**What it does:** Detects objects/faces in an image using bbox detector, then enhances each region using QWEN image generation with inpainting. Optional ControlNet guidance.

**Use case:** Post-processing fix for specific regions (face artifacts, hand distortions) without re-generating the entire image.

---

### 22. VNCCS_PoseStudio

**What it does:** Browser-based 3D character pose editor using MakeHuman mesh + Three.js rendering. NOT for CLI use.

**Pose Library:** 16 preset poses at `custom_nodes/vnccs/presets/poses/`:
- Static poses: 10, 11, 12, 13, 3, 5, 6, 8, 9
- Action poses: exercises1, exercises2, FckYou, Hi, Seeking, "stay on hand", "V Selfie"
- Wellness: Meditation, Youga
- Full set: vnccs_poseset

**Bone format:** MakeHuman 44-bone skeleton with XYZ Euler rotations (degrees). Compatible with our Anny→VNCCS bone mapping.

---

## Pose Library (VNCCS-Utils)

Additional pose presets at `custom_nodes/vnccs-utils/PoseLibrary/`:
- Same format as VNCCS presets
- JSON files with bone rotations + model rotation + camera position
- Preview PNGs for visual reference

---

## What We're Currently Using vs What We Should Be Using

### Currently Using
- `VNCCS_QWEN_Encoder` — core encoding (correctly)
- `VNCCS_RMBG2` — background removal (just added)
- `CharacterCreator` — indirectly through our Z-Image workflow

### Should Be Using (NOT YET)
| Node | Use Case | Priority |
|------|---------|----------|
| **EmotionGenerator** | Dialogue sprites: `marty_sad.png`, `marty_happy.png` | **CRITICAL** |
| **SpriteGenerator** | 12-frame animation sequences from sheets | **HIGH** |
| **VNCCSSheetManager** | Sprite sheet composition/splitting | **HIGH** |
| **CharacterSheetCropper** | Auto-crop individuals from sheets | **HIGH** |
| **DatasetGenerator** | LoRA training data preparation | **MEDIUM** |
| **VNCCS_QWEN_Detailer** | Post-process face/hand fixes | **MEDIUM** |
| **CharacterAssetSelector** | Costume/accessory switching | **LOW** |
| **PositionControl** | Camera angle control via prompts | **LOW** |
| **PoseGenerator** | OpenPose generation (we use OpenposePreprocessor instead) | **LOW** |

### Not VNCCS But Should Be Using
| Tool | Use Case |
|------|---------|
| **HY-Motion CLI** | Trained motion generation (replaces manual pose JSON) |
| **TRELLIS-2** | 3D character model generation from image |

---

## Complete Pipeline (What We Should Build)

```
Stage 0: CHARACTER DEFINITION
   CharacterCreator → define attributes
   → VNCCS_Pipe → KSampler → character sheet (neutral expression, green bg)

Stage 1: CHARACTER SHEET GENERATION
   EmotionGenerator → batch emotions per costume
   → VNCCS_Pipe → KSampler → emotion sheets
   → VNCCSSheetManager(split) → individual frames
   → CharacterSheetCropper → clean crops
   → VNCCS_RMBG2 → transparent sprites

Stage 2: LoRA TRAINING (optional, for main characters)
   DatasetGenerator → captioned training set
   → external LoRA trainer (kohya_ss)
   → trained LoRA for character identity

Stage 3: ANIMATION SPRITES
   HY-Motion CLI → joint rotation keyframes
   → BodyMeshRenderer (4 angles × N frames)
   → VNCCS_QWEN_Encoder(mesh, character, openpose)
   → VNCCS_RMBG2 → transparent animation frames

Stage 4: SHEET ASSEMBLY
   VNCCSSheetManager(compose) → sprite sheets
   → export_godot.py → Godot .tres + .png
```

---

## Emotion List (Complete)

For reference, the full list of emotion tags available in EmotionGenerator:

**Emotions:** angry, anger vein, annoyed, clenched teeth, scowl, blush, blush stickers, embarrassed, full-face blush, nose blush, bored, closed eyes, confused, crazy, despair, determined, disappointed, disdain, disgust, drunk, envy, excited, exhausted, expressionless, facepalm, flustered, frustrated, furrowed brow, grimace, guilt, happy, kubrick stare, lonely, nervous, nosebleed, one eye closed (winking), round mouth, open mouth, parted lips, pain, pout, raised eyebrow, raised inner eyebrows, rape face, rolling eyes, sad, depressed, frown, gloom (expression), tears, scared, panicking, worried, serious, sigh, sleepy, sulking, surprised, thinking, pensive, v-shaped eyebrows, wince

**Smile:** :d, crazy smile, evil smile, fingersmile, forced smile, glasgow smile, grin, evil grin, light smile, sad smile, seductive smile, smile, stifled laugh

**Smug:** doyagao, smirk, smug, troll face

**Surprised/Scared/Sad:** color drain, depressed, despair, gloom, horrified, screaming, sobbing, traumatized, turn pale, wavy mouth

**Emotes:** ;), :d, ;d, xd, d:, :3, ;3, x3, 3:, 0w0, uwu, :p, ;p, :q, ;q, >:), >:(, :t, :i, :/, :|, x mouth, :c, c:, :<, ;<, diamond mouth, :>, :>=, :o, ;o, o3o, >3<, o_o, 0_0, |_|, ._. , solid circle eyes, heart-shaped eyes, ^_^, \(^o^)/, ^q^, >_<, >o<, @_@, >_@, +_+, +_-, =_=, t_t, <o>_<o>, <|>_<|>

---

## Key Insight

We were rebuilding functionality that VNCCS already provides:
- **Background removal** → VNCCS_RMBG2 (we added `--remove-bg` with rembg, now using VNCCS_RMBG2)
- **Sprite extraction** → SpriteGenerator + VNCCSSheetManager (we were doing manual extraction)
- **Emotion generation** → EmotionGenerator (we weren't using at all)
- **Training data** → DatasetGenerator (we discussed LoRA training but didn't know this existed)
- **Camera angles** → PositionControl (we were manually rotating the mesh)
- **Face fixing** → VNCCS_QWEN_Detailer (we noted face artifacts but didn't know this existed)

The correct approach is: **use VNCCS as the platform, add HY-Motion and TRELLIS as extensions.**
