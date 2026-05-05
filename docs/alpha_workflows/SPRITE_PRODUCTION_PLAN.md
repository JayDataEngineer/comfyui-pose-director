# Sprite Production Plan — Tech Noir

## Asset Types a 2D Game Needs

**Portraits** — face/shoulders for dialogue boxes
- Emotion-driven (neutral, happy, sad, angry, scared, etc.)
- Small resolution (~256x256)
- Transparent background
- Outfit mostly irrelevant (face + shoulders only)
- Pipeline: EmotionGenerator → VNCCS → RMBG2

**Gameplay Sprites** — full body for world movement
- Pose/animation-driven (walk, idle, run, interact, die)
- Direction-dependent (front, back, left, right = 4 dirs)
- 6-12 frames per animation per direction
- Outfit-specific (same pose, different clothes = different sprite)
- Transparent background, consistent size
- Pipeline: HY-Motion → BodyMeshRenderer or TRELLIS render → VNCCS → RMBG2

**Cutscene Poses** — dramatic single-frame moments
- Full body, higher resolution
- Emotion + pose + outfit + state all visible
- Scene-specific ("standing over the body")
- Pipeline: custom VNCCS edit with specific instructions

**State Overlays** — conditions applied on top of existing sprites
- Beat up, wet, bloody, possessed, injured
- These are RE-EDITS of existing sprites via VNCCS
- Only applied where visible (portraits yes, zoomed-out gameplay maybe not)

---

## The Layers (Orthogonal)

```
Identity  — WHO is this?  (face, body type, hair, features)
  ↓
Outfit    — WHAT are they wearing?  (base clothing)
  ↓
State     — HOW do they look?  (normal, beat up, wet, possessed)
  ↓
Direction — WHICH way are they facing?  (front, back, left, right)
  ↓
Pose      — WHAT are they doing?  (walk, idle, attack, talk)
  ↓
Frame     — WHEN in the animation?  (frame 01-12)
```

Each layer multiplies the total sprite count. A character with 2 outfits, 1 state, 4 directions, 3 animations, 8 frames = 192 gameplay sprites. Plus portraits.

---

## Tiers (Investment per Character)

### T0 — Design Pass
**Purpose:** Explore designs, pick one. No commitment.

- Z-Image Turbo (or provided image)
- Generate 5-10 variants, different seeds
- Human gate: "this design"
- Output: single character image

**Cost:** ~10 min
**Tools:** Z-Image, PromptEnhancer

---

### T1 — Background NPC
**Purpose:** Appears in world, maybe says 1-2 lines. Generic walk, few emotions.

**Pipeline:**
1. CharacterCreator → define attributes, create directory structure
2. 3-5 emotion portraits (neutral, happy, sad, angry, surprised)
   - EmotionGenerator → VNCCS_Pipe → KSampler → RMBG2 → transparent PNG
3. Front-only walk + idle using shared motion library
   - Shared HY-Motion NPZs (pre-generated once, reused for ALL T1 characters)
   - BodyMeshRenderer (Anny body — generic, close enough for background NPCs)
   - VNCCS_QWEN_Encoder(image1=Anny_mesh, image2=character_photo) → RMBG2
4. No outfit variants, no state variants, no custom animations

**Why Anny is OK here:** Background NPCs are small on screen. Body type mismatch isn't visible at gameplay zoom. Not worth TRELLIS for a character that appears in 3 scenes.

**Shared motion library (generate ONCE):**
- `motions/shared/idle.npz`
- `motions/shared/walk_front.npz`
- `motions/shared/talk_gesture.npz`

**Cost:** ~30 min per character (after shared motions exist)
**Total sprites:** ~3-5 portraits + ~20 sprites (walk+idle, front only) = ~25

---

### T2 — Named Character
**Purpose:** Recurring NPC, ally, boss. Custom personality, recognizable from any angle.

**Pipeline:**
Everything from T1, PLUS:

1. **TRELLIS pass** — Image → 3D model with auto-rigged skeleton
   - This character now has THEIR OWN body mesh, not generic Anny
   - Render reference views (front, back, side) for VNCCS reference images
   - SOLVES the body type mismatch — fat shopkeeper walks with HIS body

2. **Multi-angle sprites** — all 4 directions
   - HY-Motion → apply rotations to TRELLIS skeleton → render per direction
   - TRELLIS model rendered from front/back/left/right in each pose
   - VNCCS_QWEN_Encoder(image1=TRELLIS_pose_render, image2=character_photo)

3. **Custom HY-Motion animations** — character-specific movement
   - "Old man limping walk" vs "soldier march" vs "nervous fidget"
   - These drive the character's OWN TRELLIS skeleton

4. **Expanded emotions** — 10-15 curated emotions
   - EmotionGenerator → pick emotions relevant to character's story arc
   - VNCCS_QWEN_Detailer on key portraits to fix face artifacts

5. **1-2 outfit variants** — via CharacterAssetSelector
   - Generate outfit reference images (character in specific clothes)
   - Each outfit needs its own walk/idle sprites

**Cost:** ~60-90 min per character
**Total sprites:** ~15 portraits + ~120 sprites (walk+idle+custom, 4 dirs) × 1-2 outfits = ~250

---

### T3 — Hero Character
**Purpose:** Protagonist, main villain. Hundreds of appearances. Maximum consistency.

**Pipeline:**
Everything from T2, PLUS:

1. **LoRA training** — character identity baked into the model
   - DatasetGenerator → captioned training set from ALL generated sprites
   - External LoRA trainer (needs research on Qwen Image Edit compatibility)
   - After LoRA: every VNCCS generation intrinsically "knows" the character
   - Reference images become less critical — the model has internalized identity

2. **Full emotion set** — 20-30 curated emotions (not all 100+, most are noise)
   - Cover every story beat the character appears in

3. **Full state variants** — beat up, wet, bloody, possessed
   - VNCCS re-edit pass: "take this sprite and add bruises, torn clothes"
   - Applied to portraits and key cutscene poses
   - Gameplay sprites may not need states (too zoomed out)

4. **Full costume wardrobe** — every outfit the character wears
   - Each outfit = full set of walk/idle/action sprites in all 4 directions

5. **TRELLIS model refinement** — higher quality 3D model for cutscene rendering

**Cost:** ~4 hours (LoRA training is the bottleneck at 1-2 hours)
**Total sprites:** ~30 portraits + ~500+ gameplay sprites across outfits/states

---

## State Pipeline (Orthogonal to Tiers)

States are NOT a tier. They're a re-edit pass applied to any completed character.

```
Existing sprite → VNCCS_QWEN_Encoder(
    image1 = original_sprite,
    instruction = "add bruises, torn clothing, bloody nose, exhausted expression"
) → RMBG2 → state_sprite.png
```

States that matter:
- **Beat up** — bruises, torn clothes, bloody nose
- **Wet** — rain-soaked hair and clothes, water drips
- **Bloody** — blood splatter, stained clothes
- **Possessed** — glowing eyes, unnatural posture
- **Injured** — holding arm, limping, bandaged

Only apply where visible:
- Portraits: always apply states (face visible)
- Gameplay sprites: only if the state affects body language (limping walk vs normal walk)
- Cutscene poses: always apply states

---

## Outfit Pipeline (Orthogonal to Tiers)

An outfit is a VNCCS reference image, not a separate generation pipeline.

```
1. Define outfit (CharacterAssetSelector: top, bottom, shoes, head)
2. Generate outfit reference: VNCCS_QWEN_Encoder(
    image1 = character_photo,
    instruction = "draw this character wearing [outfit description]"
) → outfit_reference.png
3. Use outfit_reference.png as image2 for ALL downstream sprite generation
```

Each outfit triggers re-generation of:
- All gameplay sprites (walk/idle/action in all directions)
- Portrait backgrounds change (shoulders visible with different clothes)
- NOT emotions — face doesn't change with outfit

---

## Background Removal

Single pass at the END of the pipeline. Not per-layer.

VNCCS_RMBG2 with BEN2 model, consistent settings across all frames:
- sensitivity: 1.0
- process_res: 1024
- mask_blur: 4
- refine_foreground: true
- background: Alpha

**Edge consistency concern:** If alpha edges vary between frames, sprites shimmer in-game.
Mitigation: test a full walk cycle, overlay all frames, check if the silhouette is stable.
If edges are inconsistent: generate a single reference mask from a T-pose render, apply to all frames.

---

## Pipeline Diagram

```
T0: Z-Image → character.png
     ↓ (human gate)

T1: character.png → CharacterCreator
     → EmotionGenerator(5 emotions) → VNCCS → RMBG2 → portraits/
     → Shared HY-Motion NPZ → BodyMeshRenderer(Anny) → VNCCS → RMBG2 → sprites/
     ↓ (human gate: "this character matters")

T2: character.png → TRELLIS → 3D model + skeleton
     → Custom HY-Motion → drive TRELLIS skeleton → render 4 dirs → VNCCS → RMBG2 → sprites/
     → EmotionGenerator(15 emotions) → VNCCS → RMBG2 → portraits/
     → CharacterAssetSelector → outfit references → repeat sprite gen per outfit
     ↓ (human gate: "this is THE character")

T3: ALL T2 sprites → DatasetGenerator → LoRA training
     → LoRA-stacked VNCCS for all subsequent generations
     → Full state pipeline → state sprites/
     → Full wardrobe → all outfits
```

---

## Unknowns to Test (In Order)

1. **EmotionGenerator → VNCCS → usable portraits?** Wire it up, generate 5 emotions, check quality
2. **TRELLIS → HY-Motion rotations → render?** Verify the skeleton mapping works
3. **Multi-angle VNCCS consistency** — does back-view look like the same character?
4. **VNCCS_RMBG2 edge consistency** across animation frames
5. **LoRA training feasibility** — what trainer supports Qwen Image Edit 2511?
6. **State re-edit quality** — can VNCCS add bruises without changing the character?
7. **Outfit reference quality** — does VNCCS maintain character identity with outfit swap?

Each test is a single workflow execution + visual inspection. Prove the component, then build the tier.

---

## Shared Motion Library (Generated Once)

Pre-generate with HY-Motion CLI, save NPZs:
```
motions/shared/
├── idle.npz          — subtle breathing, weight shift (30 frames)
├── walk_front.npz    — standard walk cycle (24 frames)
├── talk_gesture.npz  — conversational hand movement (20 frames)
├── react.npz         — startled reaction (12 frames)
└── sit.npz           — sitting down (16 frames)
```

T1 characters all use these. T2+ characters get additional custom motions.

---

## File Structure (Per Character)

```
output/{character_name}/
├── design/
│   ├── concept_001.png          — T0 candidates
│   ├── concept_002.png
│   └── approved.png             — selected design
├── reference/
│   ├── front.png                — VNCCS reference images
│   ├── back.png
│   ├── side_left.png
│   ├── side_right.png
│   └── trellis_model.glb        — T2+ 3D model
├── portraits/
│   ├── {outfit}/
│   │   ├── neutral.png
│   │   ├── happy.png
│   │   ├── sad.png
│   │   └── ...
├── sprites/
│   ├── {outfit}/
│   │   ├── {state}/
│   │   │   ├── {animation}/
│   │   │   │   ├── {direction}/
│   │   │   │   │   ├── frame_001.png
│   │   │   │   │   ├── frame_002.png
│   │   │   │   │   └── ...
├── motions/
│   ├── shared/                  — T1 shared motions
│   └── custom/                  — T2+ custom motions
│       ├── walk_casual.npz
│       └── limping.npz
└── lora/                        — T3 only
    ├── dataset/                 — captioned training images
    └── {character}_lora.safetensors
```

---

## Naming Convention

```
{character}_{outfit}_{state}_{animation}_{direction}_{frame}.png

Examples:
marty_casual_normal_idle_front_01.png
marty_casual_normal_walk_right_08.png
marty_casual_beatup_idle_front_01.png
marty_armor_normal_attack_back_04.png
```

Portraits:
```
{character}_{outfit}_{emotion}.png

Examples:
marty_casual_sad.png
marty_armor_angry.png
```
