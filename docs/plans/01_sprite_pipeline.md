# Sprite-Based 2.5D Animation Pipeline

## Vision

Generate game-ready anime sprite sheets from AI-driven pose keyframes.
Pose Director's LLM generates bone rotations → VNCCS renders and edits → sprite sheets for Godot.

**We are a VNCCS project.** Our unique value is AI-driven pose generation that replaces
VNCCS Pose Studio's manual sliders. Everything else (rendering, editing, sheet assembly)
uses VNCCS's existing infrastructure.

## Architecture Decision

```
VNCCS Pose Studio:  Manual sliders → bone rotations → mesh render → QWEN edit → sprites
Pose Director:      LLM text prompt → bone rotations → mesh render → QWEN edit → sprites
                                                       ↑
                                            Same downstream pipeline
```

We don't build parallel infrastructure. We replace VNCCS Pose Studio's manual UI with
programmatic LLM-driven pose generation. The CLI unlocks batch jobs, variations, retries,
and scripted animation sequences that the manual studio can't do.

## Game Context (Tech Noir)

- 2.5D horror game in Godot (Fear & Hunger meets Danganronpa)
- Full anime character sprites on Sprite3D billboards in 3D world
- Classic frame-swap animation (like fighting games)
- LTX Video cutscenes only for dramatic moments (death, transformation, reveals)
- Visual style: Paper Mario depth + anime aesthetic

## Pipeline Overview

```
Stage A: Character         Z-Image Turbo → character portrait (one-time)
Stage B: Pose Generation   LLM text → VNCCS-format bone rotations (our differentiator)
Stage C: Sprite Frames     VNCCS Pose Studio → VNCCS QWEN Edit → character in each pose
Stage D: Sprite Sheet      VNCCS SheetManager → sprite sheets → Godot metadata
Stage E (optional): VFX    LTX Video → cutscene MP4 for key moments
```

## VNCCS Infrastructure We Use

| Component | VNCCS Node | What it does |
|-----------|-----------|--------------|
| Mesh rendering | `BodyMeshRenderer` | Anny + pyrender (high-quality 3D, works in CLI) |
| Character editing | `VNCCS_QWEN_Encoder` | 3-image conditioning (mesh + char + openpose) |
| OpenPose extraction | `OpenposePreprocessor` | Skeleton overlay from mesh render |
| Sheet splitting | `CharacterSheetCropper` | Contour-based crop from sheets |
| Sheet assembly | `SheetManager` | 2x6 grid compose/split |
| Sprite export | `SpriteGenerator` | Batch sprite output by costume/emotion |
| Camera control | `VNCCS_VisualPositionControl` | Azimuth/elevation for directional variants |
| Pose presets | `PoseLibrary` | 20+ preset pose JSONs |

## What We Build (Our Differentiator)

### 1. LLM-Driven Pose Generation
- Text description → VNCCS-format bone rotations
- Output matches VNCCS Pose Studio's `pose_data` JSON format exactly
- Supports: keyframe sequences, walk cycles, combat animations, idle loops
- CLI-driven: batch jobs, variations, automated retries

### 2. Animation Definition & Batch Runner
- Define full animation sets as YAML/JSON
- Script renders all directions, all animations for a character
- Hands-off: describe the animation, get back the sprite sheet

### 3. Godot Integration
- Metadata JSON → Godot .tres converter
- Sprite3D controller GDScript
- Direction-aware animation system

## VNCCS Pose Data Format

The `pose_data` JSON that VNCCS_PoseStudio accepts:

```json
{
  "mesh": {
    "age": 25.0, "gender": 0.5, "weight": 0.5, "muscle": 0.5,
    "height": 0.5, "breast_size": 0.5, "firmness": 0.5,
    "penis_len": 0.5, "penis_circ": 0.5, "penis_test": 0.5
  },
  "export": {
    "view_width": 512, "view_height": 512,
    "cam_zoom": 1.0,
    "output_mode": "LIST",
    "grid_columns": 2,
    "bg_color": [40, 40, 40]
  },
  "poses": [
    {
      "bones": {"upperarm_r": [rx, ry, rz], "lowerarm_r": [rx, ry, rz]},
      "modelRotation": [rx, ry, rz]
    }
  ],
  "lights": [{"type": "ambient", "intensity": 1.0}]
}
```

### VNCCS Bone Names (MakeHuman game_engine skeleton, 44 bones)

```
Root, pelvis
spine_01, spine_02, spine_03
neck_01, head
upperarm_l, upperarm_r, lowerarm_l, lowerarm_r, hand_l, hand_r
thigh_l, thigh_r, calf_l, calf_r, foot_l, foot_r, ball_l, ball_r
thumb_01_l/r, thumb_02_l/r, thumb_03_r
index_01_l/r, index_02_l/r
middle_01_l/r, middle_02_l/r, middle_03_l
ring_01_l/r, ring_02_l/r, ring_03_l
pinky_01_l/r, pinky_02_l/r, pinky_03_r
```

### Anny → VNCCS Bone Mapping

| Anny | VNCCS | Notes |
|------|-------|-------|
| spine | spine_01 + spine_02 | Split rotation across two joints |
| chest | spine_03 | Upper torso |
| neck | neck_01 | |
| head | head | |
| right_shoulder | upperarm_r | |
| left_shoulder | upperarm_l | |
| right_elbow | lowerarm_r | |
| left_elbow | lowerarm_l | |
| right_wrist | hand_r | |
| left_wrist | hand_l | |
| right_hip | thigh_r | |
| left_hip | thigh_l | |
| right_knee | calf_r | |
| left_knee | calf_l | |
| right_ankle | foot_r | |
| left_ankle | foot_l | |

## Stage Details

### Stage A: Character (one-time)
- Z-Image Turbo → 768x1024 character portrait
- Stored as ComfyUI input image, referenced in all downstream VNCCS edits

### Stage B: Pose Generation (OUR CORE)

**Input**: Text description of animation
**Output**: VNCCS-format `pose_data` JSON with bone rotations per keyframe

Example animation definition:
```yaml
character: luna
animations:
  idle:
    frames: 4
    fps: 8
    loop: true
    directions: [front, back, left, right]
    description: "Subtle breathing, slight weight shift"
  walk:
    frames: 6
    fps: 12
    loop: true
    directions: [front, back, left, right]
    description: "Natural walk cycle, full stride"
  attack:
    frames: 8
    fps: 12
    loop: false
    directions: [front]
    description: "Overhead sword strike, wind-up to follow-through"
```

LLM generates bone rotations for each frame:
```json
{
  "poses": [
    {"bones": {"upperarm_r": [-5, -9, -47]}, "modelRotation": [0, 0, 0]},
    {"bones": {"upperarm_r": [-30, -15, -90]}, "modelRotation": [0, 0, 0]},
    {"bones": {"upperarm_r": [-93, 2, 37], "lowerarm_r": [-24, -4, 104]}, "modelRotation": [0, 0, 0]},
    ...
  ]
}
```

**Modes**:
- **Automatic**: Text → LLM → rotations (fast prototyping)
- **Curated**: Load from VNCCS PoseLibrary presets (proven poses)
- **HY-Motion**: Text → motion diffusion → BVH → convert to VNCCS format (trained animation)
- **Hybrid**: Start with HY-Motion, refine specific joints with LLM

### Stage C: Sprite Frame Rendering (VNCCS)

Per-frame flow through existing VNCCS infrastructure:
1. `VNCCS_PoseStudio` — renders MakeHuman mesh from bone rotations
2. `OpenposePreprocessor` — extracts skeleton overlay from mesh render
3. `VNCCS_QWEN_Encoder` — 3-image conditioning:
   - image1 = mesh render (pose reference)
   - image2 = character portrait (identity reference)
   - image3 = openpose skeleton (structural guide)
4. `KSampler` — 4-step Lightning inference (~5s per frame)

For directional variants, `modelRotation` rotates the whole body:
- Front: `[0, 0, 0]`
- Right: `[0, 90, 0]`
- Back: `[0, 180, 0]`
- Left: `[0, 270, 0]`

### Stage D: Sprite Sheet Assembly (VNCCS + Godot)

VNCCS's `SheetManager` packs into 2x6 grids. For Godot we also generate:
```json
{
  "animation": "walk",
  "direction": "front",
  "frames": 6,
  "frame_width": 512,
  "frame_height": 768,
  "fps": 12,
  "loop": true
}
```

### Stage E: Cutscene VFX (optional, LTX Video)
- Only for dramatic moments — not gameplay
- Use existing LTX Video pipeline on selected frames
- Death scenes, transformations, boss intros
- Pre-rendered MP4s played via Godot VideoStreamPlayer

## Godot Integration

### Sprite3D Setup
```
Character (Spatial)
  └── Sprite3D
        ├── texture: walk_front.png
        ├── hframes: 6
        └── script: character_sprite.gd
```

### GDScript Animation Controller
```gdscript
extends Sprite3D

var animations = {}
var current_anim = "idle"
var current_dir = "front"
var frame = 0
var frame_timer = 0.0

func play(anim_name, direction = "front"):
    current_anim = anim_name
    current_dir = direction
    frame = 0
    texture = load("res://sprites/%s_%s.png" % [anim_name, direction])

func _process(delta):
    frame_timer += delta
    var fps = animations[current_anim]["fps"]
    if frame_timer >= 1.0 / fps:
        frame_timer = 0.0
        frame = (frame + 1) % animations[current_anim]["frames"]
        if frame == 0 and not animations[current_anim]["loop"]:
            frame = animations[current_anim]["frames"] - 1
            return
        self.frame = frame
```

## Implementation Priority

### Phase 1: VNCCS Integration (use what exists)
1. ~~Build VNCCS-format pose output in MotionDirector~~ (done — anny_compat handles VNCCS bone names)
2. ~~Test: BodyMeshRenderer → VNCCS_QWEN_Encoder → sprite frames~~ (done — ~30s/frame, tested 4 poses)
3. ~~Add direction support via modelRotation~~ (done — model_rotation_y on BodyMeshRenderer, 4 directions)
4. ~~Batch runner script for full animation sets~~ (done — run_sprite_batch.py with presets)
5. ~~Background removal for transparent sprites~~ (done — rembg u2net, --remove-bg flag)

### Phase 2: Sprite Assembly & Godot
1. Metadata JSON → Godot .tres converter
2. Sprite3D controller script
3. Direction-aware billboard system

### Phase 3: Cutscene System
1. Select animations for LTX cutscene upgrade
2. Godot VideoStreamPlayer integration
3. Sprite → cutscene → sprite transitions

## Required Models/Tools

| Item | Status | Notes |
|------|--------|-------|
| Z-Image Turbo (character) | Installed | Stage A |
| VNCCS QWEN Edit 2511 | Installed | Stage C |
| Lightning 4-step LoRA | Installed | Fast inference |
| PoseStudio V2 LoRA | Installed | Pose accuracy |
| VNCCS Pose Studio | Installed | Mesh rendering |
| VNCCS Pose Library | Installed | 20+ preset poses |
| VNCCS SpriteGenerator | Installed | Sprite export |
| VNCCS SheetManager | Installed | Sheet assembly |
| Background removal | Installed | rembg u2net (--remove-bg flag) |
| Godot 4.x | User installs | Game engine |

## Why This Architecture

1. **Don't fight VNCCS, extend it** — their mesh, renderer, and editor already work
2. **Our value is AI poses** — LLM + CLI replaces manual sliders
3. **CLI unlocks automation** — batch jobs, variations, retries, scripted sequences
4. **Frame-swap is proven** — every fighting game, Paper Mario, Fear & Hunger
5. **No temporal coherence needed** — each frame is independent
6. **Fast iteration** — change one pose, re-render one frame, seconds not minutes
7. **Cutscene upgrade path** — keep LTX Video for the moments that matter
