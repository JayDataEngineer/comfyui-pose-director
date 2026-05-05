# Character Asset Build System — Declarative Schema

## Core Concept

A character is a YAML file. It declares what assets should exist.
A build script reads the YAML, checks the filesystem, and generates only what's missing.

Adding a death animation to an existing monster = add 3 lines to YAML, re-run build.

## Package Model

Five packages. Composable. Order-independent (build system resolves dependencies).

```
┌─────────────────────────────────────────────────┐
│  CHARACTER YAML (the build manifest)             │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │generate  │  │portraits │  │sprites   │       │
│  │(identity)│  │(emotions)│  │(motion)  │       │
│  └──────────┘  └──────────┘  └──────────┘       │
│       │              │              │             │
│  ┌──────────┐  ┌──────────┐       │             │
│  │outfit    │  │state     │       │             │
│  │(clothing)│  │(condition│       │             │
│  └──────────┘  └──────────┘       │             │
│       │              │              │             │
│       └──────────────┴──────────────┘             │
│                  ↓                                 │
│           OUTPUT DIRECTORY                         │
│       {character}/{outfit}/{state}/                │
│       {animation}/{direction}/frame_NNN.png        │
└─────────────────────────────────────────────────┘
```

---

## YAML Schema

### Minimal Example — T1 Background NPC

```yaml
character:
  name: shopkeeper
  description: "middle-aged woman, warm face, curly brown hair, apron"

generation:
  source: generate          # "generate" (Z-Image) or "import" (existing image)
  quality: turbo            # "turbo" (8 steps) or "base" (50 steps)
  seed: 42
  prompt: >
    90s OVA anime cel of a middle-aged woman, warm round face, curly brown hair
    tied back, wearing a shop apron over a simple dress. Kind eyes, slight smile.
    Clean white background. Cel-shaded animation, muted color palette.

packages:
  portraits:
    emotions: [neutral, happy]

  sprites:
    outfit: default          # uses whatever's in the generated image
    directions: [front]      # front only for background NPCs
    animations:
      - name: idle
        motion: shared       # from pre-generated shared motion library
      - name: walk
        motion: shared
```

**Build output:**
```
shopkeeper/
├── design/approved.png
├── portraits/default/neutral.png
├── portraits/default/happy.png
└── sprites/default/normal/idle/front/frame_001.png ... frame_006.png
                       walk/front/frame_001.png ... frame_008.png
```

---

### Full Example — T2 Named Character

```yaml
character:
  name: marty
  description: "young man, short messy black hair, tired eyes, leather jacket"
  tier: t2                   # enables TRELLIS pass, multi-direction

generation:
  source: generate
  quality: turbo
  seed: 1337
  prompt: >
    90s OVA anime cel of a young man in his early 20s, short messy black hair,
    tired bloodshot eyes, pale skin, wearing a worn leather jacket over a
    dark turtleneck, dark pants, boots. Stern expression. Clean white background.
    Cel-shaded animation, visible paint strokes, muted color palette, anime screencap from 1992.

packages:
  portraits:
    emotions: [neutral, happy, sad, angry, scared, determined, suspicious, exhausted, smirk, thinking]
    # 10 emotions × 1 portrait each = 10 VNCCS generations

  sprites:
    outfit: casual           # name for this outfit set
    directions: [front, back, left, right]
    animations:
      - name: idle
        motion: custom       # generate new with HY-Motion
        prompt: "nervous fidgeting, shifting weight, looking around anxiously"
      - name: walk
        motion: custom
        prompt: "cautious paranoid walk, hunched shoulders, looking left and right"
      - name: run
        motion: custom
        prompt: "desperate sprint, arms pumping, leaning forward"
      - name: die
        motion: custom
        prompt: "dramatic slow collapse, reaching out hand, falling to knees then forward"

  outfit:
    name: armor
    description: "worn leather armor with metal shoulder plates, studded bracers"
    # inherits all declared animations from sprites package
    # generates walk/idle/run/die in armor outfit

  state:
    name: beatup
    description: "bruised face, black eye, bloody nose, torn jacket, scraped hands"
    apply_to:
      - portraits            # re-edit all portraits
      - animations: [idle]   # only idle gets beat-up state, not walk/run/die
```

**Build output:**
```
marty/
├── design/approved.png
├── reference/trellis_model.glb
├── portraits/
│   ├── casual/neutral.png, happy.png, sad.png, ... (10)
│   └── casual_beatup/neutral.png, happy.png, ... (10)
└── sprites/
    ├── casual/normal/idle/front/frame_001.png ... (6 × 4 dirs = 24)
    │                  walk/front/ ... (8 × 4 dirs = 32)
    │                  run/front/ ... (8 × 4 dirs = 32)
    │                  die/front/ ... (8 × 4 dirs = 32)
    ├── casual/beatup/idle/front/ ... (24)
    ├── armor/normal/idle/front/ ... (24)
    │              walk/front/ ... (32)
    │              run/front/ ... (32)
    │              die/front/ ... (32)
    └── armor/beatup/idle/front/ ... (24)
```

---

### Import Example — Existing Monster

```yaml
character:
  name: shadow_beast
  description: "large quadruped shadow creature, glowing red eyes, wispy dark form"

generation:
  source: import             # use existing image, don't generate
  image: ./reference/shadow_beast_concept.png

packages:
  portraits:
    emotions: [neutral, angry]  # monsters don't have many emotions

  sprites:
    outfit: default
    directions: [front, back, left, right]
    animations:
      - name: idle
        motion: custom
        prompt: "prowling beast, low stance, tail swaying, breathing heavily"
      - name: attack
        motion: custom
        prompt: "leaping forward, claws extended, jaws open, aggressive pounce"
      - name: die
        motion: custom
        prompt: "collapsing, body dissolving into dark particles, final roar"
```

---

### Add-On Example — Death Animation for Existing Character

Just append to the existing YAML:

```yaml
  # ... existing packages ...

  sprites:
    # ... existing animations ...
      - name: die
        motion: custom
        prompt: "dramatic slow collapse, reaching out hand, falling to knees"
```

Re-run build. It skips everything that already exists. Only generates `die/` frames.

---

## Motion Sources

```yaml
animations:
  # Option 1: Shared — from pre-generated library, zero cost
  - name: walk
    motion: shared           # looks up motions/shared/walk.npz

  # Option 2: Custom — generate with HY-Motion CLI
  - name: sneak
    motion: custom
    prompt: "stealthy crouching walk, low to ground, careful steps"

  # Option 3: Manual — specify joint rotations directly in YAML
  - name: pose_tpose
    motion: manual
    frames:
      - {right_shoulder: [0, 0, 90], left_shoulder: [0, 0, -90]}  # arms out

  # Option 4: Reference — from an NPZ file (e.g. from a previous HY-Motion run)
  - name: special_dance
    motion: file
    path: ./motions/special_dance.npz
```

---

## Quality Settings

Quality is a parameter on the generation package, not a separate pipeline.

```yaml
generation:
  quality: turbo             # affects Z-Image character generation only
  # turbo: 8 steps, CFG 1.0, res_multistep sampler (fast, slightly less detailed)
  # base:  50 steps, CFG 4.0, euler sampler (slow, more detail, better anatomy)
```

VNCCS sprite generation always uses Lightning 4-step (already optimized).

---

## Build State Persistence

Every character gets a `build_state.json` in its output directory. This is the build cache — it tracks what was built, with what parameters, and where intermediates live.

### Why not just check the filesystem?

Checking if `frame_001.png` exists isn't enough:
- Was it built with the CURRENT prompt, or an old one?
- Do the prerequisite intermediates (TRELLIS model, NPZ) still exist?
- Can we reuse the NPZ if the motion prompt didn't change?

The build state answers these questions.

### Build State Schema

```json
{
  "character": "shadow_beast",
  "yaml_hash": "a1b2c3",
  "last_build": "2026-05-03T14:30:00",

  "generation": {
    "source": "import",
    "input_image": "./reference/shadow_beast_concept.png",
    "input_hash": "sha256:abc123...",
    "output": "output/shadow_beast/design/approved.png"
  },

  "trellis": {
    "status": "complete",
    "model_path": "output/shadow_beast/reference/trellis_model.glb",
    "reference_images": {
      "front": "output/shadow_beast/reference/front.png",
      "back": "output/shadow_beast/reference/back.png",
      "left": "output/shadow_beast/reference/left.png",
      "right": "output/shadow_beast/reference/right.png"
    }
  },

  "portraits": {
    "default": {
      "status": "complete",
      "emotions_built": ["neutral", "angry"],
      "seed": 42
    }
  },

  "sprites": {
    "default/normal/idle": {
      "status": "complete",
      "directions": ["front", "back", "left", "right"],
      "frames_per_dir": 6,
      "total_frames": 24,
      "motion_source": "custom",
      "motion_prompt": "prowling beast, low stance, tail swaying, breathing heavily",
      "npz_path": "motions/custom/shadow_beast_idle.npz",
      "seed": 42
    },
    "default/normal/attack": {
      "status": "complete",
      "directions": ["front", "back", "left", "right"],
      "frames_per_dir": 8,
      "total_frames": 32,
      "motion_source": "custom",
      "motion_prompt": "leaping forward, claws extended, jaws open, aggressive pounce",
      "npz_path": "motions/custom/shadow_beast_attack.npz",
      "seed": 42
    },
    "default/normal/die": {
      "status": "pending",
      "directions": ["front", "back", "left", "right"],
      "frames_per_dir": 8,
      "motion_source": "custom",
      "motion_prompt": "collapsing, body dissolving into dark particles, final roar"
    }
  }
}
```

### Key Fields

| Field | Purpose |
|-------|---------|
| `yaml_hash` | Hash of the character YAML. If changed, diff against state to find what needs rebuilding |
| `motion_prompt` | If this changed for an animation, regenerate NPZ + sprites |
| `npz_path` | Cached intermediate. Don't re-run HY-Motion if NPZ exists and prompt unchanged |
| `trellis.model_path` | Cached intermediate. Don't re-run TRELLIS if generation input unchanged |
| `input_hash` | SHA256 of the source image. Detect if the source changed |
| `seed` | Reproducibility — same seed = same output |
| `status` | `pending` / `complete` / `failed` — drives what needs building |

### Rebuild Triggers

| Trigger | What happens |
|---------|-------------|
| Sprite directory missing frames | Build those frames |
| YAML motion prompt changed for an animation | Regenerate NPZ + all sprite frames for that animation |
| YAML emotion list grew | Generate only the NEW emotions |
| YAML outfit added | Generate all declared animations in the new outfit |
| Source image changed (different hash) | Rebuild everything downstream (TRELLIS, sprites, portraits) |
| `--rebuild sprites.default.normal.die` | Force regenerate that specific animation |
| `--clean` | Delete build state + output, start from scratch |

### Adding to an Existing Character

Scenario: shadow_beast was built with idle + attack. Now we want to add `die`.

```bash
# 1. Add to YAML
vim characters/shadow_beast.yaml   # add die animation entry

# 2. Re-run build
uv run python scripts/build_character.py characters/shadow_beast.yaml

# Build system:
#   Reads YAML → reads build_state.json
#   generation: complete (skip)
#   trellis: complete, model_path exists (skip)
#   portraits: complete, emotions match (skip)
#   sprites/default/normal/idle: complete (skip)
#   sprites/default/normal/attack: complete (skip)
#   sprites/default/normal/die: status=pending → BUILD
#     1. HY-Motion CLI "collapsing, body dissolving..." → motions/custom/shadow_beast_die.npz
#     2. Load TRELLIS model from build_state.trellis.model_path
#     3. Apply die NPZ rotations → render per direction
#     4. VNCCS edit per frame → save sprites
#     5. Update build_state: die → complete, record npz_path
```

Nothing else regenerates. TRELLIS model, idle sprites, attack sprites — all untouched.

### Re-doing a Bad Animation

The idle animation looked wrong. We want to regenerate it.

```bash
# Option A: Force rebuild just that animation (keeps existing NPZ, re-runs VNCCS)
uv run python scripts/build_character.py characters/shadow_beast.yaml --rebuild sprites.default.normal.idle

# Option B: Change the prompt, then rebuild (regenerates NPZ too)
# Edit YAML: change idle prompt to "prowling beast, low crouch, muscles tense"
uv run python scripts/build_character.py characters/shadow_beast.yaml
# Build system detects motion_prompt changed → deletes old NPZ → regenerates

# Option C: Different seed, same everything else
uv run python scripts/build_character.py characters/shadow_beast.yaml --rebuild sprites.default.normal.idle --seed 99
```

---

## Build System CLI

```bash
# Build a character (generates only what's missing)
uv run python scripts/build_character.py characters/marty.yaml

# Force rebuild a specific package
uv run python scripts/build_character.py characters/marty.yaml --rebuild sprites.casual.normal.idle
uv run python scripts/build_character.py characters/marty.yaml --rebuild portraits
uv run python scripts/build_character.py characters/marty.yaml --rebuild trellis

# Override seed for this run
uv run python scripts/build_character.py characters/marty.yaml --seed 99

# Build multiple characters in sequence
uv run python scripts/build_character.py characters/*.yaml

# Dry run (show what would be generated, don't execute)
uv run python scripts/build_character.py characters/marty.yaml --dry-run

# Nuclear option — delete all outputs and rebuild
uv run python scripts/build_character.py characters/marty.yaml --clean
```

### Build Logic

```
1. Parse YAML → validate schema
2. Load build_state.json (or create empty)
3. Compute yaml_hash, compare with state
4. Determine build plan:
   a. generation: needed? → if source=generate, run Z-Image; if import, copy
   b. trellis: needed? (t2+ and not in state) → run TRELLIS pipeline
   c. portraits: for each emotion not in state → run emotion workflow
   d. sprites: for each animation not in state:
      - motion=shared → load motions/shared/{name}.npz
      - motion=custom → run HY-Motion CLI → save NPZ
      - motion=file → load from path
      - motion=manual → build NPZ from joint data
      - Render frames (Anny for t1, TRELLIS for t2+)
      - VNCCS edit per frame
      - RMBG2 per frame
   e. outfit: for each outfit, inherit animations from sprites package → re-run sprite gen
   f. state: for each state, re-edit declared targets
5. Execute plan, update build_state.json after each step
6. Report summary: X generated, Y skipped, Z failed
```

### Dependency Resolution

Packages auto-resolve in dependency order:

```
generation → portraits (needs reference image)
generation → trellis → sprites (needs 3D model for t2+)
sprites → outfit (needs to know which animations exist)
sprites/state → state (needs existing frames to re-edit)
```

If `--rebuild sprites` is specified, it does NOT rebuild trellis unless the model is missing.
If `--rebuild portraits` is specified, it does NOT rebuild generation unless the source image is missing.

---

## File Structure

```
comfyui-pose-director/
├── characters/                    # YAML build manifests
│   ├── marty.yaml
│   ├── shadow_beast.yaml
│   └── shopkeeper.yaml
├── motions/
│   ├── shared/                    # pre-generated shared motion library
│   │   ├── idle.npz
│   │   ├── walk_front.npz
│   │   └── talk_gesture.npz
│   └── custom/                    # generated per-character (cached)
│       ├── marty_walk.npz
│       └── shadow_beast_attack.npz
├── output/
│   ├── marty/                     # built assets
│   │   ├── design/
│   │   ├── reference/
│   │   ├── portraits/
│   │   └── sprites/
│   └── shadow_beast/
└── scripts/
    └── build_character.py         # the build system
```

---

## What the Build Script Needs to Do

For each package, map to a ComfyUI workflow execution:

| Package | Workflow | Inputs |
|---------|----------|--------|
| generation (generate) | `00_z_image_character.json` | prompt, seed, quality |
| generation (import) | copy file | image path |
| TRELLIS (t2+) | `run_trellis_pipeline.py` | character image |
| portraits | new: `emotion_portrait_workflow.json` | character ref, emotions list |
| sprites (T1) | `05_sprite_bodymesh_edit.json` | Anny mesh + character ref |
| sprites (T2+) | new: `trellis_sprite_workflow.json` | TRELLIS model + HY-Motion NPZ |
| outfit | VNCCS edit workflow | character ref + outfit description |
| state | VNCCS re-edit workflow | existing sprite + state description |

Two new workflows needed:
1. **emotion_portrait_workflow.json** — EmotionGenerator → VNCCS → RMBG2 (for portraits)
2. **trellis_sprite_workflow.json** — TRELLIS render + VNCCS edit + RMBG2 (for T2+ sprites)

---

## YAML Validation

The build script validates the YAML before executing:
- `name` is required and must be filesystem-safe
- If `source: generate`, `prompt` is required
- If `source: import`, `image` path must exist
- `emotions` must be from the VNCCS emotion list
- `motion` must be one of: shared, custom, manual, file
- `directions` must be subset of: front, back, left, right
- `tier` determines which pipeline is used (t1=Anny, t2+=TRELLIS)
- `outfit` without preceding `sprites` package = error

---

## Tier Defaults

The tier sets sensible defaults that can be overridden:

| Parameter | T1 | T2 | T3 |
|-----------|----|----|-----|
| directions | [front] | [front, back, left, right] | [front, back, left, right] |
| body source | Anny (shared) | TRELLIS (per-character) | TRELLIS + LoRA |
| motion source | shared | custom | custom |
| emotions | 3-5 | 10-15 | 20-30 |
| outfits | default only | 1-2 declared | all declared |
| states | none | declared | all + LoRA-enhanced |
| quality | turbo | turbo | base (for initial gen) |

You can override any default in the YAML. T1 with 4 directions is valid — it just costs more time.
