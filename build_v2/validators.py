"""YAML validation and tier inference for character manifests.

Reads from the `packages.*` structure defined in DECLARATIVE_SCHEMA.md.
Also supports flat keys for backward-compatible programmatic use.
"""

from __future__ import annotations

from pathlib import Path

VALID_EMOTIONS = {
    "neutral", "happy", "sad", "angry", "confused", "determined", "disgusted",
    "excited", "exhausted", "nervous", "pout", "scared", "serious", "surprised",
    "thinking", "grin", "evil grin", "light smile", "sad smile", "forced smile",
    "doyagao", "smirk", "smug", "troll face", "horrified", "screaming", "sobbing",
    "traumatized", "despair", "suspicious",
}

VALID_DIRECTIONS = {"front", "back", "left", "right"}
VALID_MOTION_SOURCES = {"shared", "custom", "manual", "file"}
VALID_QUALITIES = {"turbo", "base"}

TIER_ORDER = {"t0": 0, "t1": 1, "t1.5": 2, "t2": 3, "t3": 4}


# ── Package helpers ──────────────────────────────────────────────────────────

def _pkg(data: dict, key: str):
    """Get a key from `packages.*` dict, falling back to top-level flat key."""
    pkgs = data.get("packages", {})
    return pkgs.get(key, data.get(key))


def _portraits(data: dict) -> dict:
    return _pkg(data, "portraits") or {}

def _sprites(data: dict) -> dict:
    return _pkg(data, "sprites") or {}

def _outfit_pkg(data: dict) -> dict:
    return _pkg(data, "outfit") or {}

def _state_pkg(data: dict) -> dict:
    return _pkg(data, "state") or {}


# ── Tier logic ────────────────────────────────────────────────────────────────

def tier_gte(a: str, b: str) -> bool:
    """Check if tier a >= tier b."""
    return TIER_ORDER.get(a, 0) >= TIER_ORDER.get(b, 0)


def infer_tier(data: dict) -> str:
    """Infer the minimum tier required by the declared packages.

    Returns the highest tier required by any enabled package.

    Tier mapping:
      t0   — import or generate only (identity)
      t1   — character sheet + emotion portraits + static sprites
      t1.5 — t1 + animated sprites (HY-Motion)
      t2   — t1.5 + TRELLIS 3D model + outfit variants
      t3   — t2 + LoRA training
    """
    tier = "t0"

    # T1: sheet, portraits/emotions, static sprites
    if _pkg(data, "sheet") is not None:
        tier = _max_tier(tier, "t1")
    if _portraits(data).get("emotions"):
        tier = _max_tier(tier, "t1")
    if _sprites(data).get("static"):
        tier = _max_tier(tier, "t1")
    if _sprites(data).get("outfit"):
        tier = _max_tier(tier, "t1")

    # T1.5: animated sprites, video
    if _sprites(data).get("animated"):
        tier = _max_tier(tier, "t1.5")
    if _pkg(data, "video"):
        tier = _max_tier(tier, "t1.5")

    # T2: TRELLIS, outfit variants, state effects
    if _pkg(data, "trellis"):
        tier = _max_tier(tier, "t2")
    if _outfit_pkg(data).get("name"):
        tier = _max_tier(tier, "t2")
    if _state_pkg(data).get("name"):
        tier = _max_tier(tier, "t1")

    # T3: LoRA
    if _pkg(data, "lora"):
        tier = _max_tier(tier, "t3")

    # Also check flat keys for backward compat
    for key in ("sheet", "emotions", "trellis", "outfit", "state", "video", "godot", "lora"):
        if data.get(key):
            t = {"lora": "t3", "trellis": "t2", "video": "t1.5"}.get(key, "t1")
            tier = _max_tier(tier, t)

    # Check flat sprites for backward compat
    flat_sprites = data.get("sprites", {})
    if flat_sprites.get("animated"):
        tier = _max_tier(tier, "t1.5")
    if flat_sprites.get("static"):
        tier = _max_tier(tier, "t1")

    return tier


def _max_tier(current: str, candidate: str) -> str:
    return max(current, candidate, key=lambda t: TIER_ORDER.get(t, 0))


def get_enabled_stages(data: dict, declared_tier: str) -> set[str]:
    """Determine which stages are needed from the YAML and tier.

    Returns set of stage names.
    """
    sprites = _sprites(data)
    portraits = _portraits(data)
    outfit_pkg = _outfit_pkg(data)
    state_pkg = _state_pkg(data)

    stages = {"generate"}  # Always needed — identity

    # ── T1 stages ──
    if tier_gte(declared_tier, "t1"):
        # Sheet: always run for t1+ characters (creates VNCCS character sheet)
        stages.add("sheet")
        # Emotions: from portraits.emotions or flat emotions key
        if portraits.get("emotions") or data.get("emotions"):
            stages.add("emotions")
        # Static sprites: from sprites.static  
        if sprites.get("static"):
            stages.add("sprites_static")

    # ── T1.5 stages ──
    if sprites.get("animated") and tier_gte(declared_tier, "t1.5"):
        stages.add("sprites_animated")
    if _pkg(data, "video") and tier_gte(declared_tier, "t1.5"):
        stages.add("video")
    if data.get("video") and tier_gte(declared_tier, "t1.5"):
        stages.add("video")

    # ── T2 stages ──
    if tier_gte(declared_tier, "t2"):
        if _pkg(data, "trellis"):
            stages.add("trellis")
        if outfit_pkg.get("name"):
            stages.add("outfit")
        if data.get("trellis"):
            stages.add("trellis")
        if data.get("outfit"):
            stages.add("outfit")

    # State — t1 minimum
    if (state_pkg.get("name") or data.get("state")) and tier_gte(declared_tier, "t1"):
        stages.add("state")

    # Godot export — t1 minimum
    if (_pkg(data, "godot") or data.get("godot")) and tier_gte(declared_tier, "t1"):
        stages.add("godot")

    # ── T3 stages ──
    if (_pkg(data, "lora") or data.get("lora")) and tier_gte(declared_tier, "t3"):
        stages.add("lora")

    return stages


# ── Validation ───────────────────────────────────────────────────────────────

def validate_character(data: dict, path: Path):
    """Validate character YAML structure. Raises ValueError on issues."""
    char = data.get("character", {})
    if not char.get("name"):
        raise ValueError(f"{path}: character.name is required")
    name = char["name"]
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"{path}: character.name must be filesystem-safe")

    gen = data.get("generation", {})
    source = gen.get("source", "generate")
    if source not in ("generate", "import"):
        raise ValueError(f"{path}: generation.source must be 'generate' or 'import'")
    if source == "generate" and not gen.get("prompt"):
        raise ValueError(f"{path}: generation.prompt is required when source=generate")

    quality = gen.get("quality", "turbo")
    if quality not in VALID_QUALITIES:
        raise ValueError(f"{path}: generation.quality must be one of {VALID_QUALITIES}")

    # Validate sprites from packages.sprites or top-level sprites
    sprites = _sprites(data)
    if not sprites:
        sprites = data.get("sprites", {})

    if sprites:
        directions = sprites.get("directions", ["front"])
        invalid = set(directions) - VALID_DIRECTIONS
        if invalid:
            raise ValueError(f"{path}: invalid directions: {invalid}")

        for anim in sprites.get("animated", []) or []:
            motion = anim.get("motion", "shared")
            if motion not in VALID_MOTION_SOURCES:
                raise ValueError(f"{path}: invalid motion source '{motion}'")
            if motion == "file" and not anim.get("path"):
                raise ValueError(f"{path}: motion=file requires 'path'")
            if motion == "manual" and not anim.get("frames"):
                raise ValueError(f"{path}: motion=manual requires 'frames'")

    # Validate portraits/emotions
    portraits = _portraits(data)
    emotions = portraits.get("emotions", [])
    if not emotions:
        emotions = data.get("emotions", {}).get("list", []) or data.get("emotions", [])
    invalid_emotions = set(emotions) - VALID_EMOTIONS
    if invalid_emotions:
        raise ValueError(f"{path}: invalid emotions: {invalid_emotions}")
