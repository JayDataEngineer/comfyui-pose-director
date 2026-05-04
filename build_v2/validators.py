"""YAML validation and tier inference for character manifests."""

from __future__ import annotations

from pathlib import Path

VALID_EMOTIONS = {
    "neutral", "happy", "sad", "angry", "confused", "determined", "disgusted",
    "excited", "exhausted", "nervous", "pout", "scared", "serious", "surprised",
    "thinking", "grin", "evil grin", "light smile", "sad smile", "forced smile",
    "doyagao", "smirk", "smug", "troll face", "horrified", "screaming", "sobbing",
    "traumatized", "despair",
}

VALID_DIRECTIONS = {"front", "back", "left", "right"}
VALID_MOTION_SOURCES = {"shared", "custom", "manual", "file"}
VALID_QUALITIES = {"turbo", "base"}

# Tier requirements for each package
TIER_REQUIREMENTS = {
    "sheet": "t1",
    "emotions": "t1",
    "sprites_static": "t1",
    "sprites_animated": "t1.5",
    "trellis": "t2",
    "outfit": "t1",
    "state": "t1",
    "video": "t1.5",
    "godot": "t1",
}

TIER_ORDER = {"t0": 0, "t1": 1, "t1.5": 2, "t2": 3, "t3": 4}


def tier_gte(a: str, b: str) -> bool:
    """Check if tier a >= tier b."""
    return TIER_ORDER.get(a, 0) >= TIER_ORDER.get(b, 0)


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

    # Validate sprites
    sprites = data.get("sprites", {})
    if sprites:
        directions = sprites.get("directions", ["front"])
        invalid = set(directions) - VALID_DIRECTIONS
        if invalid:
            raise ValueError(f"{path}: invalid directions: {invalid}")

        for anim in sprites.get("animated", []):
            motion = anim.get("motion", "shared")
            if motion not in VALID_MOTION_SOURCES:
                raise ValueError(f"{path}: invalid motion source '{motion}'")
            if motion == "file" and not anim.get("path"):
                raise ValueError(f"{path}: motion=file requires 'path'")
            if motion == "manual" and not anim.get("frames"):
                raise ValueError(f"{path}: motion=manual requires 'frames'")


def infer_tier(data: dict) -> str:
    """Infer the minimum tier required by the declared packages.

    Returns the highest tier required by any enabled package.
    """
    tier = "t0"

    if data.get("sheet") is not None:
        tier = max(tier, "t1", key=lambda t: TIER_ORDER.get(t, 0))
    if data.get("emotions"):
        tier = max(tier, "t1", key=lambda t: TIER_ORDER.get(t, 0))
    if data.get("sprites", {}).get("static"):
        tier = max(tier, "t1", key=lambda t: TIER_ORDER.get(t, 0))
    if data.get("sprites", {}).get("animated"):
        tier = max(tier, "t1.5", key=lambda t: TIER_ORDER.get(t, 0))
    if data.get("trellis") is not None:
        tier = max(tier, "t2", key=lambda t: TIER_ORDER.get(t, 0))
    if data.get("outfit"):
        tier = max(tier, "t1", key=lambda t: TIER_ORDER.get(t, 0))
    if data.get("state"):
        tier = max(tier, "t1", key=lambda t: TIER_ORDER.get(t, 0))
    if data.get("video"):
        tier = max(tier, "t1.5", key=lambda t: TIER_ORDER.get(t, 0))
    if data.get("godot"):
        tier = max(tier, "t1", key=lambda t: TIER_ORDER.get(t, 0))
    if data.get("lora"):
        tier = max(tier, "t3", key=lambda t: TIER_ORDER.get(t, 0))

    return tier


def get_enabled_stages(data: dict, declared_tier: str) -> set[str]:
    """Determine which stages are needed from the YAML and tier.

    Returns set of stage names.
    """
    stages = {"generate"}  # Always needed

    if data.get("sheet") is not None and tier_gte(declared_tier, "t1"):
        stages.add("sheet")
    if data.get("emotions") and tier_gte(declared_tier, "t1"):
        stages.add("emotions")

    sprites = data.get("sprites", {})
    if sprites.get("static") and tier_gte(declared_tier, "t1"):
        stages.add("sprites_static")
    if sprites.get("animated") and tier_gte(declared_tier, "t1.5"):
        stages.add("sprites_animated")

    if data.get("trellis") and tier_gte(declared_tier, "t2"):
        stages.add("trellis")
    if data.get("outfit") and tier_gte(declared_tier, "t1"):
        stages.add("outfit")
    if data.get("state") and tier_gte(declared_tier, "t1"):
        stages.add("state")
    if data.get("video") and tier_gte(declared_tier, "t1.5"):
        stages.add("video")
    if data.get("godot") and tier_gte(declared_tier, "t1"):
        stages.add("godot")
    if data.get("lora") and tier_gte(declared_tier, "t3"):
        stages.add("lora")

    return stages
