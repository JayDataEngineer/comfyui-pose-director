"""Floor YAML adapter — extract character specs from floor manifests.

Floor YAMLs nest characters under `characters: {name: {spec}}`.
This module extracts individual character specs and wraps them in the
standalone format that BuildContext.from_yaml() expects.

Standalone format:
    {character: {name: ...}, generation: {...}, packages: {...}}

Floor format:
    characters:
      bartender:
        description: "..."
        generation: {...}
        packages: {...}
"""

from __future__ import annotations


def extract_character_from_floor(floor_data: dict, character_name: str) -> dict:
    """Extract a character spec from a floor YAML and wrap in standalone format."""
    chars = floor_data.get("characters", {})
    if character_name not in chars:
        raise ValueError(
            f"Character '{character_name}' not in floor. "
            f"Available: {list(chars.keys())}"
        )
    spec = chars[character_name]
    return {
        "character": {
            "name": character_name,
            "description": spec.get("description", ""),
        },
        "generation": spec.get("generation", {}),
        "packages": spec.get("packages", {}),
    }


def list_floor_characters(floor_data: dict) -> list[str]:
    """Return character names defined in a floor YAML."""
    return list(floor_data.get("characters", {}).keys())
