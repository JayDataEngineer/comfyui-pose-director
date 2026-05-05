"""Unit tests for floor_adapter — extracting character specs from floor YAMLs."""

import pytest

from build_v2.floor_adapter import extract_character_from_floor, list_floor_characters


# ── Test fixtures ──────────────────────────────────────────────────────

FLOOR_DATA = {
    "floor": {"name": "Club", "floor_number": 2},
    "characters": {
        "bartender": {
            "description": "cyberpunk bartender",
            "generation": {
                "source": "generate",
                "prompt": "A bartender mixing drinks",
                "quality": "turbo",
            },
            "packages": {
                "sprites": {
                    "outfit": "default",
                    "directions": ["front"],
                    "animated": [
                        {"name": "idle", "motion": "shared", "prefix": "bartender_idle"},
                    ],
                },
            },
        },
        "bouncer": {
            "description": "massive cyberpunk bouncer",
            "generation": {
                "source": "generate",
                "prompt": "A bouncer at a club door",
            },
            "packages": {
                "sprites": {
                    "static": True,
                    "directions": ["front", "back"],
                },
            },
        },
        "imported_npc": {
            "description": "An existing NPC",
            "generation": {"source": "import"},
        },
    },
}


# ── extract_character_from_floor ───────────────────────────────────────

class TestExtractCharacterFromFloor:

    def test_extracts_existing_character(self):
        result = extract_character_from_floor(FLOOR_DATA, "bartender")
        assert result["character"]["name"] == "bartender"
        assert result["character"]["description"] == "cyberpunk bartender"
        assert result["generation"]["source"] == "generate"
        assert result["generation"]["prompt"] == "A bartender mixing drinks"
        assert "sprites" in result["packages"]

    def test_extracts_different_character(self):
        result = extract_character_from_floor(FLOOR_DATA, "bouncer")
        assert result["character"]["name"] == "bouncer"
        assert result["generation"]["prompt"] == "A bouncer at a club door"
        assert result["packages"]["sprites"]["static"] is True

    def test_import_character(self):
        result = extract_character_from_floor(FLOOR_DATA, "imported_npc")
        assert result["character"]["name"] == "imported_npc"
        assert result["generation"]["source"] == "import"

    def test_missing_character_raises(self):
        with pytest.raises(ValueError, match="not_in_floor"):
            extract_character_from_floor(FLOOR_DATA, "not_in_floor")

    def test_missing_character_lists_available(self):
        with pytest.raises(ValueError, match="bartender") as exc_info:
            extract_character_from_floor(FLOOR_DATA, "nonexistent")
        # Error message should list available characters
        assert "bartender" in str(exc_info.value)
        assert "bouncer" in str(exc_info.value)

    def test_empty_characters_raises(self):
        with pytest.raises(ValueError, match="no_chars_here"):
            extract_character_from_floor({"characters": {}}, "no_chars_here")

    def test_no_characters_key_raises(self):
        with pytest.raises(ValueError, match="anything"):
            extract_character_from_floor({"floor": {"name": "X"}}, "anything")

    def test_description_defaults_to_empty(self):
        data = {"characters": {"minimal": {"generation": {"source": "generate", "prompt": "x"}}}}
        result = extract_character_from_floor(data, "minimal")
        assert result["character"]["description"] == ""

    def test_generation_defaults_to_empty_dict(self):
        data = {"characters": {"minimal": {"description": "no gen"}}}
        result = extract_character_from_floor(data, "minimal")
        assert result["generation"] == {}

    def test_packages_defaults_to_empty_dict(self):
        data = {"characters": {"minimal": {"generation": {"source": "generate", "prompt": "x"}}}}
        result = extract_character_from_floor(data, "minimal")
        assert result["packages"] == {}

    def test_output_format_matches_standalone(self):
        """The extracted format must be compatible with validate_character."""
        from build_v2.validators import validate_character
        from pathlib import Path

        result = extract_character_from_floor(FLOOR_DATA, "bartender")
        # Should not raise
        validate_character(result, Path("test_floor"))


# ── list_floor_characters ─────────────────────────────────────────────

class TestListFloorCharacters:

    def test_lists_all_characters(self):
        names = list_floor_characters(FLOOR_DATA)
        assert names == ["bartender", "bouncer", "imported_npc"]

    def test_empty_characters(self):
        assert list_floor_characters({"characters": {}}) == []

    def test_no_characters_key(self):
        assert list_floor_characters({}) == []

    def test_preserves_order(self):
        data = {"characters": {"z_char": {}, "a_char": {}, "m_char": {}}}
        names = list_floor_characters(data)
        assert names == ["z_char", "a_char", "m_char"]


# ── Integration: extraction → validation → BuildContext ────────────────

class TestFloorAdapterIntegration:

    def test_extract_and_validate_bartender(self):
        from build_v2.validators import validate_character
        from pathlib import Path

        char_data = extract_character_from_floor(FLOOR_DATA, "bartender")
        validate_character(char_data, Path("floor"))

        assert char_data["character"]["name"] == "bartender"
        assert char_data["generation"]["source"] == "generate"
        assert char_data["packages"]["sprites"]["animated"]

    def test_extract_and_validate_bouncer(self):
        from build_v2.validators import validate_character
        from pathlib import Path

        char_data = extract_character_from_floor(FLOOR_DATA, "bouncer")
        validate_character(char_data, Path("floor"))

        assert char_data["packages"]["sprites"]["static"] is True

    def test_import_character_validates(self):
        from build_v2.validators import validate_character
        from pathlib import Path

        # Import characters don't need a prompt
        char_data = extract_character_from_floor(FLOOR_DATA, "imported_npc")
        validate_character(char_data, Path("floor"))

    def test_extract_and_infer_tier(self):
        from build_v2.validators import infer_tier

        # Bartender has animated sprites → t1.5
        char_data = extract_character_from_floor(FLOOR_DATA, "bartender")
        tier = infer_tier(char_data)
        assert tier == "t1.5"

    def test_bouncer_tier(self):
        from build_v2.validators import infer_tier

        # Bouncer has static sprites only → t1
        char_data = extract_character_from_floor(FLOOR_DATA, "bouncer")
        tier = infer_tier(char_data)
        assert tier == "t1"

    def test_import_tier(self):
        from build_v2.validators import infer_tier

        # Import with no packages → t0
        char_data = extract_character_from_floor(FLOOR_DATA, "imported_npc")
        tier = infer_tier(char_data)
        assert tier == "t0"
