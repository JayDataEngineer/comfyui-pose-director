"""BuildContext — shared state for all stages in a build run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# VNCCS bug: CharacterCreator ignores new_character_name parameter.
# It always uses existing_character as the character name.
# With existing_character="None", character is registered as "None".
# All VNCCS stages must use this constant for character lookup.
VNCCS_CHAR_NAME = "None"

# Remote deployment overrides — models that work on Docker/remote ComfyUI.
# ILFlatMix fails mmap on Docker, INSPYRENET needs extra package.
DEPLOYMENT_CHECKPOINT = "sd_xl_base_1.0.safetensors"
DEPLOYMENT_RMBG_MODEL = "BEN2"


@dataclass
class BuildContext:
    """Carries everything a stage needs — parsed YAML, paths, config.

    Passed once at stage construction, not threaded through every call.
    """

    # Character data
    character_data: dict = field(default_factory=dict)
    character_name: str = ""
    tier: str = "t1"

    # Paths
    output_dir: Path = Path("output")  # output/{name}/
    workflows_dir: Path = Path("workflows")
    api_workflows_dir: Path = Path("workflows/api")
    motions_dir: Path = Path("motions")
    project_root: Path = Path(".")

    # Config
    seed: int = 42
    comfyui_url: str = "http://127.0.0.1:8465"
    dry_run: bool = False
    rebuild_targets: list[str] = field(default_factory=list)

    # Resolved at build time
    yaml_path: str = ""

    @classmethod
    def from_yaml(
        cls,
        data: dict,
        yaml_path: str,
        project_root: Path,
        comfyui_url: str = "http://127.0.0.1:8465",
        seed_override: int | None = None,
        dry_run: bool = False,
        rebuild_targets: list[str] | None = None,
        output_dir_override: Path | None = None,
    ) -> BuildContext:
        """Create a BuildContext from parsed YAML data."""
        char = data.get("character", {})
        name = char.get("name", "unknown")
        tier = char.get("tier", "t1")
        gen = data.get("generation", {})
        seed = seed_override if seed_override is not None else gen.get("seed", 42)

        return cls(
            character_data=data,
            character_name=name,
            tier=tier,
            output_dir=output_dir_override or (project_root / "output" / name),
            workflows_dir=project_root / "workflows",
            api_workflows_dir=project_root / "workflows" / "api",
            motions_dir=project_root / "motions",
            project_root=project_root,
            seed=seed,
            comfyui_url=comfyui_url,
            dry_run=dry_run,
            rebuild_targets=rebuild_targets or [],
            yaml_path=str(yaml_path),
        )

    # ── Package access helpers ─────────────────────────────────────────────

    def pkg(self, key: str) -> dict:
        """Get a package config from `packages.*`, falling back to top-level key."""
        pkgs = self.character_data.get("packages", {})
        result = pkgs.get(key)
        if result is not None:
            return result
        result = self.character_data.get(key)
        return result if isinstance(result, dict) else {}

    # ── Directory properties ────────────────────────────────────────────────

    @property
    def design_dir(self) -> Path:
        return self.output_dir / "design"

    @property
    def approved_image(self) -> Path:
        return self.design_dir / "approved.png"

    @property
    def sheet_dir(self) -> Path:
        return self.output_dir / "sheet"

    @property
    def emotions_dir(self) -> Path:
        return self.output_dir / "emotions"

    @property
    def sprites_dir(self) -> Path:
        return self.output_dir / "sprites"

    @property
    def reference_dir(self) -> Path:
        return self.output_dir / "reference"
