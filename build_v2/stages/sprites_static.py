"""Sprites Static stage — VNCCS Step 4 SpriteCreator for static sprite sheet."""

from __future__ import annotations

from pathlib import Path

from build_v2.stages.base import Stage, StageResult


class SpritesStaticStage(Stage):
    """Generate static sprite sheet from character via VNCCS SpriteCreator.

    Body reference strategy (packages.sprites.body_reference):
      - "anny" (default): Uses the existing VNCCS Step 4 workflow with Anny body mesh.
      - "trellis": Uses TRELLIS-generated 3D model for pose reference.
        Auto-falls back to "anny" if TRELLIS stage did not complete.

    Uses VNCCS Step 4 API workflow. Takes the VNCCS character name
    and produces a full sprite sheet (all directions in one image).
    """

    name = "sprites_static"
    tier_required = "t1"
    depends_on = ["sheet"]

    _SPRITE_NODE = "1"
    _SAVE_NODE = "5"

    # Remote deployment overrides (see build_v2/api_patcher.py for why)
    _CHECKPOINT = "sd_xl_base_1.0.safetensors"  # ILFlatMix fails mmap on Docker
    _RMBG_MODEL = "BEN2"  # RMBG-2.0 needs writeable cache, INSPYRENET needs extra pkg

    def is_complete(self, state: dict) -> bool:
        static_dir = self.ctx.sprites_dir / "static"
        return static_dir.exists() and bool(list(static_dir.glob("*.png")))

    # ── Strategy resolution ────────────────────────────────────────────────

    def _resolve_body_reference(self) -> str:
        """Read body_reference strategy from YAML, default to anny."""
        sprites = self.ctx.character_data.get("packages", {}).get("sprites", {})
        return sprites.get("body_reference", "anny")

    # ── Run dispatch ───────────────────────────────────────────────────────

    def run(self, state: dict) -> StageResult:
        strategy = self._resolve_body_reference()

        # Auto-fallback: trellis → anny if trellis stage didn't complete
        if strategy == "trellis":
            trellis_state = state.get("stages", {}).get("trellis", {})
            if trellis_state.get("status") != "complete":
                print("  [WARN] TRELLIS not complete, falling back to Anny body mesh")
                strategy = "anny"

        if strategy == "trellis":
            return self._run_with_trellis(state)
        else:
            return self._run_with_anny(state)

    # ── Anny strategy (current behavior) ───────────────────────────────────

    def _run_with_anny(self, state: dict) -> StageResult:
        ctx = self.ctx
        sprites = ctx.character_data.get("packages", {}).get("sprites", {})
        outfit = sprites.get("outfit", "default")

        # VNCCS bug: CharacterCreator ignores new_character_name, uses existing_character.
        # When existing_character="None", the character is registered as "None".
        vnccs_char_name = "None"

        static_dir = ctx.sprites_dir / "static"
        static_dir.mkdir(parents=True, exist_ok=True)

        patches = {
            self._SPRITE_NODE: {
                "character": vnccs_char_name,
            },
            # filename_prefix linked to SpriteGenerator output (is_list=True → list)
            # causes SaveImage to crash with "expected str, got list"
            self._SAVE_NODE: {
                "filename_prefix": f"PoseDirector/{ctx.character_name}/sprites_static",
            },
        }

        try:
            images = self.executor.run_api_workflow(
                workflow_path=ctx.api_workflows_dir / "vnccs_step4_sprites.json",
                patches=patches,
                output_node_id=self._SAVE_NODE,
                timeout=1800,  # Can be slow on first run
                checkpoint_override=self._CHECKPOINT,
                rmbg_override=self._RMBG_MODEL,
            )
        except Exception as e:
            return StageResult(status="failed", error=f"Sprite generation failed: {e}")

        if not images:
            return StageResult(status="failed", error="Sprites produced no output")

        artifacts = []
        for i, img_bytes in enumerate(images):
            out_path = static_dir / f"sprite_{outfit}_{i:03d}.png"
            out_path.write_bytes(img_bytes)
            artifacts.append(str(out_path.relative_to(ctx.output_dir)))

        return StageResult(
            status="complete",
            artifacts=artifacts,
            meta={"outfit": outfit, "body_reference": "anny", "source": "vnccs_step4_sprites"},
        )

    # ── TRELLIS strategy (stub) ────────────────────────────────────────────

    def _run_with_trellis(self, state: dict) -> StageResult:
        """Use TRELLIS-generated 3D model for sprite pose reference.

        TODO: Implement GLB loading + multi-angle rendering pipeline.
        Requires a GLB renderer that produces 2D images at specified angles.
        """
        raise NotImplementedError(
            "TRELLIS body reference strategy not yet implemented. "
            "Use body_reference: anny (default) for now."
        )
