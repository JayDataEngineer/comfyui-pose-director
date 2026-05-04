"""Emotions stage — VNCCS Step 3 EmotionStudio for face emotion variants."""

from __future__ import annotations

import json

from build_v2.stages.base import Stage, StageResult


class EmotionsStage(Stage):
    """Generate emotion portraits from character sheet via VNCCS EmotionStudio.

    Uses VNCCS Step 3 API workflow. Takes character name + emotions list,
    generates face close-ups for each declared emotion.
    """

    name = "emotions"
    tier_required = "t1"
    depends_on = ["sheet"]

    _EMOTION_NODE = "69"
    _SAVE_NODE = "9"

    def is_complete(self, state: dict) -> bool:
        emotions_dir = self.ctx.emotions_dir
        return emotions_dir.exists() and bool(list(emotions_dir.glob("*.png")))

    def run(self, state: dict) -> StageResult:
        ctx = self.ctx
        portraits = ctx.character_data.get("packages", {}).get("portraits", {})
        emotions = portraits.get("emotions", [])

        if not emotions:
            return StageResult(status="skipped", error="No emotions declared")

        emotions_dir = ctx.emotions_dir
        emotions_dir.mkdir(parents=True, exist_ok=True)

        outfit = portraits.get("outfit", "default")

        # VNCCS expects JSON-encoded string arrays for costumes/emotions
        # Costume name must match what CharacterCreator used ("Naked" is the default)
        costume_name = "Naked" if outfit == "default" else outfit
        # VNCCS bug: CharacterCreator ignores new_character_name, uses existing_character.
        # When existing_character="None", the character is registered as "None".
        vnccs_char_name = "None"  # Will be fixed when upstream patches new_character_name
        patches = {
            self._EMOTION_NODE: {
                "character": vnccs_char_name,
                "costumes_data": json.dumps([costume_name]),
                "emotions_data": json.dumps(emotions),
            },
            # filename_prefix linked to EmotionGeneratorV2 output (is_list=True → list)
            # causes SaveImage to crash with "expected str, got list"
            self._SAVE_NODE: {
                "filename_prefix": f"PoseDirector/{ctx.character_name}/emotions",
            },
        }

        try:
            images = self.executor.run_api_workflow(
                workflow_path=ctx.api_workflows_dir / "vnccs_step3_emotions.json",
                patches=patches,
                output_node_id=self._SAVE_NODE,
                timeout=2400,  # ~40 min for multiple emotions on remote GPU
            )
        except Exception as e:
            return StageResult(status="failed", error=f"Emotion generation failed: {e}")

        if not images:
            return StageResult(status="failed", error="Emotions produced no output")

        artifacts = []
        for i, img_bytes in enumerate(images):
            idx = min(i, len(emotions) - 1) if emotions else 0
            emotion_name = emotions[idx].replace(" ", "_")
            out_path = emotions_dir / f"{emotion_name}.png"
            out_path.write_bytes(img_bytes)
            artifacts.append(str(out_path.relative_to(ctx.output_dir)))

        return StageResult(
            status="complete",
            artifacts=artifacts,
            meta={
                "emotions": emotions,
                "outfit": outfit,
            },
        )
