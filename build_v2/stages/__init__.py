"""Pipeline stages for the build system."""
from build_v2.stages.base import Stage, StageResult
from build_v2.stages.generate import GenerateStage
from build_v2.stages.sheet import SheetStage
from build_v2.stages.emotions import EmotionsStage
from build_v2.stages.sprites_static import SpritesStaticStage
from build_v2.stages.sprites_animated import SpritesAnimatedStage
from build_v2.stages.outfit import OutfitStage
from build_v2.stages.lora import LoraStage
from build_v2.stages.placeholders import (
    TrellisStage,
    StateStage,
    VideoStage,
    GodotStage,
)
