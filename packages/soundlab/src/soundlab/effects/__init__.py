"""Effects chain and configurations."""

from __future__ import annotations

from soundlab.effects.chain import AudioEffect, EffectsChain
from soundlab.effects.models import (
    ChorusConfig,
    CompressorConfig,
    DelayConfig,
    DistortionConfig,
    EffectConfig,
    GainConfig,
    GateConfig,
    HighpassConfig,
    LimiterConfig,
    LowpassConfig,
    PhaserConfig,
    ReverbConfig,
)

__all__ = [
    "AudioEffect",
    "ChorusConfig",
    "CompressorConfig",
    "DelayConfig",
    "DistortionConfig",
    "EffectConfig",
    "EffectsChain",
    "GainConfig",
    "GateConfig",
    "HighpassConfig",
    "LimiterConfig",
    "LowpassConfig",
    "PhaserConfig",
    "ReverbConfig",
]
