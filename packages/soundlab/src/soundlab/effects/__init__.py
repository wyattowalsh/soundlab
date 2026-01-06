"""Audio effects module for SoundLab."""

from soundlab.effects.chain import EffectsChain
from soundlab.effects.creative import ClippingConfig, DistortionConfig
from soundlab.effects.dynamics import (
    CompressorConfig,
    GainConfig,
    GateConfig,
    LimiterConfig,
)
from soundlab.effects.eq import (
    HighpassConfig,
    HighShelfConfig,
    LowpassConfig,
    LowShelfConfig,
    PeakFilterConfig,
)
from soundlab.effects.models import EffectConfig
from soundlab.effects.time_based import (
    ChorusConfig,
    DelayConfig,
    PhaserConfig,
    ReverbConfig,
)

__all__ = [
    # Chain
    "EffectsChain",
    # Base
    "EffectConfig",
    # Dynamics
    "CompressorConfig",
    "GainConfig",
    "GateConfig",
    "LimiterConfig",
    # EQ
    "HighpassConfig",
    "HighShelfConfig",
    "LowpassConfig",
    "LowShelfConfig",
    "PeakFilterConfig",
    # Time-based
    "ChorusConfig",
    "DelayConfig",
    "PhaserConfig",
    "ReverbConfig",
    # Creative
    "ClippingConfig",
    "DistortionConfig",
]
