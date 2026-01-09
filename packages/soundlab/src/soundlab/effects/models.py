"""Pydantic models for pedalboard effects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from pedalboard import Plugin


def _build_plugin(factory: type[Any], **kwargs: Any) -> Plugin:
    """Create a pedalboard plugin, falling back to attribute assignment."""
    try:
        return factory(**kwargs)
    except TypeError:
        plugin = factory()
        for key, value in kwargs.items():
            if hasattr(plugin, key):
                with suppress(Exception):
                    setattr(plugin, key, value)
        return plugin


class EffectConfig(BaseModel, ABC):
    """Base config for an audio effect."""

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def to_plugin(self) -> Plugin:
        """Instantiate the pedalboard plugin."""


class CompressorConfig(EffectConfig):
    """Dynamic range compressor."""

    threshold_db: float = Field(default=-24.0)
    ratio: float = Field(default=2.0, ge=1.0)
    attack_ms: float = Field(default=10.0, ge=0.0)
    release_ms: float = Field(default=100.0, ge=0.0)

    def to_plugin(self) -> Plugin:
        from pedalboard import Compressor

        return _build_plugin(
            Compressor,
            threshold_db=self.threshold_db,
            ratio=self.ratio,
            attack_ms=self.attack_ms,
            release_ms=self.release_ms,
        )


class LimiterConfig(EffectConfig):
    """Peak limiter."""

    threshold_db: float = Field(default=-1.0)
    release_ms: float = Field(default=100.0, ge=0.0)

    def to_plugin(self) -> Plugin:
        from pedalboard import Limiter

        return _build_plugin(Limiter, threshold_db=self.threshold_db, release_ms=self.release_ms)


class GateConfig(EffectConfig):
    """Noise gate."""

    threshold_db: float = Field(default=-40.0)
    ratio: float = Field(default=2.0, ge=1.0)
    attack_ms: float = Field(default=10.0, ge=0.0)
    release_ms: float = Field(default=100.0, ge=0.0)

    def to_plugin(self) -> Plugin:
        from pedalboard import NoiseGate

        return _build_plugin(
            NoiseGate,
            threshold_db=self.threshold_db,
            ratio=self.ratio,
            attack_ms=self.attack_ms,
            release_ms=self.release_ms,
        )


class ReverbConfig(EffectConfig):
    """Room reverb."""

    room_size: float = Field(default=0.5, ge=0.0, le=1.0)
    damping: float = Field(default=0.5, ge=0.0, le=1.0)
    wet: float = Field(default=0.33, ge=0.0, le=1.0)
    dry: float = Field(default=0.67, ge=0.0, le=1.0)
    width: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_plugin(self) -> Plugin:
        from pedalboard import Reverb

        return _build_plugin(
            Reverb,
            room_size=self.room_size,
            damping=self.damping,
            wet=self.wet,
            dry=self.dry,
            width=self.width,
        )


class DelayConfig(EffectConfig):
    """Time delay effect."""

    delay_seconds: float = Field(default=0.25, ge=0.0)
    feedback: float = Field(default=0.5, ge=0.0, le=1.0)
    mix: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_plugin(self) -> Plugin:
        from pedalboard import Delay

        return _build_plugin(
            Delay,
            delay_seconds=self.delay_seconds,
            feedback=self.feedback,
            mix=self.mix,
        )


class ChorusConfig(EffectConfig):
    """Chorus modulation effect."""

    rate_hz: float = Field(default=1.0, ge=0.0)
    depth: float = Field(default=0.5, ge=0.0, le=1.0)
    centre_delay_ms: float = Field(default=7.0, ge=0.0)
    feedback: float = Field(default=0.0, ge=0.0, le=1.0)
    mix: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_plugin(self) -> Plugin:
        from pedalboard import Chorus

        return _build_plugin(
            Chorus,
            rate_hz=self.rate_hz,
            depth=self.depth,
            centre_delay_ms=self.centre_delay_ms,
            feedback=self.feedback,
            mix=self.mix,
        )


class DistortionConfig(EffectConfig):
    """Distortion effect."""

    drive_db: float = Field(default=20.0)

    def to_plugin(self) -> Plugin:
        from pedalboard import Distortion

        return _build_plugin(Distortion, drive_db=self.drive_db)


class PhaserConfig(EffectConfig):
    """Phaser modulation effect."""

    rate_hz: float = Field(default=1.0, ge=0.0)
    depth: float = Field(default=0.5, ge=0.0, le=1.0)
    centre_frequency_hz: float = Field(default=1300.0, ge=20.0)
    feedback: float = Field(default=0.0, ge=0.0, le=1.0)
    mix: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_plugin(self) -> Plugin:
        from pedalboard import Phaser

        return _build_plugin(
            Phaser,
            rate_hz=self.rate_hz,
            depth=self.depth,
            centre_frequency_hz=self.centre_frequency_hz,
            feedback=self.feedback,
            mix=self.mix,
        )


class HighpassConfig(EffectConfig):
    """High-pass filter."""

    cutoff_frequency_hz: float = Field(default=80.0, gt=0.0)
    q: float = Field(default=0.707, gt=0.0)

    def to_plugin(self) -> Plugin:
        from pedalboard import HighpassFilter

        return _build_plugin(
            HighpassFilter,
            cutoff_frequency_hz=self.cutoff_frequency_hz,
            q=self.q,
        )


class LowpassConfig(EffectConfig):
    """Low-pass filter."""

    cutoff_frequency_hz: float = Field(default=18000.0, gt=0.0)
    q: float = Field(default=0.707, gt=0.0)

    def to_plugin(self) -> Plugin:
        from pedalboard import LowpassFilter

        return _build_plugin(
            LowpassFilter,
            cutoff_frequency_hz=self.cutoff_frequency_hz,
            q=self.q,
        )


class GainConfig(EffectConfig):
    """Gain adjustment."""

    gain_db: float = Field(default=0.0)

    def to_plugin(self) -> Plugin:
        from pedalboard import Gain

        return _build_plugin(Gain, gain_db=self.gain_db)


__all__ = [
    "ChorusConfig",
    "CompressorConfig",
    "DelayConfig",
    "DistortionConfig",
    "EffectConfig",
    "GainConfig",
    "GateConfig",
    "HighpassConfig",
    "LimiterConfig",
    "LowpassConfig",
    "PhaserConfig",
    "ReverbConfig",
]
