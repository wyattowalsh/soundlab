"""Pydantic models for pedalboard effects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from soundlab.core.exceptions import ConfigurationError
from soundlab.effects.biquad import apply_rbj_filter

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from pedalboard import Plugin


def _build_plugin(factory: type[Any], **kwargs: Any) -> Plugin:
    """Create a pedalboard plugin and refuse parameters it cannot store."""
    try:
        return factory(**kwargs)
    except TypeError as exc:
        plugin = factory()
        missing: list[str] = []
        for key, value in kwargs.items():
            if not hasattr(plugin, key):
                missing.append(key)
                continue
            try:
                setattr(plugin, key, value)
            except Exception:
                missing.append(key)
                continue
            if getattr(plugin, key) != value and not _close(getattr(plugin, key), value):
                missing.append(key)
        if missing:
            names = ", ".join(missing)
            raise TypeError(f"{factory.__name__} did not accept: {names}") from exc
        return plugin


def _close(actual: Any, expected: Any) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= 1e-4
    except (TypeError, ValueError):
        return False


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
            wet_level=self.wet,
            dry_level=self.dry,
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

        if abs(self.q - 0.707) > 1e-6:
            raise ConfigurationError(
                "Pedalboard's high-pass filter has no Q control. "
                "Use EffectsChain, which applies this filter as a biquad."
            )
        return _build_plugin(
            HighpassFilter,
            cutoff_frequency_hz=self.cutoff_frequency_hz,
        )

    def process_array(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        """Apply the high-pass biquad, including Q."""
        return apply_rbj_filter(audio, sample_rate, self.cutoff_frequency_hz, self.q, "highpass")


class LowpassConfig(EffectConfig):
    """Low-pass filter."""

    cutoff_frequency_hz: float = Field(default=18000.0, gt=0.0)
    q: float = Field(default=0.707, gt=0.0)

    def to_plugin(self) -> Plugin:
        from pedalboard import LowpassFilter

        if abs(self.q - 0.707) > 1e-6:
            raise ConfigurationError(
                "Pedalboard's low-pass filter has no Q control. "
                "Use EffectsChain, which applies this filter as a biquad."
            )
        return _build_plugin(
            LowpassFilter,
            cutoff_frequency_hz=self.cutoff_frequency_hz,
        )

    def process_array(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        """Apply the low-pass biquad, including Q."""
        return apply_rbj_filter(audio, sample_rate, self.cutoff_frequency_hz, self.q, "lowpass")


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
