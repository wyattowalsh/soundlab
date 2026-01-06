"""Audio effects configuration models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass


class EffectConfig(BaseModel, ABC):
    """Base class for effect configurations."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True

    @abstractmethod
    def to_plugin(self) -> Any:
        """Convert to a Pedalboard plugin instance."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Effect name for display."""
        ...


# === Dynamics Effects ===

class CompressorConfig(EffectConfig):
    """Compressor effect configuration."""

    threshold_db: Annotated[float, Field(ge=-60.0, le=0.0)] = -20.0
    ratio: Annotated[float, Field(ge=1.0, le=20.0)] = 4.0
    attack_ms: Annotated[float, Field(ge=0.1, le=100.0)] = 10.0
    release_ms: Annotated[float, Field(ge=10.0, le=1000.0)] = 100.0

    @property
    def name(self) -> str:
        return "Compressor"

    def to_plugin(self) -> Any:
        from pedalboard import Compressor
        return Compressor(
            threshold_db=self.threshold_db,
            ratio=self.ratio,
            attack_ms=self.attack_ms,
            release_ms=self.release_ms,
        )


class LimiterConfig(EffectConfig):
    """Limiter effect configuration."""

    threshold_db: Annotated[float, Field(ge=-60.0, le=0.0)] = -1.0
    release_ms: Annotated[float, Field(ge=10.0, le=1000.0)] = 100.0

    @property
    def name(self) -> str:
        return "Limiter"

    def to_plugin(self) -> Any:
        from pedalboard import Limiter
        return Limiter(
            threshold_db=self.threshold_db,
            release_ms=self.release_ms,
        )


class GateConfig(EffectConfig):
    """Noise gate effect configuration."""

    threshold_db: Annotated[float, Field(ge=-80.0, le=0.0)] = -40.0
    ratio: Annotated[float, Field(ge=1.0, le=20.0)] = 10.0
    attack_ms: Annotated[float, Field(ge=0.1, le=100.0)] = 1.0
    release_ms: Annotated[float, Field(ge=10.0, le=1000.0)] = 100.0

    @property
    def name(self) -> str:
        return "NoiseGate"

    def to_plugin(self) -> Any:
        from pedalboard import NoiseGate
        return NoiseGate(
            threshold_db=self.threshold_db,
            ratio=self.ratio,
            attack_ms=self.attack_ms,
            release_ms=self.release_ms,
        )


class GainConfig(EffectConfig):
    """Gain effect configuration."""

    gain_db: Annotated[float, Field(ge=-60.0, le=60.0)] = 0.0

    @property
    def name(self) -> str:
        return "Gain"

    def to_plugin(self) -> Any:
        from pedalboard import Gain
        return Gain(gain_db=self.gain_db)


# === EQ Effects ===

class HighpassConfig(EffectConfig):
    """Highpass filter configuration."""

    cutoff_hz: Annotated[float, Field(ge=20.0, le=20000.0)] = 80.0

    @property
    def name(self) -> str:
        return "Highpass"

    def to_plugin(self) -> Any:
        from pedalboard import HighpassFilter
        return HighpassFilter(cutoff_frequency_hz=self.cutoff_hz)


class LowpassConfig(EffectConfig):
    """Lowpass filter configuration."""

    cutoff_hz: Annotated[float, Field(ge=20.0, le=20000.0)] = 15000.0

    @property
    def name(self) -> str:
        return "Lowpass"

    def to_plugin(self) -> Any:
        from pedalboard import LowpassFilter
        return LowpassFilter(cutoff_frequency_hz=self.cutoff_hz)


class HighShelfConfig(EffectConfig):
    """High shelf filter configuration."""

    cutoff_hz: Annotated[float, Field(ge=20.0, le=20000.0)] = 4000.0
    gain_db: Annotated[float, Field(ge=-24.0, le=24.0)] = 0.0
    q: Annotated[float, Field(ge=0.1, le=10.0)] = 0.707

    @property
    def name(self) -> str:
        return "HighShelf"

    def to_plugin(self) -> Any:
        from pedalboard import HighShelfFilter
        return HighShelfFilter(
            cutoff_frequency_hz=self.cutoff_hz,
            gain_db=self.gain_db,
            q=self.q,
        )


class LowShelfConfig(EffectConfig):
    """Low shelf filter configuration."""

    cutoff_hz: Annotated[float, Field(ge=20.0, le=20000.0)] = 200.0
    gain_db: Annotated[float, Field(ge=-24.0, le=24.0)] = 0.0
    q: Annotated[float, Field(ge=0.1, le=10.0)] = 0.707

    @property
    def name(self) -> str:
        return "LowShelf"

    def to_plugin(self) -> Any:
        from pedalboard import LowShelfFilter
        return LowShelfFilter(
            cutoff_frequency_hz=self.cutoff_hz,
            gain_db=self.gain_db,
            q=self.q,
        )


class PeakFilterConfig(EffectConfig):
    """Parametric EQ peak filter configuration."""

    cutoff_hz: Annotated[float, Field(ge=20.0, le=20000.0)] = 1000.0
    gain_db: Annotated[float, Field(ge=-24.0, le=24.0)] = 0.0
    q: Annotated[float, Field(ge=0.1, le=10.0)] = 1.0

    @property
    def name(self) -> str:
        return "PeakFilter"

    def to_plugin(self) -> Any:
        from pedalboard import PeakFilter
        return PeakFilter(
            cutoff_frequency_hz=self.cutoff_hz,
            gain_db=self.gain_db,
            q=self.q,
        )


# === Time-based Effects ===

class ReverbConfig(EffectConfig):
    """Reverb effect configuration."""

    room_size: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    damping: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    wet_level: Annotated[float, Field(ge=0.0, le=1.0)] = 0.33
    dry_level: Annotated[float, Field(ge=0.0, le=1.0)] = 0.4
    width: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    @property
    def name(self) -> str:
        return "Reverb"

    def to_plugin(self) -> Any:
        from pedalboard import Reverb
        return Reverb(
            room_size=self.room_size,
            damping=self.damping,
            wet_level=self.wet_level,
            dry_level=self.dry_level,
            width=self.width,
        )


class DelayConfig(EffectConfig):
    """Delay effect configuration."""

    delay_seconds: Annotated[float, Field(ge=0.0, le=2.0)] = 0.3
    feedback: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3
    mix: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5

    @property
    def name(self) -> str:
        return "Delay"

    def to_plugin(self) -> Any:
        from pedalboard import Delay
        return Delay(
            delay_seconds=self.delay_seconds,
            feedback=self.feedback,
            mix=self.mix,
        )


class ChorusConfig(EffectConfig):
    """Chorus effect configuration."""

    rate_hz: Annotated[float, Field(ge=0.1, le=10.0)] = 1.0
    depth: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    centre_delay_ms: Annotated[float, Field(ge=1.0, le=30.0)] = 7.0
    feedback: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    mix: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5

    @property
    def name(self) -> str:
        return "Chorus"

    def to_plugin(self) -> Any:
        from pedalboard import Chorus
        return Chorus(
            rate_hz=self.rate_hz,
            depth=self.depth,
            centre_delay_ms=self.centre_delay_ms,
            feedback=self.feedback,
            mix=self.mix,
        )


class PhaserConfig(EffectConfig):
    """Phaser effect configuration."""

    rate_hz: Annotated[float, Field(ge=0.1, le=10.0)] = 1.0
    depth: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    centre_frequency_hz: Annotated[float, Field(ge=100.0, le=5000.0)] = 1300.0
    feedback: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    mix: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5

    @property
    def name(self) -> str:
        return "Phaser"

    def to_plugin(self) -> Any:
        from pedalboard import Phaser
        return Phaser(
            rate_hz=self.rate_hz,
            depth=self.depth,
            centre_frequency_hz=self.centre_frequency_hz,
            feedback=self.feedback,
            mix=self.mix,
        )


# === Creative Effects ===

class DistortionConfig(EffectConfig):
    """Distortion effect configuration."""

    drive_db: Annotated[float, Field(ge=0.0, le=60.0)] = 25.0

    @property
    def name(self) -> str:
        return "Distortion"

    def to_plugin(self) -> Any:
        from pedalboard import Distortion
        return Distortion(drive_db=self.drive_db)


class ClippingConfig(EffectConfig):
    """Clipping effect configuration."""

    threshold_db: Annotated[float, Field(ge=-60.0, le=0.0)] = -6.0

    @property
    def name(self) -> str:
        return "Clipping"

    def to_plugin(self) -> Any:
        from pedalboard import Clipping
        return Clipping(threshold_db=self.threshold_db)


__all__ = [
    "EffectConfig",
    # Dynamics
    "CompressorConfig",
    "LimiterConfig",
    "GateConfig",
    "GainConfig",
    # EQ
    "HighpassConfig",
    "LowpassConfig",
    "HighShelfConfig",
    "LowShelfConfig",
    "PeakFilterConfig",
    # Time-based
    "ReverbConfig",
    "DelayConfig",
    "ChorusConfig",
    "PhaserConfig",
    # Creative
    "DistortionConfig",
    "ClippingConfig",
]
