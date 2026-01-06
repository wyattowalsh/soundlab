"""Audio analysis result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


__all__ = [
    "MusicalKey",
    "Mode",
    "TempoResult",
    "KeyDetectionResult",
    "LoudnessResult",
    "SpectralResult",
    "OnsetResult",
    "AnalysisResult",
]


class MusicalKey(StrEnum):
    """Musical key names."""

    C = "C"
    Cs = "C#"
    D = "D"
    Ds = "D#"
    E = "E"
    F = "F"
    Fs = "F#"
    G = "G"
    Gs = "G#"
    A = "A"
    As = "A#"
    B = "B"


class Mode(StrEnum):
    """Musical mode."""

    MAJOR = "major"
    MINOR = "minor"


class TempoResult(BaseModel):
    """Result from tempo/BPM detection."""

    model_config = ConfigDict(frozen=True)

    bpm: Annotated[float, Field(ge=0.0)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    beats: list[float] = Field(default_factory=list)  # Beat timestamps in seconds

    @property
    def beat_count(self) -> int:
        """Number of detected beats."""
        return len(self.beats)

    @property
    def beat_interval(self) -> float:
        """Average interval between beats in seconds."""
        if self.bpm <= 0:
            return 0.0
        return 60.0 / self.bpm


class KeyDetectionResult(BaseModel):
    """Result from musical key detection."""

    model_config = ConfigDict(frozen=True)

    key: MusicalKey
    mode: Mode
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    all_correlations: dict[str, float] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        """Full key name (e.g., 'A minor')."""
        return f"{self.key.value} {self.mode.value}"

    @property
    def camelot(self) -> str:
        """Camelot notation for DJ mixing."""
        camelot_map = {
            ("C", "major"): "8B", ("A", "minor"): "8A",
            ("G", "major"): "9B", ("E", "minor"): "9A",
            ("D", "major"): "10B", ("B", "minor"): "10A",
            ("A", "major"): "11B", ("F#", "minor"): "11A",
            ("E", "major"): "12B", ("C#", "minor"): "12A",
            ("B", "major"): "1B", ("G#", "minor"): "1A",
            ("F#", "major"): "2B", ("D#", "minor"): "2A",
            ("C#", "major"): "3B", ("A#", "minor"): "3A",
            ("G#", "major"): "4B", ("F", "minor"): "4A",
            ("D#", "major"): "5B", ("C", "minor"): "5A",
            ("A#", "major"): "6B", ("G", "minor"): "6A",
            ("F", "major"): "7B", ("D", "minor"): "7A",
        }
        return camelot_map.get((self.key.value, self.mode.value), "?")

    @property
    def open_key(self) -> str:
        """Open Key notation for DJ mixing."""
        open_key_map = {
            ("C", "major"): "1d", ("A", "minor"): "1m",
            ("G", "major"): "2d", ("E", "minor"): "2m",
            ("D", "major"): "3d", ("B", "minor"): "3m",
            ("A", "major"): "4d", ("F#", "minor"): "4m",
            ("E", "major"): "5d", ("C#", "minor"): "5m",
            ("B", "major"): "6d", ("G#", "minor"): "6m",
            ("F#", "major"): "7d", ("D#", "minor"): "7m",
            ("C#", "major"): "8d", ("A#", "minor"): "8m",
            ("G#", "major"): "9d", ("F", "minor"): "9m",
            ("D#", "major"): "10d", ("C", "minor"): "10m",
            ("A#", "major"): "11d", ("G", "minor"): "11m",
            ("F", "major"): "12d", ("D", "minor"): "12m",
        }
        return open_key_map.get((self.key.value, self.mode.value), "?")


class LoudnessResult(BaseModel):
    """Result from loudness analysis."""

    model_config = ConfigDict(frozen=True)

    integrated_lufs: Annotated[float, Field(le=0.0)]
    short_term_lufs: float | None = None
    momentary_lufs: float | None = None
    loudness_range: Annotated[float, Field(ge=0.0)] | None = None
    true_peak_db: float | None = None
    dynamic_range_db: Annotated[float, Field(ge=0.0)] | None = None

    @property
    def is_broadcast_safe(self) -> bool:
        """Check if loudness meets broadcast standards (-24 to -14 LUFS)."""
        return -24.0 <= self.integrated_lufs <= -14.0

    @property
    def is_streaming_optimized(self) -> bool:
        """Check if loudness is optimized for streaming (-14 LUFS)."""
        return -15.0 <= self.integrated_lufs <= -13.0


class SpectralResult(BaseModel):
    """Result from spectral analysis."""

    model_config = ConfigDict(frozen=True)

    spectral_centroid: float  # Hz
    spectral_bandwidth: float  # Hz
    spectral_rolloff: float  # Hz (95% energy)
    spectral_flatness: Annotated[float, Field(ge=0.0, le=1.0)]
    zero_crossing_rate: float

    @property
    def brightness(self) -> str:
        """Qualitative brightness assessment."""
        if self.spectral_centroid < 1500:
            return "dark"
        elif self.spectral_centroid < 3000:
            return "balanced"
        else:
            return "bright"


class OnsetResult(BaseModel):
    """Result from onset/transient detection."""

    model_config = ConfigDict(frozen=True)

    onset_times: list[float] = Field(default_factory=list)  # timestamps in seconds
    onset_strengths: list[float] = Field(default_factory=list)

    @property
    def onset_count(self) -> int:
        """Number of detected onsets."""
        return len(self.onset_times)

    @property
    def average_interval(self) -> float:
        """Average time between onsets."""
        if len(self.onset_times) < 2:
            return 0.0
        intervals = [
            self.onset_times[i + 1] - self.onset_times[i]
            for i in range(len(self.onset_times) - 1)
        ]
        return sum(intervals) / len(intervals)


class AnalysisResult(BaseModel):
    """Comprehensive audio analysis result."""

    model_config = ConfigDict(frozen=True)

    duration_seconds: float
    sample_rate: int
    channels: int

    tempo: TempoResult | None = None
    key: KeyDetectionResult | None = None
    loudness: LoudnessResult | None = None
    spectral: SpectralResult | None = None
    onsets: OnsetResult | None = None

    @property
    def summary(self) -> dict[str, str | float | None]:
        """Get a summary of all analysis results."""
        result: dict[str, str | float | None] = {
            "duration": f"{self.duration_seconds:.2f}s",
            "sample_rate": f"{self.sample_rate} Hz",
            "channels": self.channels,
        }

        if self.tempo:
            result["bpm"] = f"{self.tempo.bpm:.1f}"
        if self.key:
            result["key"] = self.key.name
            result["camelot"] = self.key.camelot
        if self.loudness:
            result["lufs"] = f"{self.loudness.integrated_lufs:.1f}"
        if self.spectral:
            result["brightness"] = self.spectral.brightness
        if self.onsets:
            result["onset_count"] = self.onsets.onset_count

        return result
