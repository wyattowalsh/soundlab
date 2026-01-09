"""Analysis result models."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


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
    """Tempo detection result."""

    model_config = ConfigDict(frozen=True)

    bpm: Annotated[float, Field(gt=0)]
    confidence: Annotated[float, Field(ge=0, le=1)]
    beats: list[float] = Field(default_factory=list)


class KeyDetectionResult(BaseModel):
    """Result of key detection analysis."""

    model_config = ConfigDict(frozen=True)

    key: MusicalKey
    mode: Mode
    confidence: Annotated[float, Field(ge=0, le=1)]
    all_correlations: dict[str, float] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        """Full key name (e.g., 'A minor')."""
        return f"{self.key.value} {self.mode.value}"

    @property
    def camelot(self) -> str:
        """Camelot notation for DJ mixing."""
        camelot_map = {
            ("C", "major"): "8B",
            ("A", "minor"): "8A",
            ("G", "major"): "9B",
            ("E", "minor"): "9A",
            ("D", "major"): "10B",
            ("B", "minor"): "10A",
            ("A", "major"): "11B",
            ("F#", "minor"): "11A",
            ("E", "major"): "12B",
            ("C#", "minor"): "12A",
            ("B", "major"): "1B",
            ("G#", "minor"): "1A",
            ("F#", "major"): "2B",
            ("D#", "minor"): "2A",
            ("C#", "major"): "3B",
            ("A#", "minor"): "3A",
            ("G#", "major"): "4B",
            ("F", "minor"): "4A",
            ("D#", "major"): "5B",
            ("C", "minor"): "5A",
            ("A#", "major"): "6B",
            ("G", "minor"): "6A",
            ("F", "major"): "7B",
            ("D", "minor"): "7A",
        }
        return camelot_map.get((self.key.value, self.mode.value), "?")


class LoudnessResult(BaseModel):
    """Loudness measurement result."""

    model_config = ConfigDict(frozen=True)

    lufs: float
    dynamic_range: float
    peak: float


class SpectralResult(BaseModel):
    """Spectral analysis result."""

    model_config = ConfigDict(frozen=True)

    centroid: float
    bandwidth: float
    rolloff: float


class AnalysisResult(BaseModel):
    """Composite analysis result."""

    model_config = ConfigDict(frozen=True)

    tempo: TempoResult | None = None
    key: KeyDetectionResult | None = None
    loudness: LoudnessResult | None = None
    spectral: SpectralResult | None = None


__all__ = [
    "AnalysisResult",
    "KeyDetectionResult",
    "LoudnessResult",
    "Mode",
    "MusicalKey",
    "SpectralResult",
    "TempoResult",
]
