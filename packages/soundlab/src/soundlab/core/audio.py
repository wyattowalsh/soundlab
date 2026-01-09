"""Core audio data models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import Annotated

import numpy as np
from numpy.typing import NDArray  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AudioFormat(StrEnum):
    """Supported audio formats."""

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    AIFF = "aiff"
    M4A = "m4a"


class SampleRate(StrEnum):
    """Common sample rates."""

    SR_22050 = "22050"
    SR_44100 = "44100"
    SR_48000 = "48000"
    SR_96000 = "96000"

    @property
    def hz(self) -> int:
        return int(self.value)


class BitDepth(StrEnum):
    """Audio bit depths."""

    INT16 = "16"
    INT24 = "24"
    FLOAT32 = "32"


class AudioMetadata(BaseModel):
    """Metadata for an audio file."""

    model_config = ConfigDict(frozen=True)

    duration_seconds: Annotated[float, Field(ge=0)]
    sample_rate: int
    channels: Annotated[int, Field(ge=1, le=8)]
    bit_depth: BitDepth | None = None
    format: AudioFormat | None = None

    @property
    def duration_str(self) -> str:
        """Human-readable duration (MM:SS.ms)."""
        mins, secs = divmod(self.duration_seconds, 60)
        return f"{int(mins):02d}:{secs:05.2f}"

    @property
    def is_stereo(self) -> bool:
        return self.channels == 2

    @property
    def is_mono(self) -> bool:
        return self.channels == 1


class AudioSegment(BaseModel):
    """In-memory audio representation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    samples: NDArray[np.float32]
    sample_rate: int
    source_path: Path | None = None
    metadata: AudioMetadata | None = None

    @field_validator("samples", mode="before")
    @classmethod
    def ensure_float32(cls, v: NDArray) -> NDArray[np.float32]:
        array = np.asarray(v)
        if array.dtype != np.float32:
            return array.astype(np.float32)
        return array

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def channels(self) -> int:
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[0]

    def to_mono(self) -> AudioSegment:
        """Convert to mono by averaging channels."""
        if self.channels == 1:
            return self
        mono_samples = np.mean(self.samples, axis=0)
        return AudioSegment(
            samples=mono_samples,
            sample_rate=self.sample_rate,
            source_path=self.source_path,
        )
