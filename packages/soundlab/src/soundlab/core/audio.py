"""Audio data models for SoundLab.

This module defines the core data structures for representing audio data,
metadata, and format specifications.
"""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator


__all__ = [
    "AudioFormat",
    "SampleRate",
    "BitDepth",
    "AudioMetadata",
    "AudioSegment",
]


class AudioFormat(StrEnum):
    """Supported audio file formats."""

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    AIFF = "aiff"
    M4A = "m4a"


class SampleRate(StrEnum):
    """Standard audio sample rates."""

    SR_22050 = "22050"
    SR_44100 = "44100"
    SR_48000 = "48000"
    SR_96000 = "96000"

    @property
    def hz(self) -> int:
        """Return the sample rate as an integer in Hz."""
        return int(self.value)


class BitDepth(StrEnum):
    """Audio bit depth formats."""

    INT16 = "int16"
    INT24 = "int24"
    FLOAT32 = "float32"


class AudioMetadata(BaseModel, frozen=True):
    """Metadata for audio files and segments.

    This model stores essential information about audio data including
    duration, sample rate, channel count, and format details.
    """

    duration_seconds: Annotated[float, Field(ge=0)]
    sample_rate: int
    channels: Annotated[int, Field(ge=1, le=8)]
    bit_depth: BitDepth | None = None
    format: AudioFormat | None = None

    @property
    def duration_str(self) -> str:
        """Return duration formatted as MM:SS.mmm."""
        minutes = int(self.duration_seconds // 60)
        seconds = self.duration_seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"

    @property
    def is_stereo(self) -> bool:
        """Return True if audio has exactly 2 channels."""
        return self.channels == 2

    @property
    def is_mono(self) -> bool:
        """Return True if audio has exactly 1 channel."""
        return self.channels == 1


class AudioSegment(BaseModel):
    """Represents an audio segment with samples and metadata.

    This is the core data structure for audio processing in SoundLab.
    Samples are stored as a numpy array with shape (channels, samples)
    or (samples,) for mono audio.
    """

    model_config = {"arbitrary_types_allowed": True}

    samples: NDArray[np.float32]
    sample_rate: int
    source_path: Path | None = None
    metadata: AudioMetadata | None = None

    @field_validator("samples")
    @classmethod
    def ensure_float32(cls, v: NDArray) -> NDArray[np.float32]:
        """Ensure samples are stored as float32."""
        if v.dtype != np.float32:
            return v.astype(np.float32)
        return v

    @property
    def duration_seconds(self) -> float:
        """Calculate duration in seconds from samples."""
        # Handle both mono (1D) and multi-channel (2D) arrays
        if self.samples.ndim == 1:
            num_samples = len(self.samples)
        else:
            num_samples = self.samples.shape[1]
        return num_samples / self.sample_rate

    @property
    def channels(self) -> int:
        """Return the number of audio channels."""
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[0]

    def to_mono(self) -> "AudioSegment":
        """Convert audio to mono by averaging all channels.

        Returns:
            A new AudioSegment with mono audio.
        """
        if self.channels == 1:
            # Already mono, return a copy
            return AudioSegment(
                samples=self.samples.copy(),
                sample_rate=self.sample_rate,
                source_path=self.source_path,
                metadata=self.metadata,
            )

        # Average across channels (axis 0)
        mono_samples = np.mean(self.samples, axis=0)

        # Update metadata if present
        new_metadata = None
        if self.metadata is not None:
            new_metadata = AudioMetadata(
                duration_seconds=self.metadata.duration_seconds,
                sample_rate=self.metadata.sample_rate,
                channels=1,
                bit_depth=self.metadata.bit_depth,
                format=self.metadata.format,
            )

        return AudioSegment(
            samples=mono_samples,
            sample_rate=self.sample_rate,
            source_path=self.source_path,
            metadata=new_metadata,
        )
