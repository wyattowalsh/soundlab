"""Stem separation configuration and result models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import Annotated

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class DemucsModel(StrEnum):
    """Available Demucs models."""

    HTDEMUCS = "htdemucs"
    HTDEMUCS_FT = "htdemucs_ft"
    HTDEMUCS_6S = "htdemucs_6s"
    MDX_EXTRA = "mdx_extra"
    MDX_EXTRA_Q = "mdx_extra_q"

    @property
    def stem_count(self) -> int:
        return 6 if self == DemucsModel.HTDEMUCS_6S else 4

    @property
    def stems(self) -> list[str]:
        base = ["vocals", "drums", "bass", "other"]
        if self == DemucsModel.HTDEMUCS_6S:
            return [*base, "piano", "guitar"]
        return base


class SeparationConfig(BaseModel):
    """Configuration for stem separation."""

    model_config = ConfigDict(frozen=True)

    model: DemucsModel = DemucsModel.HTDEMUCS_FT

    segment_length: Annotated[float, Field(ge=1.0, le=30.0)] = 7.8
    overlap: Annotated[float, Field(ge=0.1, le=0.9)] = 0.25
    shifts: Annotated[int, Field(ge=0, le=5)] = 1

    two_stems: str | None = None
    float32: bool = False
    int24: bool = True
    mp3_bitrate: Annotated[int, Field(ge=128, le=320)] = 320

    device: str = "auto"
    split: bool = True


class StemResult(BaseModel):
    """Result from stem separation."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    stems: dict[str, Path]
    source_path: Path
    config: SeparationConfig
    processing_time_seconds: float

    _instrumental_path: Path | None = PrivateAttr(default=None)

    @property
    def vocals(self) -> Path | None:
        return self.stems.get("vocals")

    @property
    def instrumental(self) -> Path | None:
        """Combined non-vocal stems (computed on demand).

        Creates an instrumental track by summing drums, bass, and other stems.
        The result is cached and saved to {output_dir}/instrumental.wav.

        Returns:
            Path to the instrumental file, or None if required stems are missing.
        """
        # Return cached result if available
        if self._instrumental_path is not None:
            return self._instrumental_path

        # Identify non-vocal stems to combine
        instrumental_stems = ["drums", "bass", "other"]
        available_stems = [s for s in instrumental_stems if s in self.stems]

        # Need at least one stem to create instrumental
        if not available_stems:
            return None

        # Derive output directory from existing stem paths
        sample_stem_path = next(iter(self.stems.values()))
        output_dir = sample_stem_path.parent

        # Load and sum audio arrays
        combined: np.ndarray | None = None
        sample_rate: int | None = None

        for stem_name in available_stems:
            stem_path = self.stems[stem_name]
            if not stem_path.exists():
                continue

            audio_data, sr = sf.read(stem_path)

            if combined is None:
                combined = audio_data
                sample_rate = sr
            else:
                # Ensure arrays can be summed (handle length differences)
                min_len = min(len(combined), len(audio_data))
                combined = combined[:min_len] + audio_data[:min_len]

        # Check if we successfully loaded any audio
        if combined is None or sample_rate is None:
            return None

        # Save the instrumental mix
        instrumental_path = output_dir / "instrumental.wav"
        sf.write(instrumental_path, combined, sample_rate)

        # Cache and return the path
        object.__setattr__(self, "_instrumental_path", instrumental_path)
        return self._instrumental_path
