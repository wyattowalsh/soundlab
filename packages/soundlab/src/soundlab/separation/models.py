"""Stem separation configuration and result models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import Annotated

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from soundlab.core.exceptions import ProcessingError


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


def stem_audio_subtype(*, float32: bool, int24: bool) -> str:
    """Choose a soundfile subtype. An explicit float32 request wins over int24."""
    if float32:
        return "FLOAT"
    if int24:
        return "PCM_24"
    return "PCM_16"


def output_stem_names(
    source_names: list[str],
    source_count: int,
    two_stems: str | None,
) -> tuple[list[str], int | None]:
    """Name Demucs tensors from ``model.sources`` order.

    Returns the output stem names and, when collapsing to two stems, the
    index of the requested source. The index is into the original source list.
    """
    if len(source_names) != source_count:
        raise ProcessingError(
            f"Demucs returned {source_count} sources but model.sources lists "
            f"{len(source_names)} names"
        )
    if not two_stems:
        return list(source_names), None
    if two_stems not in source_names:
        raise ProcessingError(f"Invalid two_stems target: {two_stems}")
    return [two_stems, f"no_{two_stems}"], source_names.index(two_stems)


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
        """Return an existing instrumental or vocals-only complement path.

        This property does not write audio. Call ``export_instrumental`` to
        sum the non-vocal stems into a new file.
        """
        if self._instrumental_path is not None:
            return self._instrumental_path
        for name in ("no_vocals", "instrumental"):
            path = self.stems.get(name)
            if path is not None:
                return path
        if not self.stems:
            return None
        candidate = next(iter(self.stems.values())).parent / "instrumental.wav"
        if candidate.exists():
            return candidate
        return None

    def export_instrumental(self) -> Path | None:
        """Write a summed instrumental mix and return its path.

        Vocals-only runs already store the complement as ``no_vocals`` and
        that path is returned without a second write.
        """
        if self._instrumental_path is not None:
            return self._instrumental_path
        complement = self.stems.get("no_vocals")
        if complement is not None:
            object.__setattr__(self, "_instrumental_path", complement)
            return complement

        instrumental_stems = ["drums", "bass", "other", "piano", "guitar"]
        available = [name for name in instrumental_stems if name in self.stems]
        if not available:
            return None

        output_dir = next(iter(self.stems.values())).parent
        combined: np.ndarray | None = None
        sample_rate: int | None = None
        for stem_name in available:
            stem_path = self.stems[stem_name]
            if not stem_path.exists():
                continue
            audio_data, sr = sf.read(stem_path)
            if combined is None:
                combined = audio_data
                sample_rate = sr
            else:
                length = min(len(combined), len(audio_data))
                combined = combined[:length] + audio_data[:length]

        if combined is None or sample_rate is None:
            return None

        instrumental_path = output_dir / "instrumental.wav"
        sf.write(instrumental_path, combined, sample_rate)
        object.__setattr__(self, "_instrumental_path", instrumental_path)
        return instrumental_path
