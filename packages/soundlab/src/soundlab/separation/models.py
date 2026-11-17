"""Stem separation configuration and result models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import Annotated, Literal

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pydub import AudioSegment as PydubAudioSegment

from soundlab.core.exceptions import ConfigurationError, ProcessingError


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
        """Known stem names for this checkpoint family.

        The list is a catalog, not the tensor order. Separation names each
        output from the loaded model's ``sources``.
        """
        base = ["vocals", "drums", "bass", "other"]
        if self == DemucsModel.HTDEMUCS_6S:
            return [*base, "piano", "guitar"]
        return base


StemFormat = Literal["wav", "flac", "mp3"]


def write_stem_audio(
    samples: np.ndarray,
    path: Path,
    *,
    sample_rate: int,
    output_format: StemFormat = "wav",
    float32: bool = False,
    int24: bool = True,
    mp3_bitrate: int = 320,
) -> Path:
    """Write one stem and return the path that was written.

    ``samples`` are mono ``(frames,)`` or channels-last ``(frames, channels)``.
    ``float32`` selects a float WAV. FLAC and MP3 store integer samples, so a
    float32 request with either format raises ``ConfigurationError``. MP3 is
    encoded at ``mp3_bitrate`` kilobits per second.
    """
    target = path.with_suffix(f".{output_format}")
    if output_format == "wav":
        subtype = stem_audio_subtype(float32=float32, int24=int24)
        sf.write(target, samples, sample_rate, format="WAV", subtype=subtype)
        return target
    if float32:
        raise ConfigurationError(
            f"{output_format} cannot store float32 samples. Choose wav to keep the float stem."
        )
    if output_format == "flac":
        subtype = "PCM_24" if int24 else "PCM_16"
        sf.write(target, samples, sample_rate, format="FLAC", subtype=subtype)
        return target
    if output_format == "mp3":
        _write_mp3(samples, target, sample_rate, mp3_bitrate)
        return target
    raise ConfigurationError(f"Unsupported stem format: {output_format}")


def _write_mp3(samples: np.ndarray, target: Path, sample_rate: int, bitrate: int) -> None:
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = np.ascontiguousarray((clipped * 32767.0).astype(np.int16))
    channels = 1 if pcm.ndim == 1 else int(pcm.shape[1])
    segment = PydubAudioSegment(
        data=pcm.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=channels,
    )
    segment.export(target, format="mp3", bitrate=f"{int(bitrate)}k")


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
    output_format: StemFormat = "wav"

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
        """Return an existing instrumental mix, including vocals-only ``no_vocals``.

        Reading this property does not create or modify audio. Call
        ``export_instrumental`` to sum the non-vocal stems into a new file.
        """
        if self._instrumental_path is not None:
            return self._instrumental_path
        for name in ("no_vocals", "instrumental"):
            path = self.stems.get(name)
            if path is not None and path.is_file():
                return path
        if not self.stems:
            return None
        candidate = next(iter(self.stems.values())).parent / "instrumental.wav"
        if candidate.is_file():
            return candidate
        return None

    def export_instrumental(self) -> Path | None:
        """Write the non-vocal mix and return its path.

        Every stem except ``vocals`` and ``no_*`` complements is summed, in
        the result's stem order, so extra sources such as guitar are included
        when they are present. A vocals-only run already stores that mix as
        ``no_vocals`` and that file is returned without a second write.

        Stems that share the mix must have the same sample rate and shape.
        Mismatched audio raises ``ProcessingError`` instead of being shortened.
        The sum is stored as float WAV so peaks above full scale are kept.
        """
        if self._instrumental_path is not None:
            return self._instrumental_path

        complement = self.stems.get("no_vocals")
        if complement is not None and complement.is_file():
            object.__setattr__(self, "_instrumental_path", complement)
            return complement

        component_names = [
            name
            for name in self.stems
            if name not in {"vocals", "instrumental"} and not name.startswith("no_")
        ]
        if not component_names:
            return None

        combined: np.ndarray | None = None
        sample_rate: int | None = None
        reference_name: str | None = None
        for stem_name in component_names:
            stem_path = self.stems[stem_name]
            if not stem_path.is_file():
                continue
            audio_data, sr = sf.read(stem_path, always_2d=True)
            audio = np.array(audio_data, dtype=np.float64, copy=True)
            rate = int(sr)
            if combined is None or sample_rate is None or reference_name is None:
                combined = audio
                sample_rate = rate
                reference_name = stem_name
                continue
            if rate != sample_rate or audio.shape != combined.shape:
                raise ProcessingError(
                    f"Stem {stem_name} cannot be mixed with {reference_name} "
                    f"without clipping or truncation: expected {sample_rate} Hz "
                    f"and shape {combined.shape}, got {rate} Hz and shape {audio.shape}."
                )
            combined = combined + audio

        if combined is None or sample_rate is None:
            return None

        output_dir = next(iter(self.stems.values())).parent
        instrumental_path = output_dir / "instrumental.wav"
        sf.write(instrumental_path, combined, sample_rate, subtype="FLOAT")
        object.__setattr__(self, "_instrumental_path", instrumental_path)
        return instrumental_path
