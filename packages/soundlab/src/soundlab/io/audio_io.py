"""Audio I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment as PydubAudioSegment

from soundlab.core.audio import AudioFormat, AudioMetadata, AudioSegment, BitDepth
from soundlab.core.exceptions import AudioFormatError, AudioLoadError


def _infer_format(path: Path, fmt: str | AudioFormat | None) -> AudioFormat | None:
    if isinstance(fmt, AudioFormat):
        return fmt

    raw = fmt or path.suffix.lstrip(".")
    if not raw:
        return None

    normalized = raw.strip().lower()
    if normalized == "aif":
        normalized = "aiff"

    try:
        return AudioFormat(normalized)
    except ValueError:
        return None


def _bit_depth_from_subtype(subtype: str | None) -> BitDepth | None:
    if not subtype:
        return None

    normalized = subtype.upper()
    if "PCM_16" in normalized:
        return BitDepth.INT16
    if "PCM_24" in normalized:
        return BitDepth.INT24
    if "PCM_32" in normalized or "FLOAT" in normalized:
        return BitDepth.FLOAT32
    return None


def _bit_depth_from_width(sample_width: int) -> BitDepth | None:
    if sample_width == 2:
        return BitDepth.INT16
    if sample_width == 3:
        return BitDepth.INT24
    if sample_width == 4:
        return BitDepth.FLOAT32
    return None


def _channels_first(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    if samples.shape[1] == 1:
        return samples[:, 0]
    return samples.T


def _channels_last(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    return samples.T


def _decode_with_pydub(path: Path) -> tuple[np.ndarray, int, AudioMetadata]:
    audio = PydubAudioSegment.from_file(path)
    samples = np.array(audio.get_array_of_samples())
    channels = audio.channels
    sample_rate = audio.frame_rate

    if channels > 1:
        frames = len(samples) // channels
        samples = samples[: frames * channels]
        samples = samples.reshape((frames, channels)).T
    else:
        samples = samples.astype(np.int16)

    sample_width = audio.sample_width
    max_value = float(1 << (8 * sample_width - 1)) if sample_width else 1.0
    samples = samples.astype(np.float32) / max_value

    duration_seconds = len(audio) / 1000.0
    metadata = AudioMetadata(
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=_bit_depth_from_width(sample_width),
        format=_infer_format(path, None),
    )

    return samples, sample_rate, metadata


def load_audio(path: str | Path) -> AudioSegment:
    """Load an audio file into an AudioSegment."""
    source = Path(path)
    try:
        info = sf.info(source)
        samples, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        samples = _channels_first(samples)
        metadata = AudioMetadata(
            duration_seconds=info.frames / info.samplerate,
            sample_rate=info.samplerate,
            channels=info.channels,
            bit_depth=_bit_depth_from_subtype(info.subtype),
            format=_infer_format(source, info.format),
        )
        return AudioSegment(
            samples=samples,
            sample_rate=sample_rate,
            source_path=source,
            metadata=metadata,
        )
    except Exception:
        try:
            samples, sample_rate, metadata = _decode_with_pydub(source)
            return AudioSegment(
                samples=samples,
                sample_rate=sample_rate,
                source_path=source,
                metadata=metadata,
            )
        except Exception as pydub_exc:
            raise AudioLoadError(f"Failed to load audio: {source}") from pydub_exc


def get_audio_metadata(path: str | Path) -> AudioMetadata:
    """Return metadata for an audio file without loading full samples."""
    source = Path(path)
    try:
        info = sf.info(source)
        return AudioMetadata(
            duration_seconds=info.frames / info.samplerate,
            sample_rate=info.samplerate,
            channels=info.channels,
            bit_depth=_bit_depth_from_subtype(info.subtype),
            format=_infer_format(source, info.format),
        )
    except Exception:
        audio = PydubAudioSegment.from_file(source)
        return AudioMetadata(
            duration_seconds=len(audio) / 1000.0,
            sample_rate=audio.frame_rate,
            channels=audio.channels,
            bit_depth=_bit_depth_from_width(audio.sample_width),
            format=_infer_format(source, None),
        )


def save_audio(
    segment: AudioSegment,
    path: str | Path,
    format: str | AudioFormat | None = None,
) -> None:
    """Save an AudioSegment to disk."""
    target = Path(path)
    audio_format = _infer_format(target, format)
    data = _channels_last(segment.samples)

    if audio_format is None:
        raise AudioFormatError(f"Unsupported audio format for {target}")

    try:
        sf.write(
            target,
            data,
            segment.sample_rate,
            format=audio_format.value.upper(),
        )
        return
    except Exception:
        pass

    samples = segment.samples
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]
    channels, frames = samples.shape
    clipped = np.clip(samples, -1.0, 1.0)
    int_samples = (clipped * 32767).astype(np.int16)
    interleaved = int_samples.T.reshape(frames * channels)

    pydub_segment = PydubAudioSegment(
        data=interleaved.tobytes(),
        sample_width=2,
        frame_rate=segment.sample_rate,
        channels=channels,
    )
    pydub_segment.export(target, format=audio_format.value)
