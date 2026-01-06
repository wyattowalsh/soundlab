"""Audio file I/O operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from loguru import logger

from soundlab.core.audio import AudioFormat, AudioMetadata, AudioSegment, BitDepth
from soundlab.core.exceptions import AudioFormatError, AudioLoadError
from soundlab.core.types import PathLike
from soundlab.utils.retry import io_retry

if TYPE_CHECKING:
    pass


__all__ = ["load_audio", "save_audio", "get_audio_metadata"]


def _detect_format(path: Path) -> AudioFormat | None:
    """Detect audio format from file extension."""
    ext = path.suffix.lower().lstrip(".")
    try:
        return AudioFormat(ext)
    except ValueError:
        return None


def _get_bit_depth(subtype: str) -> BitDepth | None:
    """Map soundfile subtype to BitDepth."""
    if "24" in subtype:
        return BitDepth.INT24
    elif "32" in subtype or "FLOAT" in subtype:
        return BitDepth.FLOAT32
    elif "16" in subtype:
        return BitDepth.INT16
    return None


@io_retry
def load_audio(
    path: PathLike,
    *,
    target_sr: int | None = None,
    mono: bool = False,
) -> AudioSegment:
    """
    Load an audio file into an AudioSegment.

    Parameters
    ----------
    path
        Path to the audio file.
    target_sr
        Target sample rate. If provided, audio will be resampled.
    mono
        If True, convert to mono.

    Returns
    -------
    AudioSegment
        Loaded audio data.

    Raises
    ------
    AudioLoadError
        If the file cannot be loaded.
    AudioFormatError
        If the format is not supported.
    """
    path = Path(path)

    if not path.exists():
        raise AudioLoadError(f"File not found: {path}")

    audio_format = _detect_format(path)
    if audio_format is None:
        raise AudioFormatError(f"Unsupported format: {path.suffix}")

    logger.debug(f"Loading audio: {path}")

    try:
        # Load with soundfile
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)

        # Get file info for metadata
        info = sf.info(path)

        # Transpose to (channels, samples)
        samples = samples.T

        # Convert to mono if requested
        if mono and samples.shape[0] > 1:
            samples = np.mean(samples, axis=0, keepdims=False)
        elif samples.shape[0] == 1:
            samples = samples[0]  # Remove channel dimension for mono

        # Resample if target_sr specified
        if target_sr is not None and target_sr != sample_rate:
            import librosa
            samples = librosa.resample(
                samples,
                orig_sr=sample_rate,
                target_sr=target_sr,
            )
            sample_rate = target_sr

        # Build metadata
        metadata = AudioMetadata(
            duration_seconds=len(samples) / sample_rate if samples.ndim == 1 else samples.shape[1] / sample_rate,
            sample_rate=sample_rate,
            channels=1 if samples.ndim == 1 else samples.shape[0],
            bit_depth=_get_bit_depth(info.subtype),
            format=audio_format,
        )

        return AudioSegment(
            samples=samples.astype(np.float32),
            sample_rate=sample_rate,
            source_path=path,
            metadata=metadata,
        )

    except sf.SoundFileError as e:
        # Try pydub fallback for formats like MP3
        try:
            from pydub import AudioSegment as PydubSegment

            audio = PydubSegment.from_file(str(path))
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / (2 ** (audio.sample_width * 8 - 1))  # Normalize

            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).T

            if mono and len(samples.shape) > 1:
                samples = np.mean(samples, axis=0)

            sample_rate = audio.frame_rate

            if target_sr is not None and target_sr != sample_rate:
                import librosa
                samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
                sample_rate = target_sr

            metadata = AudioMetadata(
                duration_seconds=len(audio) / 1000.0,
                sample_rate=sample_rate,
                channels=1 if samples.ndim == 1 else samples.shape[0],
                format=audio_format,
            )

            return AudioSegment(
                samples=samples.astype(np.float32),
                sample_rate=sample_rate,
                source_path=path,
                metadata=metadata,
            )
        except Exception:
            raise AudioLoadError(f"Failed to load audio: {path}. Error: {e}") from e


@io_retry
def save_audio(
    segment: AudioSegment,
    path: PathLike,
    *,
    format: AudioFormat | None = None,
    bit_depth: BitDepth = BitDepth.INT24,
) -> Path:
    """
    Save an AudioSegment to file.

    Parameters
    ----------
    segment
        Audio segment to save.
    path
        Output path.
    format
        Output format. If None, detected from path extension.
    bit_depth
        Bit depth for output.

    Returns
    -------
    Path
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format is None:
        format = _detect_format(path) or AudioFormat.WAV

    # Prepare samples
    samples = segment.samples
    if samples.ndim == 2:
        samples = samples.T  # (channels, samples) -> (samples, channels)

    # Determine subtype
    subtype_map = {
        BitDepth.INT16: "PCM_16",
        BitDepth.INT24: "PCM_24",
        BitDepth.FLOAT32: "FLOAT",
    }
    subtype = subtype_map.get(bit_depth, "PCM_24")

    logger.debug(f"Saving audio: {path} ({format}, {bit_depth})")
    sf.write(path, samples, segment.sample_rate, subtype=subtype)

    return path


def get_audio_metadata(path: PathLike) -> AudioMetadata:
    """
    Get metadata for an audio file without loading the full audio.

    Parameters
    ----------
    path
        Path to audio file.

    Returns
    -------
    AudioMetadata
        File metadata.
    """
    path = Path(path)

    if not path.exists():
        raise AudioLoadError(f"File not found: {path}")

    info = sf.info(path)

    return AudioMetadata(
        duration_seconds=info.duration,
        sample_rate=info.samplerate,
        channels=info.channels,
        bit_depth=_get_bit_depth(info.subtype),
        format=_detect_format(path),
    )
