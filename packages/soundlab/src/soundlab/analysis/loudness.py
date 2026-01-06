"""Loudness analysis using pyloudnorm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from soundlab.analysis.models import LoudnessResult

if TYPE_CHECKING:
    pass


__all__ = [
    "measure_loudness",
    "normalize_loudness",
]


def measure_loudness(
    y: NDArray[np.float32],
    sr: int,
    *,
    block_size: float = 0.4,
) -> LoudnessResult:
    """
    Measure audio loudness according to ITU-R BS.1770-4.

    Parameters
    ----------
    y
        Audio time series.
    sr
        Sample rate.
    block_size
        Block size for momentary loudness in seconds.

    Returns
    -------
    LoudnessResult
        Loudness measurements including LUFS values.

    Examples
    --------
    >>> y, sr = librosa.load("song.mp3", sr=44100)
    >>> result = measure_loudness(y, sr)
    >>> print(f"Integrated: {result.integrated_lufs:.1f} LUFS")
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        logger.error("pyloudnorm is required for loudness measurement")
        raise ImportError("Install pyloudnorm: pip install pyloudnorm")

    logger.debug(f"Measuring loudness (sr={sr})")

    # Prepare audio for pyloudnorm (expects shape: samples x channels)
    if y.ndim == 1:
        # Mono - reshape to (samples, 1)
        audio = y.reshape(-1, 1)
    elif y.shape[0] <= 8:
        # (channels, samples) -> (samples, channels)
        audio = y.T
    else:
        # Already (samples, channels)
        audio = y

    # Create meter
    meter = pyln.Meter(sr, block_size=block_size)

    # Integrated loudness
    try:
        integrated_lufs = meter.integrated_loudness(audio)
    except Exception as e:
        logger.warning(f"Failed to measure integrated loudness: {e}")
        integrated_lufs = -70.0  # Silence

    # Handle -inf for silence
    if np.isinf(integrated_lufs):
        integrated_lufs = -70.0

    # True peak measurement
    true_peak_db = _measure_true_peak(audio)

    # Dynamic range (difference between peak and average RMS)
    dynamic_range = _measure_dynamic_range(y, sr)

    # Loudness range (LRA) approximation
    loudness_range = _measure_loudness_range(audio, meter)

    logger.debug(f"Loudness: {integrated_lufs:.1f} LUFS, Peak: {true_peak_db:.1f} dB")

    return LoudnessResult(
        integrated_lufs=integrated_lufs,
        true_peak_db=true_peak_db,
        dynamic_range_db=dynamic_range,
        loudness_range=loudness_range,
    )


def _measure_true_peak(audio: NDArray[np.float32]) -> float:
    """Measure true peak in dB."""
    peak = np.max(np.abs(audio))
    if peak <= 0:
        return -96.0
    return float(20 * np.log10(peak))


def _measure_dynamic_range(
    y: NDArray[np.float32],
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> float:
    """Measure dynamic range as difference between peak and median RMS."""
    import librosa

    # Ensure mono for RMS calculation
    if y.ndim > 1:
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y

    # Calculate RMS in frames
    rms = librosa.feature.rms(
        y=y_mono,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    # Filter out very quiet frames (likely silence)
    rms_nonzero = rms[rms > 0.001]

    if len(rms_nonzero) == 0:
        return 0.0

    # Dynamic range: peak to median ratio in dB
    peak_rms = np.max(rms_nonzero)
    median_rms = np.median(rms_nonzero)

    if median_rms <= 0:
        return 0.0

    return float(20 * np.log10(peak_rms / median_rms))


def _measure_loudness_range(
    audio: NDArray[np.float32],
    meter,
) -> float:
    """Approximate loudness range (LRA)."""
    # Simple approximation: segment audio and measure variance in loudness
    sr = meter.rate
    segment_duration = 3.0  # 3-second segments
    segment_samples = int(sr * segment_duration)

    n_samples = audio.shape[0]
    if n_samples < segment_samples:
        return 0.0

    n_segments = n_samples // segment_samples
    loudness_values = []

    for i in range(n_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = audio[start:end]

        try:
            lufs = meter.integrated_loudness(segment)
            if not np.isinf(lufs) and lufs > -70:
                loudness_values.append(lufs)
        except Exception:
            pass

    if len(loudness_values) < 2:
        return 0.0

    # LRA approximation: difference between 95th and 10th percentile
    loudness_values = np.array(loudness_values)
    high = np.percentile(loudness_values, 95)
    low = np.percentile(loudness_values, 10)

    return float(max(0, high - low))


def normalize_loudness(
    y: NDArray[np.float32],
    sr: int,
    target_lufs: float = -14.0,
) -> NDArray[np.float32]:
    """
    Normalize audio to target loudness.

    Parameters
    ----------
    y
        Audio time series.
    sr
        Sample rate.
    target_lufs
        Target integrated loudness in LUFS.

    Returns
    -------
    NDArray[np.float32]
        Normalized audio.
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        raise ImportError("pyloudnorm is required for loudness normalization")

    # Prepare audio
    if y.ndim == 1:
        audio = y.reshape(-1, 1)
    elif y.shape[0] <= 8:
        audio = y.T
    else:
        audio = y

    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(audio)

    if np.isinf(current_lufs):
        logger.warning("Cannot normalize silence")
        return y

    # Calculate gain
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)

    # Apply gain
    normalized = audio * gain_linear

    # Prevent clipping
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized = normalized / peak * 0.99
        logger.warning("Applied limiting to prevent clipping")

    # Return in original shape
    if y.ndim == 1:
        return normalized.flatten().astype(np.float32)
    elif y.shape[0] <= 8:
        return normalized.T.astype(np.float32)
    else:
        return normalized.astype(np.float32)
