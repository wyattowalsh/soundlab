"""Tempo/BPM detection using librosa."""

from __future__ import annotations

from typing import TYPE_CHECKING

import librosa
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from soundlab.analysis.models import TempoResult

if TYPE_CHECKING:
    pass


__all__ = [
    "detect_tempo",
    "detect_tempo_with_alternatives",
]


def detect_tempo(
    y: NDArray[np.float32],
    sr: int,
    *,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    aggregate: str = "median",
) -> TempoResult:
    """
    Detect tempo/BPM from audio.

    Parameters
    ----------
    y
        Audio time series (mono).
    sr
        Sample rate.
    hop_length
        Hop length for beat tracking.
    start_bpm
        Initial tempo estimate.
    aggregate
        Aggregation method: "mean", "median"

    Returns
    -------
    TempoResult
        Detected tempo with confidence and beat positions.

    Examples
    --------
    >>> import librosa
    >>> y, sr = librosa.load("song.mp3", sr=22050, mono=True)
    >>> result = detect_tempo(y, sr)
    >>> print(f"BPM: {result.bpm:.1f}")
    """
    logger.debug(f"Detecting tempo (sr={sr}, hop={hop_length})")

    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
    )

    # Handle newer librosa versions that return an array
    if isinstance(tempo, np.ndarray):
        if len(tempo) > 0:
            if aggregate == "median":
                tempo_value = float(np.median(tempo))
            else:
                tempo_value = float(np.mean(tempo))
        else:
            tempo_value = 0.0
    else:
        tempo_value = float(tempo)

    # Convert beat frames to times
    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=hop_length,
    )

    # Calculate confidence based on beat regularity
    confidence = _calculate_tempo_confidence(beat_times, tempo_value)

    logger.debug(f"Detected tempo: {tempo_value:.1f} BPM (confidence: {confidence:.2f})")

    return TempoResult(
        bpm=tempo_value,
        confidence=confidence,
        beats=beat_times.tolist(),
    )


def _calculate_tempo_confidence(
    beat_times: NDArray[np.float64],
    tempo: float,
) -> float:
    """Calculate confidence score based on beat regularity."""
    if len(beat_times) < 3 or tempo <= 0:
        return 0.0

    # Expected interval between beats
    expected_interval = 60.0 / tempo

    # Calculate actual intervals
    intervals = np.diff(beat_times)

    if len(intervals) == 0:
        return 0.0

    # Calculate deviation from expected
    deviations = np.abs(intervals - expected_interval) / expected_interval

    # Mean deviation as confidence (lower is better)
    mean_deviation = np.mean(deviations)

    # Convert to 0-1 confidence score (1 = very regular beats)
    confidence = max(0.0, 1.0 - mean_deviation)

    return float(np.clip(confidence, 0.0, 1.0))


def detect_tempo_with_alternatives(
    y: NDArray[np.float32],
    sr: int,
    *,
    hop_length: int = 512,
) -> list[tuple[float, float]]:
    """
    Detect multiple possible tempos.

    Parameters
    ----------
    y
        Audio time series (mono).
    sr
        Sample rate.
    hop_length
        Hop length.

    Returns
    -------
    list[tuple[float, float]]
        List of (tempo, strength) tuples, sorted by strength.
    """
    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Compute onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Get tempo estimates with strengths
    # Use tempogram for multiple tempo hypotheses
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
    )

    # Get global tempo estimate
    tempo_frequencies = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=hop_length)

    # Average tempogram over time
    tempo_strengths = np.mean(tempogram, axis=1)

    # Find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(tempo_strengths, height=0.1, distance=5)

    if len(peaks) == 0:
        # Fallback to basic detection
        result = detect_tempo(y, sr, hop_length=hop_length)
        return [(result.bpm, result.confidence)]

    # Get tempo and strength for each peak
    results = []
    max_strength = np.max(tempo_strengths[peaks])

    for peak in peaks:
        tempo = tempo_frequencies[peak]
        if 30 <= tempo <= 300:  # Reasonable BPM range
            strength = tempo_strengths[peak] / max_strength
            results.append((float(tempo), float(strength)))

    # Sort by strength
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:5]  # Return top 5
