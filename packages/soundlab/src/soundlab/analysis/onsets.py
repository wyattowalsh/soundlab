"""Onset/transient detection using librosa."""

from __future__ import annotations

from typing import TYPE_CHECKING

import librosa
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from soundlab.analysis.models import OnsetResult

if TYPE_CHECKING:
    pass


__all__ = [
    "detect_onsets",
    "detect_beats_and_onsets",
    "get_onset_strength_envelope",
    "segment_by_onsets",
]


def detect_onsets(
    y: NDArray[np.float32],
    sr: int,
    *,
    hop_length: int = 512,
    backtrack: bool = True,
    units: str = "time",
) -> OnsetResult:
    """
    Detect onset times in audio.

    Parameters
    ----------
    y
        Audio time series (mono).
    sr
        Sample rate.
    hop_length
        Hop length for analysis.
    backtrack
        If True, backtrack from onset peaks to nearest preceding minimum.
    units
        Units for onset times: "time" (seconds), "frames", "samples".

    Returns
    -------
    OnsetResult
        Onset times and strengths.

    Examples
    --------
    >>> y, sr = librosa.load("drums.wav", sr=22050, mono=True)
    >>> result = detect_onsets(y, sr)
    >>> print(f"Found {result.onset_count} onsets")
    """
    logger.debug(f"Detecting onsets (sr={sr}, hop={hop_length})")

    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )

    # Detect onset frames
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=backtrack,
        units="frames",
    )

    # Convert to requested units
    if units == "time":
        onset_times = librosa.frames_to_time(
            onset_frames,
            sr=sr,
            hop_length=hop_length,
        )
    elif units == "samples":
        onset_times = librosa.frames_to_samples(
            onset_frames,
            hop_length=hop_length,
        )
    else:
        onset_times = onset_frames.astype(float)

    # Get onset strengths at detected positions
    onset_strengths = onset_env[onset_frames].tolist()

    # Normalize strengths to 0-1
    if onset_strengths:
        max_strength = max(onset_strengths)
        if max_strength > 0:
            onset_strengths = [s / max_strength for s in onset_strengths]

    logger.debug(f"Detected {len(onset_times)} onsets")

    return OnsetResult(
        onset_times=onset_times.tolist(),
        onset_strengths=onset_strengths,
    )


def detect_beats_and_onsets(
    y: NDArray[np.float32],
    sr: int,
    *,
    hop_length: int = 512,
) -> tuple[OnsetResult, list[float]]:
    """
    Detect both onsets and beat positions.

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
    tuple[OnsetResult, list[float]]
        Onset result and beat times in seconds.
    """
    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Onset detection
    onsets = detect_onsets(y, sr, hop_length=hop_length)

    # Beat tracking
    _, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=hop_length,
    )

    return onsets, beat_times.tolist()


def get_onset_strength_envelope(
    y: NDArray[np.float32],
    sr: int,
    *,
    hop_length: int = 512,
    aggregate: str = "mean",
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """
    Get the onset strength envelope.

    Parameters
    ----------
    y
        Audio time series.
    sr
        Sample rate.
    hop_length
        Hop length.
    aggregate
        Aggregation method: "mean", "median", "max".

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float64]]
        Onset strength envelope and corresponding times.
    """
    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Compute onset strength
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
        aggregate=getattr(np, aggregate),
    )

    # Get time axis
    times = librosa.frames_to_time(
        np.arange(len(onset_env)),
        sr=sr,
        hop_length=hop_length,
    )

    return onset_env.astype(np.float32), times


def segment_by_onsets(
    y: NDArray[np.float32],
    sr: int,
    *,
    min_segment_duration: float = 0.1,
    max_segments: int = 100,
) -> list[tuple[float, float, NDArray[np.float32]]]:
    """
    Segment audio by detected onsets.

    Parameters
    ----------
    y
        Audio time series.
    sr
        Sample rate.
    min_segment_duration
        Minimum segment duration in seconds.
    max_segments
        Maximum number of segments to return.

    Returns
    -------
    list[tuple[float, float, NDArray[np.float32]]]
        List of (start_time, end_time, audio_segment) tuples.
    """
    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Detect onsets
    result = detect_onsets(y, sr)
    onset_times = result.onset_times

    if not onset_times:
        duration = len(y) / sr
        return [(0.0, duration, y)]

    # Add start and end times
    times = [0.0] + onset_times + [len(y) / sr]

    segments = []
    for i in range(len(times) - 1):
        start_time = times[i]
        end_time = times[i + 1]

        # Skip short segments
        if end_time - start_time < min_segment_duration:
            continue

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        segments.append((start_time, end_time, segment))

        if len(segments) >= max_segments:
            break

    return segments
