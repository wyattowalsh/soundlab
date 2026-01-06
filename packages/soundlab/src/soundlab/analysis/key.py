"""Musical key detection using Krumhansl-Schmuckler algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import librosa
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from soundlab.analysis.models import KeyDetectionResult, Mode, MusicalKey

if TYPE_CHECKING:
    pass


__all__ = [
    "detect_key",
    "get_relative_key",
    "get_parallel_key",
    "get_compatible_keys",
]


# Krumhansl-Schmuckler key profiles (normalized)
_MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
])
_MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])


def detect_key(
    y: NDArray[np.float32],
    sr: int,
    *,
    hop_length: int = 512,
    use_cqt: bool = True,
) -> KeyDetectionResult:
    """
    Detect the musical key using the Krumhansl-Schmuckler algorithm.

    Parameters
    ----------
    y
        Audio time series (mono).
    sr
        Sample rate.
    hop_length
        Hop length for chroma computation.
    use_cqt
        Use CQT-based chroma (better frequency resolution).

    Returns
    -------
    KeyDetectionResult
        Detected key, mode, and confidence score.

    Examples
    --------
    >>> y, sr = librosa.load("song.mp3", sr=22050, mono=True)
    >>> result = detect_key(y, sr)
    >>> print(result.name)
    'A minor'
    >>> print(result.camelot)
    '8A'
    """
    logger.debug(f"Detecting key (sr={sr}, hop={hop_length})")

    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Compute chroma features
    if use_cqt:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    else:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    # Average across time to get pitch class distribution
    chroma_avg = np.mean(chroma, axis=1)

    # Normalize
    chroma_norm = chroma_avg / (np.linalg.norm(chroma_avg) + 1e-8)
    major_norm = _MAJOR_PROFILE / np.linalg.norm(_MAJOR_PROFILE)
    minor_norm = _MINOR_PROFILE / np.linalg.norm(_MINOR_PROFILE)

    keys = list(MusicalKey)
    all_correlations: dict[str, float] = {}
    best_corr = -1.0
    best_key = MusicalKey.C
    best_mode = Mode.MAJOR

    for i, key in enumerate(keys):
        # Roll chroma to align with current key
        rolled = np.roll(chroma_norm, -i)

        # Correlate with both profiles
        maj_corr = float(np.corrcoef(rolled, major_norm)[0, 1])
        min_corr = float(np.corrcoef(rolled, minor_norm)[0, 1])

        # Handle NaN correlations
        if np.isnan(maj_corr):
            maj_corr = 0.0
        if np.isnan(min_corr):
            min_corr = 0.0

        all_correlations[f"{key.value} major"] = maj_corr
        all_correlations[f"{key.value} minor"] = min_corr

        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = key
            best_mode = Mode.MAJOR

        if min_corr > best_corr:
            best_corr = min_corr
            best_key = key
            best_mode = Mode.MINOR

    # Convert correlation to confidence (0-1 scale)
    confidence = (best_corr + 1) / 2  # Map [-1, 1] to [0, 1]
    confidence = float(np.clip(confidence, 0.0, 1.0))

    logger.debug(f"Detected key: {best_key.value} {best_mode.value} (confidence: {confidence:.2f})")

    return KeyDetectionResult(
        key=best_key,
        mode=best_mode,
        confidence=confidence,
        all_correlations=all_correlations,
    )


def get_relative_key(result: KeyDetectionResult) -> tuple[MusicalKey, Mode]:
    """
    Get the relative major/minor key.

    Parameters
    ----------
    result
        Key detection result.

    Returns
    -------
    tuple[MusicalKey, Mode]
        Relative key and mode.
    """
    keys = list(MusicalKey)
    current_idx = keys.index(result.key)

    if result.mode == Mode.MAJOR:
        # Relative minor is 3 semitones down
        relative_idx = (current_idx - 3) % 12
        return (keys[relative_idx], Mode.MINOR)
    else:
        # Relative major is 3 semitones up
        relative_idx = (current_idx + 3) % 12
        return (keys[relative_idx], Mode.MAJOR)


def get_parallel_key(result: KeyDetectionResult) -> tuple[MusicalKey, Mode]:
    """
    Get the parallel major/minor key (same root, different mode).

    Parameters
    ----------
    result
        Key detection result.

    Returns
    -------
    tuple[MusicalKey, Mode]
        Parallel key and mode.
    """
    if result.mode == Mode.MAJOR:
        return (result.key, Mode.MINOR)
    else:
        return (result.key, Mode.MAJOR)


def get_compatible_keys(result: KeyDetectionResult) -> list[tuple[MusicalKey, Mode, str]]:
    """
    Get harmonically compatible keys for mixing.

    Parameters
    ----------
    result
        Key detection result.

    Returns
    -------
    list[tuple[MusicalKey, Mode, str]]
        List of (key, mode, relationship) tuples.
    """
    compatible = []
    keys = list(MusicalKey)
    current_idx = keys.index(result.key)

    # Same key
    compatible.append((result.key, result.mode, "same"))

    # Relative major/minor
    rel_key, rel_mode = get_relative_key(result)
    compatible.append((rel_key, rel_mode, "relative"))

    # Dominant (5th up)
    dom_idx = (current_idx + 7) % 12
    compatible.append((keys[dom_idx], result.mode, "dominant"))

    # Subdominant (5th down / 4th up)
    sub_idx = (current_idx + 5) % 12
    compatible.append((keys[sub_idx], result.mode, "subdominant"))

    # Parallel major/minor
    par_key, par_mode = get_parallel_key(result)
    compatible.append((par_key, par_mode, "parallel"))

    return compatible
