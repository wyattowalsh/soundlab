"""Musical key detection utilities."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np

from soundlab.analysis.models import KeyDetectionResult, Mode, MusicalKey

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Krumhansl-Schmuckler key profiles (normalized)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _to_mono(y: NDArray[np.float32]) -> NDArray[np.float32]:
    if y.ndim == 1:
        return y
    if y.shape[0] <= y.shape[-1]:
        return np.mean(y, axis=0)
    return np.mean(y, axis=1)


def detect_key(
    y: NDArray[np.float32],
    sr: int,
    *,
    hop_length: int = 512,
) -> KeyDetectionResult:
    """
    Detect the musical key using the Krumhansl-Schmuckler algorithm.
    """
    librosa = importlib.import_module("librosa")
    mono = _to_mono(y)

    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=hop_length)
    chroma_avg = np.mean(chroma, axis=1)

    chroma_avg = chroma_avg / (np.linalg.norm(chroma_avg) + 1e-8)
    major_norm = _MAJOR_PROFILE / np.linalg.norm(_MAJOR_PROFILE)
    minor_norm = _MINOR_PROFILE / np.linalg.norm(_MINOR_PROFILE)

    keys = list(MusicalKey)
    all_correlations: dict[str, float] = {}
    best_corr = -1.0
    best_key = MusicalKey.C
    best_mode = Mode.MAJOR

    for i, key in enumerate(keys):
        rolled = np.roll(chroma_avg, -i)
        maj_corr = float(np.corrcoef(rolled, major_norm)[0, 1])
        min_corr = float(np.corrcoef(rolled, minor_norm)[0, 1])

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

    confidence = (best_corr + 1) / 2

    return KeyDetectionResult(
        key=best_key,
        mode=best_mode,
        confidence=confidence,
        all_correlations=all_correlations,
    )


__all__ = ["detect_key"]
