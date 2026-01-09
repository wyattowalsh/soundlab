"""Spectral feature extraction utilities."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np

from soundlab.analysis.models import SpectralResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _to_mono(y: NDArray[np.float32]) -> NDArray[np.float32]:
    if y.ndim == 1:
        return y
    if y.shape[0] <= y.shape[-1]:
        return np.mean(y, axis=0)
    return np.mean(y, axis=1)


def analyze_spectral(y: NDArray[np.float32], sr: int) -> SpectralResult:
    """Analyze spectral centroid, bandwidth, and rolloff."""
    librosa = importlib.import_module("librosa")
    mono = _to_mono(y)

    centroid = librosa.feature.spectral_centroid(y=mono, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=mono, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=mono, sr=sr)

    return SpectralResult(
        centroid=float(np.mean(centroid)),
        bandwidth=float(np.mean(bandwidth)),
        rolloff=float(np.mean(rolloff)),
    )
