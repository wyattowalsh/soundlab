"""Loudness measurement utilities."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np

from soundlab.analysis.models import LoudnessResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _prepare_loudness_input(y: NDArray[np.float32]) -> NDArray[np.float32]:
    if y.ndim == 1:
        return y
    if y.shape[0] <= y.shape[-1]:
        return y.T
    return y


def measure_loudness(y: NDArray[np.float32], sr: int) -> LoudnessResult:
    """Measure integrated loudness using pyloudnorm."""
    pyln = importlib.import_module("pyloudnorm")

    audio = _prepare_loudness_input(y)
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(audio))
    dynamic_range = float(meter.loudness_range(audio))
    peak = float(np.max(np.abs(y))) if y.size else 0.0

    return LoudnessResult(lufs=lufs, dynamic_range=dynamic_range, peak=peak)
