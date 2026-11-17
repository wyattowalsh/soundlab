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
    """Measure integrated loudness with pyloudnorm.

    ``dynamic_range`` is peak level in dBFS minus integrated LUFS. It is a
    crest proxy, not an ITU loudness-range measurement.
    """
    try:
        pyln = importlib.import_module("pyloudnorm")
    except ImportError as exc:
        raise ImportError(
            "pyloudnorm is required for loudness measurement. "
            'Install this package with its core dependencies, or `uv pip install "pyloudnorm>=0.1.1,<0.2"`.'
        ) from exc

    audio = _prepare_loudness_input(y)
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    peak_db = 20.0 * float(np.log10(peak + 1e-12))
    dynamic_range = peak_db - lufs if np.isfinite(lufs) else 0.0

    return LoudnessResult(lufs=lufs, dynamic_range=float(dynamic_range), peak=peak)
