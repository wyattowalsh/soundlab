"""Tempo detection utilities."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np

from soundlab.analysis.models import TempoResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _to_mono(y: NDArray[np.float32]) -> NDArray[np.float32]:
    if y.ndim == 1:
        return y
    if y.shape[0] <= y.shape[-1]:
        return np.mean(y, axis=0)
    return np.mean(y, axis=1)


def detect_tempo(
    y: NDArray[np.float32],
    sr: int,
    *,
    hop_length: int = 512,
) -> TempoResult:
    """Detect tempo using librosa's beat tracking."""
    librosa = importlib.import_module("librosa")
    mono = _to_mono(y)
    tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=sr, hop_length=hop_length)

    beat_frames = np.asarray(beat_frames) if beat_frames is not None else np.array([], dtype=int)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    beats = [float(value) for value in np.asarray(beat_times).tolist()]
    confidence = 0.0 if beat_frames.size == 0 else 1.0

    return TempoResult(bpm=float(tempo), confidence=confidence, beats=beats)
