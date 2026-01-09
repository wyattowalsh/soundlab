"""Onset detection utilities."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from numpy.typing import NDArray


class OnsetResult(BaseModel):
    """Onset detection result."""

    model_config = ConfigDict(frozen=True)

    timestamps: list[float] = Field(default_factory=list)
    count: Annotated[int, Field(ge=0)] = 0
    strength: Annotated[float, Field(ge=0.0)] = 0.0


def _to_mono(y: NDArray[np.float32]) -> NDArray[np.float32]:
    if y.ndim == 1:
        return y
    if y.shape[0] <= y.shape[-1]:
        return np.mean(y, axis=0)
    return np.mean(y, axis=1)


def detect_onsets(
    y: NDArray[np.float32],
    sr: int,
    *,
    hop_length: int = 512,
) -> OnsetResult:
    """Detect onsets and return timing statistics."""
    librosa = importlib.import_module("librosa")
    mono = _to_mono(y)

    onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    timestamps = [float(value) for value in np.asarray(onset_times).tolist()]
    strength = float(np.mean(onset_env)) if onset_env.size else 0.0

    return OnsetResult(timestamps=timestamps, count=len(timestamps), strength=strength)


__all__ = ["OnsetResult", "detect_onsets"]
