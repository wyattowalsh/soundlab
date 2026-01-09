"""Post-processing utilities for pipeline outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from soundlab.transcription.models import NoteEvent

if TYPE_CHECKING:
    from collections.abc import Sequence


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    if samples.shape[0] <= samples.shape[-1]:
        return np.mean(samples, axis=0)
    return np.mean(samples, axis=1)


def _zero_silence(samples: np.ndarray, threshold: float) -> np.ndarray:
    if samples.size == 0:
        return samples

    abs_samples = np.abs(samples)
    mask = abs_samples < threshold
    cleaned = samples.copy()
    if cleaned.ndim == 1:
        cleaned[mask] = 0.0
    else:
        cleaned[mask] = 0.0
    return cleaned


def clean_stems(
    stems: dict[str, np.ndarray],
    *,
    silence_threshold: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Zero low-level noise without changing alignment."""
    return {name: _zero_silence(audio, silence_threshold) for name, audio in stems.items()}


def mono_amt_exports(stems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert stems to mono for AMT while preserving length."""
    return {name: _to_mono(audio) for name, audio in stems.items()}


def cleanup_midi_notes(
    notes: Sequence[NoteEvent],
    *,
    min_duration: float = 0.02,
    min_velocity: int = 1,
) -> list[NoteEvent]:
    """Filter out very short/quiet notes and clamp values."""
    cleaned: list[NoteEvent] = []
    for note in notes:
        duration = note.end - note.start
        if duration < min_duration:
            continue
        velocity = min(127, max(min_velocity, note.velocity))
        cleaned.append(
            NoteEvent(
                start=note.start,
                end=note.end,
                pitch=min(127, max(0, note.pitch)),
                velocity=velocity,
            )
        )
    return cleaned


__all__ = ["clean_stems", "cleanup_midi_notes", "mono_amt_exports"]
