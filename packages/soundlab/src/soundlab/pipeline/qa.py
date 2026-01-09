"""Pipeline QA utilities."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np

from soundlab.pipeline.models import MIDIQAResult, QAConfig, StemQAResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from soundlab.transcription.models import NoteEvent


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    if samples.shape[0] <= samples.shape[-1]:
        return np.mean(samples, axis=0)
    return np.mean(samples, axis=1)


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def reconstruction_error(mix: np.ndarray, stems: dict[str, np.ndarray]) -> float:
    """Return RMS reconstruction error between mix and sum of stems."""
    if not stems:
        return 1.0

    total = np.zeros_like(mix)
    for stem in stems.values():
        if stem.shape != mix.shape:
            stem = np.broadcast_to(stem, mix.shape)
        total = total + stem

    residual = mix - total
    return _rms(residual) / (_rms(mix) + 1e-8)


def spectral_flatness(samples: np.ndarray, sr: int) -> float:
    """Compute mean spectral flatness for the given audio."""
    _ = sr
    librosa = importlib.import_module("librosa")
    mono = _to_mono(samples)
    flatness = librosa.feature.spectral_flatness(y=mono)
    return float(np.mean(flatness)) if flatness.size else 0.0


def clipping_ratio(samples: np.ndarray, threshold: float = 0.999) -> float:
    """Return fraction of samples exceeding a clipping threshold."""
    if samples.size == 0:
        return 0.0
    return float(np.mean(np.abs(samples) >= threshold))


def stereo_coherence(samples: np.ndarray) -> float:
    """Estimate stereo coherence as correlation between channels."""
    if samples.ndim == 1:
        return 1.0
    if samples.shape[0] <= samples.shape[-1]:
        left = samples[0]
        right = samples[1] if samples.shape[0] > 1 else samples[0]
    else:
        left = samples[:, 0]
        right = samples[:, 1] if samples.shape[1] > 1 else samples[:, 0]

    if left.size == 0 or right.size == 0:
        return 0.0

    corr = np.corrcoef(left, right)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def leakage_ratio(stems: dict[str, np.ndarray]) -> float:
    """Compute a simple leakage proxy using residual energy."""
    if not stems:
        return 1.0

    energies = {name: _rms(audio) for name, audio in stems.items()}
    total_energy = sum(energies.values()) + 1e-8
    dominant = max(energies.values()) if energies else 0.0
    return float((total_energy - dominant) / total_energy)


def score_separation(
    mix: np.ndarray,
    stems: dict[str, np.ndarray],
    sr: int,
    *,
    qa: QAConfig | None = None,
) -> StemQAResult:
    """Compute QA metrics and aggregate score for stems."""
    qa_config = qa or QAConfig()

    recon_error = reconstruction_error(mix, stems)
    residual = mix - sum((stem for stem in stems.values()), np.zeros_like(mix)) if stems else mix
    flatness = spectral_flatness(residual, sr)
    clip_ratio = clipping_ratio(mix)
    coherence = stereo_coherence(mix)
    leakage = leakage_ratio(stems)

    penalties = [
        recon_error / max(qa_config.max_reconstruction_error, 1e-6),
        clip_ratio / max(qa_config.max_clipping_ratio, 1e-6),
        max(0.0, qa_config.min_spectral_flatness - flatness)
        / max(qa_config.min_spectral_flatness, 1e-6),
        max(0.0, qa_config.min_stereo_coherence - coherence)
        / max(qa_config.min_stereo_coherence, 1e-6),
        leakage / max(qa_config.max_leakage_ratio, 1e-6),
    ]

    score = float(max(0.0, 1.0 - np.mean(penalties)))
    metrics = {
        "reconstruction_error": recon_error,
        "spectral_flatness": flatness,
        "clipping_ratio": clip_ratio,
        "stereo_coherence": coherence,
        "leakage_ratio": leakage,
    }
    stem_scores = {name: _rms(audio) for name, audio in stems.items()}

    return StemQAResult(
        score=score,
        metrics=metrics,
        stem_scores=stem_scores,
        passed=score >= qa_config.min_overall_score,
    )


def _notes_duration(notes: Sequence[NoteEvent]) -> float:
    if not notes:
        return 0.0
    return max(note.end for note in notes) - min(note.start for note in notes)


def _max_polyphony(notes: Sequence[NoteEvent]) -> int:
    events: list[tuple[float, int]] = []
    for note in notes:
        events.append((note.start, 1))
        events.append((note.end, -1))
    events.sort()
    current = 0
    peak = 0
    for _time, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def score_midi(
    notes: Sequence[NoteEvent],
    *,
    duration: float | None = None,
    qa: QAConfig | None = None,
) -> MIDIQAResult:
    """Score MIDI sanity metrics."""
    qa_config = qa or QAConfig()

    total_notes = len(notes)
    duration_seconds = duration or _notes_duration(notes)
    duration_seconds = duration_seconds if duration_seconds > 0 else 1.0

    notes_per_second = total_notes / duration_seconds
    max_polyphony = _max_polyphony(notes)
    pitch_range = (
        (max(note.pitch for note in notes) - min(note.pitch for note in notes)) if notes else 0
    )

    score = 1.0
    if total_notes == 0:
        score = 0.0
    if notes_per_second < 0.1 or notes_per_second > 25.0:
        score *= 0.6
    if max_polyphony > 12:
        score *= 0.7

    metrics = {
        "notes": float(total_notes),
        "notes_per_second": float(notes_per_second),
        "max_polyphony": float(max_polyphony),
        "pitch_range": float(pitch_range),
    }

    return MIDIQAResult(
        score=float(score),
        metrics=metrics,
        passed=score >= qa_config.min_midi_score,
    )


__all__ = [
    "clipping_ratio",
    "leakage_ratio",
    "reconstruction_error",
    "score_midi",
    "score_separation",
    "spectral_flatness",
    "stereo_coherence",
]
