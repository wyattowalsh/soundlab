"""Audio analysis utilities."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from soundlab.analysis.key import detect_key
from soundlab.analysis.loudness import measure_loudness
from soundlab.analysis.models import (
    AnalysisResult,
    KeyDetectionResult,
    LoudnessResult,
    Mode,
    MusicalKey,
    SpectralResult,
    TempoResult,
)
from soundlab.analysis.onsets import OnsetResult, detect_onsets
from soundlab.analysis.spectral import analyze_spectral
from soundlab.analysis.tempo import detect_tempo
from soundlab.io import load_audio


def analyze_audio(path: str | Path) -> AnalysisResult:
    """Run tempo, key, loudness, and spectral analysis on an audio file."""
    segment = load_audio(path)
    samples = segment.samples
    sr = segment.sample_rate

    tempo = detect_tempo(samples, sr)
    key = detect_key(samples, sr)
    loudness = measure_loudness(samples, sr)
    spectral = analyze_spectral(samples, sr)
    _onsets = detect_onsets(samples, sr)

    return AnalysisResult(
        tempo=tempo,
        key=key,
        loudness=loudness,
        spectral=spectral,
    )


__all__ = [
    "AnalysisResult",
    "KeyDetectionResult",
    "LoudnessResult",
    "Mode",
    "MusicalKey",
    "OnsetResult",
    "SpectralResult",
    "TempoResult",
    "analyze_audio",
    "analyze_spectral",
    "detect_key",
    "detect_onsets",
    "detect_tempo",
    "measure_loudness",
]
