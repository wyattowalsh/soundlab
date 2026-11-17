"""Tempo detection regressions."""

from __future__ import annotations

import importlib

import numpy as np

from soundlab.analysis.tempo import detect_tempo


def test_detect_tempo_accepts_ndarray_bpm(monkeypatch) -> None:
    """librosa 0.11 returns tempo as an array. That must still become one BPM."""
    librosa = importlib.import_module("librosa")

    def fake_beat_track(**_kwargs):
        return np.array([120.0]), np.array([0, 32])

    monkeypatch.setattr(librosa.beat, "beat_track", fake_beat_track)
    samples = np.zeros(4096, dtype=np.float32)
    result = detect_tempo(samples, 22050)

    assert result.bpm == 120.0
    assert result.confidence == 1.0
    assert result.beats
