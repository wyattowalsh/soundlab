"""Loudness measurement on a synthetic tone."""

from __future__ import annotations

import numpy as np
import pytest

from soundlab.analysis.loudness import measure_loudness


def test_measure_loudness_reports_lufs_for_a_one_second_tone() -> None:
    pytest.importorskip("pyloudnorm")
    sample_rate = 48000
    seconds = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 440 * seconds)).astype(np.float32)

    result = measure_loudness(tone, sample_rate)

    assert np.isfinite(result.lufs)
    assert result.peak == pytest.approx(0.2, abs=1e-3)
