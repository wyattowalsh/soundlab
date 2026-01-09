"""Shared pytest fixtures for SoundLab."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio"
SINE_PATH = FIXTURES_DIR / "sine_440hz_3s.wav"
SILENCE_PATH = FIXTURES_DIR / "silence_1s.wav"
SAMPLE_RATE = 22050


@pytest.fixture
def sample_audio_path() -> Path:
    return SINE_PATH


@pytest.fixture
def sample_mono_audio():
    np = pytest.importorskip("numpy")
    duration = 1.0
    count = int(SAMPLE_RATE * duration)
    samples = np.array(
        [0.2 * math.sin(2 * math.pi * 440.0 * (i / SAMPLE_RATE)) for i in range(count)],
        dtype=np.float32,
    )
    return samples


@pytest.fixture
def sample_stereo_audio(sample_mono_audio):
    np = pytest.importorskip("numpy")
    return np.stack([sample_mono_audio, sample_mono_audio])


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def mock_gpu_available(monkeypatch):
    from soundlab.utils import gpu as gpu_utils

    monkeypatch.setattr(gpu_utils, "is_cuda_available", lambda: True)
    return gpu_utils
