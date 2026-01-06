"""Pytest configuration and fixtures for SoundLab tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf


# === Path Fixtures ===

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_output_dir(temp_dir: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    return output_dir


# === Audio Fixtures ===

@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate for tests."""
    return 44100


@pytest.fixture
def sample_mono_audio(sample_rate: int) -> np.ndarray:
    """Generate a mono sine wave audio sample (440Hz, 1 second)."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


@pytest.fixture
def sample_stereo_audio(sample_mono_audio: np.ndarray) -> np.ndarray:
    """Generate stereo audio from mono (same signal both channels)."""
    return np.stack([sample_mono_audio, sample_mono_audio], axis=0)


@pytest.fixture
def sample_audio_3s(sample_rate: int) -> np.ndarray:
    """Generate a 3-second mono audio sample with varying frequencies."""
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Mix of frequencies to simulate real audio
    audio = (
        0.3 * np.sin(2 * np.pi * 220 * t) +  # A3
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.2 * np.sin(2 * np.pi * 880 * t)    # A5
    )
    return audio.astype(np.float32)


@pytest.fixture
def silence_audio(sample_rate: int) -> np.ndarray:
    """Generate 1 second of silence."""
    return np.zeros(sample_rate, dtype=np.float32)


# === Audio File Fixtures ===

@pytest.fixture
def sample_audio_path(temp_dir: Path, sample_mono_audio: np.ndarray, sample_rate: int) -> Path:
    """Create a temporary WAV file with sample audio."""
    audio_path = temp_dir / "test_audio.wav"
    sf.write(audio_path, sample_mono_audio, sample_rate)
    return audio_path


@pytest.fixture
def sample_stereo_audio_path(temp_dir: Path, sample_stereo_audio: np.ndarray, sample_rate: int) -> Path:
    """Create a temporary stereo WAV file."""
    audio_path = temp_dir / "test_stereo.wav"
    sf.write(audio_path, sample_stereo_audio.T, sample_rate)  # Transpose for soundfile
    return audio_path


@pytest.fixture
def sample_audio_3s_path(temp_dir: Path, sample_audio_3s: np.ndarray, sample_rate: int) -> Path:
    """Create a temporary 3-second WAV file."""
    audio_path = temp_dir / "test_audio_3s.wav"
    sf.write(audio_path, sample_audio_3s, sample_rate)
    return audio_path


# === GPU Mocking Fixtures ===

@pytest.fixture
def mock_cuda_available():
    """Mock CUDA as available."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="Mock GPU"):
            with patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 16 * 1024**3)):
                yield


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA as unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_gpu_available(mock_cuda_available):
    """Alias for mock_cuda_available."""
    yield


# === Model Mocking Fixtures ===

@pytest.fixture
def mock_demucs_model():
    """Mock Demucs model for testing without loading actual model."""
    mock_model = MagicMock()
    mock_model.samplerate = 44100
    mock_model.audio_channels = 2
    mock_model.sources = ["vocals", "drums", "bass", "other"]

    with patch("demucs.pretrained.get_model", return_value=mock_model):
        yield mock_model


@pytest.fixture
def mock_basic_pitch():
    """Mock Basic Pitch for testing without actual transcription."""
    mock_output = (
        MagicMock(),  # model_output
        MagicMock(),  # midi_data
        [  # note_events: (start, end, pitch, velocity, pitch_bend)
            (0.0, 0.5, 60, 0.8, None),
            (0.5, 1.0, 62, 0.7, None),
            (1.0, 1.5, 64, 0.9, None),
        ],
    )

    with patch("basic_pitch.inference.predict", return_value=mock_output):
        yield


# === AudioSegment Fixtures ===

@pytest.fixture
def audio_segment(sample_mono_audio: np.ndarray, sample_rate: int):
    """Create an AudioSegment instance."""
    from soundlab.core.audio import AudioSegment
    return AudioSegment(
        samples=sample_mono_audio,
        sample_rate=sample_rate,
    )


@pytest.fixture
def audio_segment_stereo(sample_stereo_audio: np.ndarray, sample_rate: int):
    """Create a stereo AudioSegment instance."""
    from soundlab.core.audio import AudioSegment
    return AudioSegment(
        samples=sample_stereo_audio,
        sample_rate=sample_rate,
    )


# === Config Fixtures ===

@pytest.fixture
def separation_config():
    """Create a default separation config."""
    from soundlab.separation.models import SeparationConfig
    return SeparationConfig()


@pytest.fixture
def transcription_config():
    """Create a default transcription config."""
    from soundlab.transcription.models import TranscriptionConfig
    return TranscriptionConfig()


# === Markers ===

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
