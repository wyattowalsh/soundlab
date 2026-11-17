"""Tests for core audio models."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pydantic")

from soundlab.core.audio import AudioFormat, AudioMetadata, AudioSegment, BitDepth


def test_audio_metadata_properties() -> None:
    metadata = AudioMetadata(
        duration_seconds=65.123,
        sample_rate=44100,
        channels=2,
        bit_depth=BitDepth.INT24,
        format=AudioFormat.WAV,
    )

    assert metadata.is_stereo is True
    assert metadata.is_mono is False
    assert metadata.duration_str == "01:05.12"


def test_audio_segment_validation_and_channels() -> None:
    samples = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    segment = AudioSegment(samples=samples, sample_rate=22050)

    assert segment.samples.dtype == np.float32
    assert segment.channels == 1
    assert pytest.approx(segment.duration_seconds) == 3 / 22050


def test_stereo_duration_uses_time_axis() -> None:
    """Channels-first duration follows shape[-1], the time axis."""
    samples = np.ones((2, 48000), dtype=np.float32)
    segment = AudioSegment(samples=samples, sample_rate=48000)

    assert segment.channels == 2
    assert segment.duration_seconds == pytest.approx(1.0)
    assert type(segment.duration_seconds) is float

    three_channel = AudioSegment(
        samples=np.zeros((3, 100), dtype=np.float32),
        sample_rate=1000,
    )
    assert three_channel.channels == 3
    assert three_channel.duration_seconds == pytest.approx(100 / 1000)


def test_audio_segment_to_mono() -> None:
    samples = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    segment = AudioSegment(samples=samples, sample_rate=48000)
    mono = segment.to_mono()

    assert mono.channels == 1
    assert np.allclose(mono.samples, np.array([0.5, 0.5], dtype=np.float32))

    mono_direct = AudioSegment(samples=np.array([0.1, 0.2], dtype=np.float32), sample_rate=48000)
    assert mono_direct.to_mono() is mono_direct
