"""Tests for soundlab.core.audio models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from soundlab.core.audio import (
    AudioFormat,
    AudioMetadata,
    AudioSegment,
    BitDepth,
    SampleRate,
)


class TestAudioFormat:
    """Test AudioFormat enum."""

    def test_wav_format(self):
        """WAV format should have correct value."""
        assert AudioFormat.WAV.value == "wav"

    def test_mp3_format(self):
        """MP3 format should have correct value."""
        assert AudioFormat.MP3.value == "mp3"

    def test_all_formats_exist(self):
        """All expected formats should exist."""
        formats = {f.value for f in AudioFormat}
        expected = {"wav", "mp3", "flac", "ogg", "aiff", "m4a"}
        assert expected == formats


class TestSampleRate:
    """Test SampleRate enum."""

    def test_sr_44100_hz_property(self):
        """SampleRate.SR_44100 should have hz property returning 44100."""
        assert SampleRate.SR_44100.hz == 44100

    def test_sr_48000_hz_property(self):
        """SampleRate.SR_48000 should have hz property returning 48000."""
        assert SampleRate.SR_48000.hz == 48000

    def test_all_sample_rates(self):
        """All sample rates should have valid hz property."""
        for sr in SampleRate:
            assert isinstance(sr.hz, int)
            assert sr.hz > 0


class TestBitDepth:
    """Test BitDepth enum."""

    def test_int16(self):
        """INT16 should have value 'int16'."""
        assert BitDepth.INT16.value == "int16"

    def test_int24(self):
        """INT24 should have value 'int24'."""
        assert BitDepth.INT24.value == "int24"

    def test_float32(self):
        """FLOAT32 should have value 'float32'."""
        assert BitDepth.FLOAT32.value == "float32"


class TestAudioMetadata:
    """Test AudioMetadata model."""

    def test_create_valid_metadata(self):
        """Should create valid metadata."""
        meta = AudioMetadata(
            duration_seconds=10.5,
            sample_rate=44100,
            channels=2,
        )
        assert meta.duration_seconds == 10.5
        assert meta.sample_rate == 44100
        assert meta.channels == 2

    def test_duration_str_property(self):
        """duration_str should return formatted time."""
        meta = AudioMetadata(duration_seconds=125.5, sample_rate=44100, channels=2)
        assert meta.duration_str == "02:05.500"

    def test_is_stereo(self):
        """is_stereo should return True for 2 channels."""
        meta = AudioMetadata(duration_seconds=1.0, sample_rate=44100, channels=2)
        assert meta.is_stereo is True

    def test_is_mono(self):
        """is_mono should return True for 1 channel."""
        meta = AudioMetadata(duration_seconds=1.0, sample_rate=44100, channels=1)
        assert meta.is_mono is True

    def test_invalid_negative_duration(self):
        """Should reject negative duration."""
        with pytest.raises(ValidationError):
            AudioMetadata(duration_seconds=-1.0, sample_rate=44100, channels=1)

    def test_invalid_zero_channels(self):
        """Should reject zero channels."""
        with pytest.raises(ValidationError):
            AudioMetadata(duration_seconds=1.0, sample_rate=44100, channels=0)

    def test_metadata_is_frozen(self):
        """Metadata should be immutable."""
        meta = AudioMetadata(duration_seconds=1.0, sample_rate=44100, channels=1)
        with pytest.raises(ValidationError):
            meta.duration_seconds = 2.0


class TestAudioSegment:
    """Test AudioSegment model."""

    def test_create_mono_segment(self):
        """Should create mono segment."""
        samples = np.zeros(44100, dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)
        assert segment.channels == 1

    def test_create_stereo_segment(self):
        """Should create stereo segment."""
        samples = np.zeros((2, 44100), dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)
        assert segment.channels == 2

    def test_duration_seconds_property(self):
        """duration_seconds should be calculated correctly."""
        samples = np.zeros(44100, dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)
        assert segment.duration_seconds == 1.0

    def test_ensure_float32_conversion(self):
        """Should convert non-float32 to float32."""
        samples = np.zeros(44100, dtype=np.float64)
        segment = AudioSegment(samples=samples, sample_rate=44100)
        assert segment.samples.dtype == np.float32

    def test_to_mono_from_stereo(self):
        """to_mono should convert stereo to mono."""
        left = np.ones(44100, dtype=np.float32) * 0.5
        right = np.ones(44100, dtype=np.float32) * 0.5
        stereo = np.stack([left, right], axis=0)

        segment = AudioSegment(samples=stereo, sample_rate=44100)
        mono = segment.to_mono()

        assert mono.channels == 1
        np.testing.assert_array_almost_equal(mono.samples, np.full(44100, 0.5, dtype=np.float32))

    def test_to_mono_from_mono_returns_same(self):
        """to_mono on mono segment should return same data."""
        samples = np.ones(44100, dtype=np.float32) * 0.5
        segment = AudioSegment(samples=samples, sample_rate=44100)
        mono = segment.to_mono()

        assert mono.channels == 1
        np.testing.assert_array_equal(mono.samples, segment.samples)

    def test_source_path_optional(self):
        """source_path should be optional."""
        samples = np.zeros(44100, dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)
        assert segment.source_path is None

    def test_source_path_preserved(self):
        """source_path should be preserved."""
        samples = np.zeros(44100, dtype=np.float32)
        path = Path("/test/audio.wav")
        segment = AudioSegment(samples=samples, sample_rate=44100, source_path=path)
        assert segment.source_path == path
