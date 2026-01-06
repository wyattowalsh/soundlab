"""Tests for soundlab.io.audio_io."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from soundlab.core.audio import AudioFormat, AudioSegment, BitDepth
from soundlab.core.exceptions import AudioFormatError, AudioLoadError
from soundlab.io.audio_io import get_audio_metadata, load_audio, save_audio


class TestLoadAudio:
    """Test load_audio function."""

    def test_load_wav_file(self, sample_audio_path: Path):
        """Should load WAV file correctly."""
        segment = load_audio(sample_audio_path)

        assert isinstance(segment, AudioSegment)
        assert segment.sample_rate == 44100
        assert segment.samples.dtype == np.float32

    def test_load_preserves_sample_rate(self, temp_dir: Path):
        """Should preserve original sample rate."""
        # Create file with specific sample rate
        audio = np.zeros(22050, dtype=np.float32)
        path = temp_dir / "test_22050.wav"
        sf.write(path, audio, 22050)

        segment = load_audio(path)
        assert segment.sample_rate == 22050

    def test_load_with_target_sr(self, sample_audio_path: Path):
        """Should resample to target sample rate."""
        segment = load_audio(sample_audio_path, target_sr=22050)
        assert segment.sample_rate == 22050

    def test_load_mono(self, sample_stereo_audio_path: Path):
        """Should convert to mono when requested."""
        segment = load_audio(sample_stereo_audio_path, mono=True)
        assert segment.channels == 1

    def test_load_nonexistent_file(self):
        """Should raise AudioLoadError for missing file."""
        with pytest.raises(AudioLoadError):
            load_audio("/nonexistent/path/audio.wav")

    def test_load_unsupported_format(self, temp_dir: Path):
        """Should raise AudioFormatError for unsupported format."""
        path = temp_dir / "test.xyz"
        path.write_text("not audio")

        with pytest.raises(AudioFormatError):
            load_audio(path)

    def test_load_sets_source_path(self, sample_audio_path: Path):
        """Should set source_path on loaded segment."""
        segment = load_audio(sample_audio_path)
        assert segment.source_path == sample_audio_path

    def test_load_sets_metadata(self, sample_audio_path: Path):
        """Should set metadata on loaded segment."""
        segment = load_audio(sample_audio_path)
        assert segment.metadata is not None
        assert segment.metadata.sample_rate == 44100


class TestSaveAudio:
    """Test save_audio function."""

    def test_save_creates_file(self, audio_segment: AudioSegment, temp_dir: Path):
        """Should create audio file."""
        output_path = temp_dir / "output.wav"
        result = save_audio(audio_segment, output_path)

        assert result.exists()
        assert result == output_path

    def test_save_creates_parent_directories(self, audio_segment: AudioSegment, temp_dir: Path):
        """Should create parent directories if needed."""
        output_path = temp_dir / "nested" / "dirs" / "output.wav"
        result = save_audio(audio_segment, output_path)

        assert result.exists()

    def test_save_preserves_audio_data(self, temp_dir: Path, sample_mono_audio: np.ndarray, sample_rate: int):
        """Should preserve audio data through save/load cycle."""
        segment = AudioSegment(samples=sample_mono_audio, sample_rate=sample_rate)
        output_path = temp_dir / "output.wav"

        save_audio(segment, output_path)
        loaded = load_audio(output_path)

        np.testing.assert_array_almost_equal(
            loaded.samples,
            sample_mono_audio,
            decimal=4,
        )

    def test_save_with_int24(self, audio_segment: AudioSegment, temp_dir: Path):
        """Should save with INT24 bit depth."""
        output_path = temp_dir / "output.wav"
        save_audio(audio_segment, output_path, bit_depth=BitDepth.INT24)

        info = sf.info(output_path)
        assert "24" in info.subtype

    def test_save_with_int16(self, audio_segment: AudioSegment, temp_dir: Path):
        """Should save with INT16 bit depth."""
        output_path = temp_dir / "output.wav"
        save_audio(audio_segment, output_path, bit_depth=BitDepth.INT16)

        info = sf.info(output_path)
        assert "16" in info.subtype


class TestGetAudioMetadata:
    """Test get_audio_metadata function."""

    def test_get_metadata(self, sample_audio_path: Path):
        """Should return audio metadata."""
        meta = get_audio_metadata(sample_audio_path)

        assert meta.sample_rate == 44100
        assert meta.channels == 1
        assert meta.duration_seconds > 0

    def test_get_metadata_stereo(self, sample_stereo_audio_path: Path):
        """Should detect stereo channels."""
        meta = get_audio_metadata(sample_stereo_audio_path)
        assert meta.channels == 2

    def test_get_metadata_nonexistent_file(self):
        """Should raise AudioLoadError for missing file."""
        with pytest.raises(AudioLoadError):
            get_audio_metadata("/nonexistent/file.wav")


class TestRoundTrip:
    """Test save and load roundtrip."""

    def test_mono_roundtrip(self, temp_dir: Path, sample_mono_audio: np.ndarray, sample_rate: int):
        """Mono audio should survive roundtrip."""
        original = AudioSegment(samples=sample_mono_audio, sample_rate=sample_rate)
        path = temp_dir / "mono.wav"

        save_audio(original, path)
        loaded = load_audio(path)

        assert loaded.channels == 1
        assert loaded.sample_rate == sample_rate
        np.testing.assert_array_almost_equal(loaded.samples, sample_mono_audio, decimal=4)

    def test_stereo_roundtrip(self, temp_dir: Path, sample_stereo_audio: np.ndarray, sample_rate: int):
        """Stereo audio should survive roundtrip."""
        original = AudioSegment(samples=sample_stereo_audio, sample_rate=sample_rate)
        path = temp_dir / "stereo.wav"

        save_audio(original, path)
        loaded = load_audio(path)

        assert loaded.channels == 2
        assert loaded.sample_rate == sample_rate
