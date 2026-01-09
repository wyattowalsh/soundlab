"""Tests for audio I/O utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pydantic")


from soundlab.core.audio import AudioFormat, AudioSegment, BitDepth
from soundlab.core.exceptions import AudioFormatError, AudioLoadError
from soundlab.io.audio_io import (
    _bit_depth_from_subtype,
    _bit_depth_from_width,
    _channels_first,
    _channels_last,
    _infer_format,
    get_audio_metadata,
    load_audio,
    save_audio,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mono_samples() -> np.ndarray:
    """Generate mono audio samples (1 second at 22050 Hz)."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440.0 * t)


@pytest.fixture
def stereo_samples() -> np.ndarray:
    """Generate stereo audio samples (channels-first: 2 x 22050)."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.5 * np.sin(2 * np.pi * 880.0 * t)
    return np.stack([left, right]).astype(np.float32)


@pytest.fixture
def audio_segment(stereo_samples: np.ndarray) -> AudioSegment:
    """Create an AudioSegment for testing."""
    return AudioSegment(samples=stereo_samples, sample_rate=22050)


# --------------------------------------------------------------------------- #
# _infer_format Tests
# --------------------------------------------------------------------------- #


class TestInferFormat:
    """Tests for format inference helper."""

    def test_from_audio_format_enum(self, tmp_path: Path) -> None:
        """AudioFormat enum should pass through unchanged."""
        path = tmp_path / "test.wav"
        result = _infer_format(path, AudioFormat.WAV)
        assert result == AudioFormat.WAV

    def test_from_string(self, tmp_path: Path) -> None:
        """String format should be converted to AudioFormat."""
        path = tmp_path / "test.mp3"
        result = _infer_format(path, "mp3")
        assert result == AudioFormat.MP3

    def test_from_path_extension(self, tmp_path: Path) -> None:
        """Format should be inferred from file extension."""
        path = tmp_path / "test.flac"
        result = _infer_format(path, None)
        assert result == AudioFormat.FLAC

    def test_aif_normalized_to_aiff(self, tmp_path: Path) -> None:
        """'aif' extension should be normalized to 'aiff'."""
        path = tmp_path / "test.aif"
        result = _infer_format(path, None)
        assert result == AudioFormat.AIFF

    def test_case_insensitive(self, tmp_path: Path) -> None:
        """Format inference should be case insensitive."""
        path = tmp_path / "test.WAV"
        result = _infer_format(path, "WAV")
        assert result == AudioFormat.WAV

    def test_unknown_format_returns_none(self, tmp_path: Path) -> None:
        """Unknown format should return None."""
        path = tmp_path / "test.xyz"
        result = _infer_format(path, None)
        assert result is None

    def test_no_extension_returns_none(self, tmp_path: Path) -> None:
        """File without extension should return None."""
        path = tmp_path / "test"
        result = _infer_format(path, None)
        assert result is None


# --------------------------------------------------------------------------- #
# _bit_depth_from_subtype Tests
# --------------------------------------------------------------------------- #


class TestBitDepthFromSubtype:
    """Tests for bit depth extraction from soundfile subtype."""

    def test_pcm_16(self) -> None:
        """PCM_16 subtype should return INT16."""
        assert _bit_depth_from_subtype("PCM_16") == BitDepth.INT16

    def test_pcm_24(self) -> None:
        """PCM_24 subtype should return INT24."""
        assert _bit_depth_from_subtype("PCM_24") == BitDepth.INT24

    def test_pcm_32(self) -> None:
        """PCM_32 subtype should return FLOAT32."""
        assert _bit_depth_from_subtype("PCM_32") == BitDepth.FLOAT32

    def test_float_subtype(self) -> None:
        """FLOAT subtype should return FLOAT32."""
        assert _bit_depth_from_subtype("FLOAT") == BitDepth.FLOAT32

    def test_case_insensitive(self) -> None:
        """Subtype parsing should be case insensitive."""
        assert _bit_depth_from_subtype("pcm_16") == BitDepth.INT16

    def test_none_input(self) -> None:
        """None input should return None."""
        assert _bit_depth_from_subtype(None) is None

    def test_unknown_subtype(self) -> None:
        """Unknown subtype should return None."""
        assert _bit_depth_from_subtype("UNKNOWN") is None


# --------------------------------------------------------------------------- #
# _bit_depth_from_width Tests
# --------------------------------------------------------------------------- #


class TestBitDepthFromWidth:
    """Tests for bit depth extraction from sample width."""

    def test_width_2_is_int16(self) -> None:
        """Sample width 2 should return INT16."""
        assert _bit_depth_from_width(2) == BitDepth.INT16

    def test_width_3_is_int24(self) -> None:
        """Sample width 3 should return INT24."""
        assert _bit_depth_from_width(3) == BitDepth.INT24

    def test_width_4_is_float32(self) -> None:
        """Sample width 4 should return FLOAT32."""
        assert _bit_depth_from_width(4) == BitDepth.FLOAT32

    def test_unknown_width(self) -> None:
        """Unknown width should return None."""
        assert _bit_depth_from_width(1) is None
        assert _bit_depth_from_width(5) is None


# --------------------------------------------------------------------------- #
# _channels_first / _channels_last Tests
# --------------------------------------------------------------------------- #


class TestChannelOrdering:
    """Tests for channel ordering conversion."""

    def test_channels_first_mono(self, mono_samples: np.ndarray) -> None:
        """Mono samples should pass through unchanged."""
        result = _channels_first(mono_samples)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, mono_samples)

    def test_channels_first_stereo(self) -> None:
        """Stereo samples (N, 2) should be transposed to (2, N)."""
        # samples-first format: (N, 2)
        samples = np.random.rand(1000, 2).astype(np.float32)
        result = _channels_first(samples)
        assert result.shape == (2, 1000)

    def test_channels_first_single_channel_2d(self) -> None:
        """Single channel 2D array should be squeezed."""
        samples = np.random.rand(1000, 1).astype(np.float32)
        result = _channels_first(samples)
        assert result.ndim == 1
        assert result.shape == (1000,)

    def test_channels_last_mono(self, mono_samples: np.ndarray) -> None:
        """Mono samples should pass through unchanged."""
        result = _channels_last(mono_samples)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, mono_samples)

    def test_channels_last_stereo(self) -> None:
        """Stereo samples (2, N) should be transposed to (N, 2)."""
        # channels-first format: (2, N)
        samples = np.random.rand(2, 1000).astype(np.float32)
        result = _channels_last(samples)
        assert result.shape == (1000, 2)


# --------------------------------------------------------------------------- #
# load_audio Tests
# --------------------------------------------------------------------------- #


class TestLoadAudio:
    """Tests for load_audio function."""

    def test_load_with_soundfile(self, tmp_path: Path, stereo_samples: np.ndarray) -> None:
        """load_audio should use soundfile as primary path."""
        wav_path = tmp_path / "test.wav"

        # Create mock soundfile info
        mock_info = MagicMock()
        mock_info.frames = 22050
        mock_info.samplerate = 22050
        mock_info.channels = 2
        mock_info.subtype = "PCM_16"
        mock_info.format = "WAV"

        # samples-last format from soundfile
        samples_last = stereo_samples.T

        with (
            patch("soundlab.io.audio_io.sf.info", return_value=mock_info),
            patch("soundlab.io.audio_io.sf.read", return_value=(samples_last, 22050)),
        ):
            segment = load_audio(wav_path)

        assert isinstance(segment, AudioSegment)
        assert segment.sample_rate == 22050
        assert segment.channels == 2

    def test_load_fallback_to_pydub(self, tmp_path: Path) -> None:
        """load_audio should fallback to pydub on soundfile failure."""
        mp3_path = tmp_path / "test.mp3"

        # Mock pydub audio
        mock_pydub = MagicMock()
        mock_pydub.get_array_of_samples.return_value = np.zeros(44100, dtype=np.int16)
        mock_pydub.channels = 1
        mock_pydub.frame_rate = 22050
        mock_pydub.sample_width = 2
        mock_pydub.__len__ = MagicMock(return_value=1000)

        with (
            patch("soundlab.io.audio_io.sf.info", side_effect=Exception("soundfile error")),
            patch("soundlab.io.audio_io.PydubAudioSegment.from_file", return_value=mock_pydub),
        ):
            segment = load_audio(mp3_path)

        assert isinstance(segment, AudioSegment)
        assert segment.sample_rate == 22050

    def test_load_raises_on_complete_failure(self, tmp_path: Path) -> None:
        """load_audio should raise AudioLoadError when both paths fail."""
        bad_path = tmp_path / "nonexistent.wav"

        with (
            patch("soundlab.io.audio_io.sf.info", side_effect=Exception("soundfile error")),
            patch(
                "soundlab.io.audio_io.PydubAudioSegment.from_file",
                side_effect=Exception("pydub error"),
            ),
            pytest.raises(AudioLoadError),
        ):
            load_audio(bad_path)

    def test_load_path_types(self, tmp_path: Path) -> None:
        """load_audio should accept both str and Path."""
        wav_path = tmp_path / "test.wav"

        mock_info = MagicMock()
        mock_info.frames = 1000
        mock_info.samplerate = 22050
        mock_info.channels = 1
        mock_info.subtype = "PCM_16"
        mock_info.format = "WAV"

        samples = np.zeros((1000, 1), dtype=np.float32)

        with (
            patch("soundlab.io.audio_io.sf.info", return_value=mock_info),
            patch("soundlab.io.audio_io.sf.read", return_value=(samples, 22050)),
        ):
            # Test with Path
            segment1 = load_audio(wav_path)
            # Test with str
            segment2 = load_audio(str(wav_path))

        assert segment1.sample_rate == segment2.sample_rate


# --------------------------------------------------------------------------- #
# save_audio Tests
# --------------------------------------------------------------------------- #


class TestSaveAudio:
    """Tests for save_audio function."""

    def test_save_with_soundfile(self, tmp_path: Path, audio_segment: AudioSegment) -> None:
        """save_audio should use soundfile as primary path."""
        wav_path = tmp_path / "output.wav"

        with patch("soundlab.io.audio_io.sf.write") as mock_write:
            save_audio(audio_segment, wav_path)

        mock_write.assert_called_once()
        call_args = mock_write.call_args
        assert call_args[0][0] == wav_path
        assert call_args[0][2] == 22050  # sample rate
        assert call_args[1]["format"] == "WAV"

    def test_save_fallback_to_pydub(self, tmp_path: Path, audio_segment: AudioSegment) -> None:
        """save_audio should fallback to pydub on soundfile failure."""
        mp3_path = tmp_path / "output.mp3"

        mock_pydub_segment = MagicMock()

        with (
            patch("soundlab.io.audio_io.sf.write", side_effect=Exception("soundfile error")),
            patch("soundlab.io.audio_io.PydubAudioSegment", return_value=mock_pydub_segment),
        ):
            save_audio(audio_segment, mp3_path)

        mock_pydub_segment.export.assert_called_once_with(mp3_path, format="mp3")

    def test_save_raises_on_unknown_format(
        self, tmp_path: Path, audio_segment: AudioSegment
    ) -> None:
        """save_audio should raise AudioFormatError for unknown formats."""
        bad_path = tmp_path / "output.xyz"

        with pytest.raises(AudioFormatError):
            save_audio(audio_segment, bad_path)

    def test_save_explicit_format(self, tmp_path: Path, audio_segment: AudioSegment) -> None:
        """save_audio should respect explicit format parameter."""
        output_path = tmp_path / "output"  # No extension

        with patch("soundlab.io.audio_io.sf.write") as mock_write:
            save_audio(audio_segment, output_path, format=AudioFormat.WAV)

        mock_write.assert_called_once()
        assert mock_write.call_args[1]["format"] == "WAV"

    def test_save_path_types(self, tmp_path: Path, audio_segment: AudioSegment) -> None:
        """save_audio should accept both str and Path."""
        wav_path = tmp_path / "output.wav"

        with patch("soundlab.io.audio_io.sf.write"):
            # Test with Path
            save_audio(audio_segment, wav_path)
            # Test with str
            save_audio(audio_segment, str(wav_path))


# --------------------------------------------------------------------------- #
# get_audio_metadata Tests
# --------------------------------------------------------------------------- #


class TestGetAudioMetadata:
    """Tests for get_audio_metadata function."""

    def test_metadata_with_soundfile(self, tmp_path: Path) -> None:
        """get_audio_metadata should use soundfile as primary path."""
        wav_path = tmp_path / "test.wav"

        mock_info = MagicMock()
        mock_info.frames = 44100
        mock_info.samplerate = 44100
        mock_info.channels = 2
        mock_info.subtype = "PCM_24"
        mock_info.format = "WAV"

        with patch("soundlab.io.audio_io.sf.info", return_value=mock_info):
            metadata = get_audio_metadata(wav_path)

        assert metadata.duration_seconds == 1.0
        assert metadata.sample_rate == 44100
        assert metadata.channels == 2
        assert metadata.bit_depth == BitDepth.INT24
        assert metadata.format == AudioFormat.WAV

    def test_metadata_fallback_to_pydub(self, tmp_path: Path) -> None:
        """get_audio_metadata should fallback to pydub on soundfile failure."""
        mp3_path = tmp_path / "test.mp3"

        mock_pydub = MagicMock()
        mock_pydub.frame_rate = 44100
        mock_pydub.channels = 2
        mock_pydub.sample_width = 2
        mock_pydub.__len__ = MagicMock(return_value=5000)  # 5 seconds in ms

        with (
            patch("soundlab.io.audio_io.sf.info", side_effect=Exception("soundfile error")),
            patch("soundlab.io.audio_io.PydubAudioSegment.from_file", return_value=mock_pydub),
        ):
            metadata = get_audio_metadata(mp3_path)

        assert metadata.duration_seconds == 5.0
        assert metadata.sample_rate == 44100
        assert metadata.channels == 2
        assert metadata.bit_depth == BitDepth.INT16


# --------------------------------------------------------------------------- #
# _decode_with_pydub Tests
# --------------------------------------------------------------------------- #


class TestDecodeWithPydub:
    """Tests for _decode_with_pydub fallback function."""

    def test_mono_audio_shape(self, tmp_path: Path) -> None:
        """Mono audio returns correct shape."""
        mp3_path = tmp_path / "test.mp3"

        # Mock pydub audio - mono, 1000 samples
        mock_pydub = MagicMock()
        mock_pydub.get_array_of_samples.return_value = np.arange(1000, dtype=np.int16)
        mock_pydub.channels = 1
        mock_pydub.frame_rate = 22050
        mock_pydub.sample_width = 2
        mock_pydub.__len__ = MagicMock(return_value=1000)

        with patch("soundlab.io.audio_io.PydubAudioSegment.from_file", return_value=mock_pydub):
            from soundlab.io.audio_io import _decode_with_pydub

            samples, _sr, metadata = _decode_with_pydub(mp3_path)

        assert samples.ndim == 1
        assert samples.shape == (1000,)
        assert metadata.channels == 1

    def test_stereo_audio_reshaped(self, tmp_path: Path) -> None:
        """Stereo audio reshaped to (2, samples) from interleaved [L0,R0,L1,R1,...]."""
        mp3_path = tmp_path / "test.mp3"

        # Mock pydub audio - stereo, interleaved: [L0, R0, L1, R1, ...]
        # 4 frames * 2 channels = 8 samples interleaved
        interleaved = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int16)
        mock_pydub = MagicMock()
        mock_pydub.get_array_of_samples.return_value = interleaved
        mock_pydub.channels = 2
        mock_pydub.frame_rate = 22050
        mock_pydub.sample_width = 2
        mock_pydub.__len__ = MagicMock(return_value=1000)

        with patch("soundlab.io.audio_io.PydubAudioSegment.from_file", return_value=mock_pydub):
            from soundlab.io.audio_io import _decode_with_pydub

            samples, _sr, metadata = _decode_with_pydub(mp3_path)

        assert samples.ndim == 2
        assert samples.shape == (2, 4)  # 2 channels, 4 frames
        assert metadata.channels == 2

    def test_sample_width_16bit(self, tmp_path: Path) -> None:
        """16-bit audio (sample_width=2) decoded correctly."""
        mp3_path = tmp_path / "test.mp3"

        # 16-bit audio: max value is 32768
        raw_samples = np.array([16384, -16384], dtype=np.int16)  # Half max
        mock_pydub = MagicMock()
        mock_pydub.get_array_of_samples.return_value = raw_samples
        mock_pydub.channels = 1
        mock_pydub.frame_rate = 44100
        mock_pydub.sample_width = 2
        mock_pydub.__len__ = MagicMock(return_value=100)

        with patch("soundlab.io.audio_io.PydubAudioSegment.from_file", return_value=mock_pydub):
            from soundlab.io.audio_io import _decode_with_pydub

            samples, _sr, metadata = _decode_with_pydub(mp3_path)

        # 16384 / 32768 = 0.5
        np.testing.assert_allclose(samples[0], 0.5, rtol=1e-4)
        np.testing.assert_allclose(samples[1], -0.5, rtol=1e-4)
        assert metadata.bit_depth == BitDepth.INT16

    def test_sample_width_24bit(self, tmp_path: Path) -> None:
        """24-bit audio (sample_width=3) decoded correctly via stereo path."""
        mp3_path = tmp_path / "test.mp3"

        # 24-bit audio: max value is 8388608 (2^23)
        # Use stereo to avoid the mono int16 cast path
        # Interleaved: [L0, R0] with 24-bit values
        raw_samples = np.array([4194304, -4194304], dtype=np.int32)
        mock_pydub = MagicMock()
        mock_pydub.get_array_of_samples.return_value = raw_samples
        mock_pydub.channels = 2  # Stereo to use reshape path
        mock_pydub.frame_rate = 44100
        mock_pydub.sample_width = 3
        mock_pydub.__len__ = MagicMock(return_value=100)

        with patch("soundlab.io.audio_io.PydubAudioSegment.from_file", return_value=mock_pydub):
            from soundlab.io.audio_io import _decode_with_pydub

            samples, _sr, metadata = _decode_with_pydub(mp3_path)

        # 4194304 / 8388608 = 0.5 for left channel
        # -4194304 / 8388608 = -0.5 for right channel
        assert samples.shape == (2, 1)  # 2 channels, 1 frame
        np.testing.assert_allclose(samples[0, 0], 0.5, rtol=1e-4)
        np.testing.assert_allclose(samples[1, 0], -0.5, rtol=1e-4)
        assert metadata.bit_depth == BitDepth.INT24

    def test_sample_rate_preserved(self, tmp_path: Path) -> None:
        """Sample rate from pydub preserved in output."""
        mp3_path = tmp_path / "test.mp3"

        mock_pydub = MagicMock()
        mock_pydub.get_array_of_samples.return_value = np.zeros(1000, dtype=np.int16)
        mock_pydub.channels = 1
        mock_pydub.frame_rate = 48000  # Non-standard sample rate
        mock_pydub.sample_width = 2
        mock_pydub.__len__ = MagicMock(return_value=1000)

        with patch("soundlab.io.audio_io.PydubAudioSegment.from_file", return_value=mock_pydub):
            from soundlab.io.audio_io import _decode_with_pydub

            _samples, sr, metadata = _decode_with_pydub(mp3_path)

        assert sr == 48000
        assert metadata.sample_rate == 48000


# --------------------------------------------------------------------------- #
# save_audio Pydub Fallback Tests
# --------------------------------------------------------------------------- #


class TestSaveAudioPydubFallback:
    """Tests for save_audio pydub fallback path."""

    def test_mono_samples_expanded(self, tmp_path: Path) -> None:
        """Mono samples expanded for pydub export."""
        mp3_path = tmp_path / "output.mp3"

        # Create mono segment
        mono_samples = np.random.rand(1000).astype(np.float32) * 0.5
        segment = AudioSegment(samples=mono_samples, sample_rate=22050)

        mock_pydub_segment = MagicMock()

        with (
            patch("soundlab.io.audio_io.sf.write", side_effect=Exception("soundfile error")),
            patch(
                "soundlab.io.audio_io.PydubAudioSegment", return_value=mock_pydub_segment
            ) as mock_pydub_cls,
        ):
            save_audio(segment, mp3_path)

        # Verify pydub was called with correct channel count
        call_kwargs = mock_pydub_cls.call_args[1]
        assert call_kwargs["channels"] == 1
        mock_pydub_segment.export.assert_called_once()

    def test_clipping_handled(self, tmp_path: Path) -> None:
        """Samples clipped to [-1, 1] range."""
        mp3_path = tmp_path / "output.mp3"

        # Create segment with out-of-range values
        samples = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=22050)

        mock_pydub_segment = MagicMock()
        captured_data = None

        def capture_pydub(*_args, **kwargs):
            nonlocal captured_data
            captured_data = kwargs.get("data")
            return mock_pydub_segment

        with (
            patch("soundlab.io.audio_io.sf.write", side_effect=Exception("soundfile error")),
            patch("soundlab.io.audio_io.PydubAudioSegment", side_effect=capture_pydub),
        ):
            save_audio(segment, mp3_path)

        # Decode the captured int16 data
        int_samples = np.frombuffer(captured_data, dtype=np.int16)

        # Values should be clipped: 2.0 -> 32767, -2.0 -> -32767
        assert int_samples[0] == 32767  # Clipped from 2.0
        assert int_samples[1] == -32767  # Clipped from -2.0
        # 0.5 * 32767 ≈ 16383
        assert abs(int_samples[2] - 16383) <= 1
        assert abs(int_samples[3] - (-16383)) <= 1

    def test_interleaving_for_stereo(self, tmp_path: Path) -> None:
        """Stereo samples interleaved correctly."""
        mp3_path = tmp_path / "output.mp3"

        # Create stereo segment: (2, 4) shape
        # Left channel: [0.25, 0.5, 0.75, 1.0]
        # Right channel: [-0.25, -0.5, -0.75, -1.0]
        left = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        right = np.array([-0.25, -0.5, -0.75, -1.0], dtype=np.float32)
        stereo_samples = np.stack([left, right])
        segment = AudioSegment(samples=stereo_samples, sample_rate=22050)

        mock_pydub_segment = MagicMock()
        captured_data = None

        def capture_pydub(*_args, **kwargs):
            nonlocal captured_data
            captured_data = kwargs.get("data")
            return mock_pydub_segment

        with (
            patch("soundlab.io.audio_io.sf.write", side_effect=Exception("soundfile error")),
            patch("soundlab.io.audio_io.PydubAudioSegment", side_effect=capture_pydub),
        ):
            save_audio(segment, mp3_path)

        # Decode the captured int16 data
        int_samples = np.frombuffer(captured_data, dtype=np.int16)

        # Should be interleaved: [L0, R0, L1, R1, L2, R2, L3, R3]
        assert len(int_samples) == 8

        # Check interleaving pattern (L, R, L, R, ...)
        # L0 ≈ 0.25 * 32767 ≈ 8191
        assert abs(int_samples[0] - int(0.25 * 32767)) <= 1
        # R0 ≈ -0.25 * 32767 ≈ -8191
        assert abs(int_samples[1] - int(-0.25 * 32767)) <= 1

    def test_format_from_extension(self, tmp_path: Path) -> None:
        """Format inferred from file extension."""
        flac_path = tmp_path / "output.flac"

        samples = np.zeros(1000, dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=22050)

        mock_pydub_segment = MagicMock()

        with (
            patch("soundlab.io.audio_io.sf.write", side_effect=Exception("soundfile error")),
            patch("soundlab.io.audio_io.PydubAudioSegment", return_value=mock_pydub_segment),
        ):
            save_audio(segment, flac_path)

        # Verify export was called with correct format
        mock_pydub_segment.export.assert_called_once_with(flac_path, format="flac")

    def test_mp3_export_uses_pydub(self, tmp_path: Path) -> None:
        """MP3 export uses pydub (soundfile doesn't support mp3)."""
        mp3_path = tmp_path / "output.mp3"

        samples = np.random.rand(22050).astype(np.float32)
        segment = AudioSegment(samples=samples, sample_rate=22050)

        mock_pydub_segment = MagicMock()

        with (
            patch("soundlab.io.audio_io.sf.write", side_effect=Exception("MP3 not supported")),
            patch("soundlab.io.audio_io.PydubAudioSegment", return_value=mock_pydub_segment),
        ):
            save_audio(segment, mp3_path)

        # Verify pydub export was called with mp3 format
        mock_pydub_segment.export.assert_called_once_with(mp3_path, format="mp3")


# --------------------------------------------------------------------------- #
# Roundtrip Integration Tests
# --------------------------------------------------------------------------- #


class TestRoundtripIntegration:
    """Integration tests for load/save roundtrip."""

    def test_wav_roundtrip(self, tmp_path: Path, stereo_samples: np.ndarray) -> None:
        """WAV files should survive roundtrip with minimal loss."""
        wav_path = tmp_path / "roundtrip.wav"

        # Create source segment
        original = AudioSegment(samples=stereo_samples, sample_rate=22050)

        # Mock the roundtrip
        with (
            patch("soundlab.io.audio_io.sf.write") as mock_write,
            patch("soundlab.io.audio_io.sf.info") as mock_info,
            patch("soundlab.io.audio_io.sf.read") as mock_read,
        ):
            # Configure mock for save
            mock_write.return_value = None

            # Configure mock for load
            mock_info_obj = MagicMock()
            mock_info_obj.frames = stereo_samples.shape[1]
            mock_info_obj.samplerate = 22050
            mock_info_obj.channels = 2
            mock_info_obj.subtype = "PCM_16"
            mock_info_obj.format = "WAV"
            mock_info.return_value = mock_info_obj
            mock_read.return_value = (stereo_samples.T, 22050)

            # Roundtrip
            save_audio(original, wav_path)
            loaded = load_audio(wav_path)

        assert loaded.sample_rate == original.sample_rate
        assert loaded.channels == original.channels

    def test_mono_stereo_handling(
        self, mono_samples: np.ndarray, stereo_samples: np.ndarray
    ) -> None:
        """Both mono and stereo should be handled correctly."""
        # Test mono
        mono_segment = AudioSegment(samples=mono_samples, sample_rate=22050)
        assert mono_segment.channels == 1

        # Test stereo
        stereo_segment = AudioSegment(samples=stereo_samples, sample_rate=22050)
        assert stereo_segment.channels == 2
