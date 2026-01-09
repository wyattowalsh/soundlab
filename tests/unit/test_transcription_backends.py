"""Tests for transcription backends (CREPE and Drum)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("pydantic")
pydantic = pytest.importorskip("pydantic")

from soundlab.transcription.models import DrumTranscriptionConfig


# --------------------------------------------------------------------------- #
# DrumTranscriptionConfig Tests
# --------------------------------------------------------------------------- #


class TestDrumTranscriptionConfig:
    """Tests for DrumTranscriptionConfig model."""

    def test_drum_config_defaults(self) -> None:
        """DrumTranscriptionConfig should have sensible defaults."""
        config = DrumTranscriptionConfig()

        # Onset detection defaults
        assert config.onset_threshold == 0.3
        assert config.min_note_length == 0.02
        assert config.velocity_scale == 1.2

        # General MIDI drum mapping defaults
        assert config.kick_note == 36
        assert config.snare_note == 38
        assert config.hihat_closed_note == 42
        assert config.hihat_open_note == 46
        assert config.tom_low_note == 45
        assert config.tom_mid_note == 47
        assert config.tom_high_note == 50

        # Spectral threshold defaults
        assert config.kick_max_centroid == 300.0
        assert config.snare_max_centroid == 1000.0

    def test_drum_config_validation_onset_threshold_bounds(self) -> None:
        """onset_threshold should be bounded [0.1, 0.9]."""
        with pytest.raises(pydantic.ValidationError):
            DrumTranscriptionConfig(onset_threshold=0.05)

        with pytest.raises(pydantic.ValidationError):
            DrumTranscriptionConfig(onset_threshold=0.95)

        config_min = DrumTranscriptionConfig(onset_threshold=0.1)
        config_max = DrumTranscriptionConfig(onset_threshold=0.9)

        assert config_min.onset_threshold == 0.1
        assert config_max.onset_threshold == 0.9

    def test_drum_config_validation_min_note_length_bounds(self) -> None:
        """min_note_length should be bounded [0.01, 0.2]."""
        with pytest.raises(pydantic.ValidationError):
            DrumTranscriptionConfig(min_note_length=0.005)

        with pytest.raises(pydantic.ValidationError):
            DrumTranscriptionConfig(min_note_length=0.3)

        config_min = DrumTranscriptionConfig(min_note_length=0.01)
        config_max = DrumTranscriptionConfig(min_note_length=0.2)

        assert config_min.min_note_length == 0.01
        assert config_max.min_note_length == 0.2

    def test_drum_config_validation_velocity_scale_bounds(self) -> None:
        """velocity_scale should be bounded [0.5, 2.0]."""
        with pytest.raises(pydantic.ValidationError):
            DrumTranscriptionConfig(velocity_scale=0.3)

        with pytest.raises(pydantic.ValidationError):
            DrumTranscriptionConfig(velocity_scale=2.5)

        config_min = DrumTranscriptionConfig(velocity_scale=0.5)
        config_max = DrumTranscriptionConfig(velocity_scale=2.0)

        assert config_min.velocity_scale == 0.5
        assert config_max.velocity_scale == 2.0

    def test_drum_config_validation_note_bounds(self) -> None:
        """Note values should be bounded [0, 127]."""
        with pytest.raises(pydantic.ValidationError):
            DrumTranscriptionConfig(kick_note=-1)

        with pytest.raises(pydantic.ValidationError):
            DrumTranscriptionConfig(kick_note=128)

        with pytest.raises(pydantic.ValidationError):
            DrumTranscriptionConfig(snare_note=128)

        config = DrumTranscriptionConfig(kick_note=0, snare_note=127)
        assert config.kick_note == 0
        assert config.snare_note == 127

    def test_drum_config_frozen(self) -> None:
        """DrumTranscriptionConfig should be immutable."""
        config = DrumTranscriptionConfig()

        with pytest.raises(pydantic.ValidationError):
            config.onset_threshold = 0.5  # type: ignore[misc]

        with pytest.raises(pydantic.ValidationError):
            config.kick_note = 40  # type: ignore[misc]

    def test_drum_config_custom_values(self) -> None:
        """DrumTranscriptionConfig should accept custom values."""
        config = DrumTranscriptionConfig(
            onset_threshold=0.4,
            min_note_length=0.05,
            velocity_scale=1.5,
            kick_note=35,
            snare_note=40,
            kick_max_centroid=250.0,
            snare_max_centroid=800.0,
        )

        assert config.onset_threshold == 0.4
        assert config.min_note_length == 0.05
        assert config.velocity_scale == 1.5
        assert config.kick_note == 35
        assert config.snare_note == 40
        assert config.kick_max_centroid == 250.0
        assert config.snare_max_centroid == 800.0


# --------------------------------------------------------------------------- #
# DrumTranscriber Tests (with mocked audio)
# --------------------------------------------------------------------------- #


class TestDrumTranscriber:
    """Tests for DrumTranscriber backend."""

    @pytest.fixture
    def mock_librosa(self) -> MagicMock:
        """Create a mock librosa module."""
        mock = MagicMock()
        # Mock audio loading - returns mono audio and sample rate
        mock.load.return_value = (np.random.randn(44100).astype(np.float32), 44100)
        # Mock onset strength
        mock.onset.onset_strength.return_value = np.array([0.5, 0.8, 0.3, 0.9])
        mock.time_to_frames.return_value = 0
        # Mock spectral centroid
        mock.feature.spectral_centroid.return_value = np.array([[500.0]])
        return mock

    @pytest.fixture
    def mock_detect_onsets(self) -> MagicMock:
        """Create a mock detect_onsets function."""
        mock_result = MagicMock()
        mock_result.timestamps = [0.1, 0.3, 0.5, 0.7]
        return MagicMock(return_value=mock_result)

    def test_drum_transcriber_creates_midi(
        self, mock_librosa: MagicMock, mock_detect_onsets: MagicMock, tmp_path: Path
    ) -> None:
        """DrumTranscriber should create MIDI output."""
        from soundlab.transcription.drum_backend import DrumConfig, DrumTranscriber

        # Create a dummy input file
        audio_file = tmp_path / "drums.wav"
        audio_file.touch()

        output_dir = tmp_path / "output"

        with (
            patch.dict(
                "sys.modules",
                {"librosa": mock_librosa, "librosa.onset": mock_librosa.onset, "librosa.feature": mock_librosa.feature},
            ),
            patch("soundlab.transcription.drum_backend._load_librosa", return_value=mock_librosa),
            patch("soundlab.transcription.drum_backend.detect_onsets", mock_detect_onsets),
            patch("soundlab.transcription.drum_backend.save_midi") as mock_save_midi,
        ):
            transcriber = DrumTranscriber(config=DrumConfig())
            result = transcriber.transcribe(audio_file, output_dir)

            # Should have created notes
            assert len(result.notes) == 4
            # Should have tried to save MIDI
            mock_save_midi.assert_called_once()
            # Path should be set
            assert result.path.suffix == ".mid"
            assert "drums" in result.path.stem

    def test_drum_transcriber_onset_to_velocity(self) -> None:
        """Test velocity mapping from onset strength."""
        from soundlab.transcription.drum_backend import _normalize_strengths_to_velocities

        # Test empty list
        assert _normalize_strengths_to_velocities([]) == []

        # Test uniform strengths
        strengths = [1.0, 1.0, 1.0]
        velocities = _normalize_strengths_to_velocities(strengths)
        assert all(v == 127 for v in velocities)

        # Test varying strengths
        strengths = [0.0, 0.5, 1.0]
        velocities = _normalize_strengths_to_velocities(strengths)
        assert velocities[0] == 1  # Minimum velocity
        assert velocities[1] == 64  # Mid-range
        assert velocities[2] == 127  # Maximum velocity

        # Test all zeros
        strengths = [0.0, 0.0, 0.0]
        velocities = _normalize_strengths_to_velocities(strengths)
        assert all(v == 64 for v in velocities)  # Default mid-velocity

    @pytest.mark.parametrize(
        "centroid,expected_pitch,description",
        [
            (100.0, 36, "kick_low_centroid"),  # Low centroid = kick
            (200.0, 36, "kick_mid_low_centroid"),  # Still kick
            (400.0, 38, "snare_mid_centroid"),  # Mid centroid = snare
            (800.0, 38, "snare_high_mid_centroid"),  # Still snare
            (1500.0, 42, "hihat_high_centroid"),  # High centroid = hi-hat
            (3000.0, 42, "hihat_very_high_centroid"),  # Very high = hi-hat
        ],
    )
    def test_drum_classification_by_centroid(
        self, centroid: float, expected_pitch: int, description: str
    ) -> None:
        """Test drum hit classification based on spectral centroid."""
        from soundlab.transcription.drum_backend import DrumConfig, _classify_drum_hit

        config = DrumConfig()
        pitch = _classify_drum_hit(centroid, config)

        assert pitch == expected_pitch, f"Failed for {description}"

    def test_drum_classification_custom_thresholds(self) -> None:
        """Test drum classification with custom centroid thresholds."""
        from soundlab.transcription.drum_backend import DrumConfig, _classify_drum_hit

        config = DrumConfig(
            low_centroid_max=200.0,
            mid_centroid_max=600.0,
        )

        # With narrower thresholds
        assert _classify_drum_hit(150.0, config) == config.kick_pitch
        assert _classify_drum_hit(400.0, config) == config.snare_pitch
        assert _classify_drum_hit(800.0, config) == config.hihat_pitch

    def test_drum_transcriber_empty_onsets(
        self, mock_librosa: MagicMock, tmp_path: Path
    ) -> None:
        """DrumTranscriber should handle empty onset detection."""
        from soundlab.transcription.drum_backend import DrumTranscriber

        # Create a mock that returns no onsets
        mock_empty_onsets = MagicMock()
        mock_empty_onsets.return_value.timestamps = []

        audio_file = tmp_path / "silence.wav"
        audio_file.touch()

        output_dir = tmp_path / "output"

        with (
            patch("soundlab.transcription.drum_backend._load_librosa", return_value=mock_librosa),
            patch("soundlab.transcription.drum_backend.detect_onsets", mock_empty_onsets),
            patch("soundlab.transcription.drum_backend.save_midi"),
        ):
            transcriber = DrumTranscriber()
            result = transcriber.transcribe(audio_file, output_dir)

            assert len(result.notes) == 0
            assert result.path is not None

    def test_clamp_velocity_function(self) -> None:
        """Test velocity clamping function."""
        from soundlab.transcription.drum_backend import _clamp_velocity

        assert _clamp_velocity(-10) == 1
        assert _clamp_velocity(0) == 1  # Drum backend uses 1 as minimum
        assert _clamp_velocity(64) == 64
        assert _clamp_velocity(127) == 127
        assert _clamp_velocity(200) == 127
        assert _clamp_velocity(100.5) == 100  # Rounds (round() uses banker's rounding)


# --------------------------------------------------------------------------- #
# CREPETranscriber Tests (with mocked CREPE)
# --------------------------------------------------------------------------- #


class TestCREPETranscriber:
    """Tests for CREPETranscriber backend."""

    def test_crepe_transcriber_import_error(self) -> None:
        """Test graceful handling when CREPE is not installed."""
        from soundlab.transcription.crepe_backend import _load_crepe

        with patch("importlib.import_module", side_effect=ImportError("No module named 'crepe'")):
            with pytest.raises(ImportError) as exc_info:
                _load_crepe()

            assert "crepe is required" in str(exc_info.value)
            assert "pip install crepe" in str(exc_info.value)

    def test_crepe_librosa_import_error(self) -> None:
        """Test graceful handling when librosa is not installed."""
        from soundlab.transcription.crepe_backend import _load_librosa

        with patch("importlib.import_module", side_effect=ImportError("No module named 'librosa'")):
            with pytest.raises(ImportError) as exc_info:
                _load_librosa()

            assert "librosa is required" in str(exc_info.value)

    @pytest.mark.parametrize(
        "frequency,expected_midi",
        [
            (440.0, 69),  # A4
            (261.63, 60),  # Middle C (C4)
            (880.0, 81),  # A5
            (220.0, 57),  # A3
            (0.0, 0),  # Zero frequency
            (-100.0, 0),  # Negative frequency
            (20000.0, 127),  # Very high frequency (should clamp)
            (8.176, 0),  # Very low (C-1)
        ],
    )
    def test_crepe_frequency_to_midi(self, frequency: float, expected_midi: int) -> None:
        """Test frequency to MIDI pitch conversion."""
        from soundlab.transcription.crepe_backend import _freq_to_midi

        result = _freq_to_midi(frequency)
        # Allow for rounding differences (within 1 MIDI note)
        assert abs(result - expected_midi) <= 1, f"freq={frequency}, got {result}, expected {expected_midi}"

    def test_crepe_frequency_to_midi_specific_values(self) -> None:
        """Test specific frequency to MIDI conversions."""
        from soundlab.transcription.crepe_backend import _freq_to_midi

        # A4 = 440 Hz = MIDI 69 (exact)
        assert _freq_to_midi(440.0) == 69

        # A3 = 220 Hz = MIDI 57 (exact)
        assert _freq_to_midi(220.0) == 57

        # Test clamping at boundaries
        assert _freq_to_midi(0.0) == 0
        assert _freq_to_midi(-1.0) == 0

    @pytest.fixture
    def mock_crepe_module(self) -> MagicMock:
        """Create a mock CREPE module."""
        mock = MagicMock()
        # Mock predict to return time, frequency, confidence, activation
        mock.predict.return_value = (
            np.array([0.0, 0.1, 0.2]),
            np.array([440.0, 442.0, 438.0]),  # Frequencies around A4
            np.array([0.9, 0.85, 0.88]),  # High confidence
            np.zeros((3, 360)),  # Activation (not used)
        )
        return mock

    @pytest.fixture
    def mock_librosa_for_crepe(self) -> MagicMock:
        """Create a mock librosa module for CREPE tests."""
        mock = MagicMock()
        # Mock audio loading
        mock.load.return_value = (np.random.randn(44100).astype(np.float32), 44100)
        # Mock onset detection - ensure onset_frames are within bounds of onset_env
        onset_env = np.array([0.5, 0.8, 0.6, 0.7, 0.4])
        mock.onset.onset_strength.return_value = onset_env
        mock.onset.onset_detect.return_value = np.array([0, 1, 2])  # Indices within onset_env
        mock.frames_to_time.return_value = np.array([0.0, 0.5, 1.0])
        return mock

    def test_crepe_transcriber_creates_midi(
        self,
        mock_crepe_module: MagicMock,
        mock_librosa_for_crepe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CREPETranscriber should create MIDI output."""
        from soundlab.transcription.crepe_backend import CREPETranscriber
        from soundlab.transcription.models import TranscriptionConfig

        audio_file = tmp_path / "melody.wav"
        audio_file.touch()

        output_dir = tmp_path / "output"

        with (
            patch("soundlab.transcription.crepe_backend._load_crepe", return_value=mock_crepe_module),
            patch("soundlab.transcription.crepe_backend._load_librosa", return_value=mock_librosa_for_crepe),
            patch("soundlab.transcription.crepe_backend.save_midi") as mock_save_midi,
        ):
            config = TranscriptionConfig()
            transcriber = CREPETranscriber(config=config)
            result = transcriber.transcribe_full(audio_file, output_dir)

            # Should have called save_midi
            mock_save_midi.assert_called_once()
            # Path should have correct suffix
            assert result.path.suffix == ".mid"
            assert "crepe" in result.path.stem

    def test_crepe_transcriber_no_onsets(
        self,
        mock_crepe_module: MagicMock,
        mock_librosa_for_crepe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CREPETranscriber should handle no onsets detected."""
        from soundlab.transcription.crepe_backend import CREPETranscriber

        # Modify mock to return no onsets
        mock_librosa_for_crepe.onset.onset_detect.return_value = np.array([])
        mock_librosa_for_crepe.frames_to_time.return_value = np.array([])

        audio_file = tmp_path / "silence.wav"
        audio_file.touch()

        output_dir = tmp_path / "output"

        with (
            patch("soundlab.transcription.crepe_backend._load_crepe", return_value=mock_crepe_module),
            patch("soundlab.transcription.crepe_backend._load_librosa", return_value=mock_librosa_for_crepe),
            patch("soundlab.transcription.crepe_backend.save_midi"),
        ):
            transcriber = CREPETranscriber()
            result = transcriber.transcribe_full(audio_file, output_dir)

            assert len(result.notes) == 0
            assert result.processing_time >= 0

    def test_crepe_transcribe_protocol_interface(
        self,
        mock_crepe_module: MagicMock,
        mock_librosa_for_crepe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CREPETranscriber.transcribe should return Path (protocol interface)."""
        from soundlab.transcription.crepe_backend import CREPETranscriber

        audio_file = tmp_path / "melody.wav"
        audio_file.touch()

        output_dir = tmp_path / "output"

        with (
            patch("soundlab.transcription.crepe_backend._load_crepe", return_value=mock_crepe_module),
            patch("soundlab.transcription.crepe_backend._load_librosa", return_value=mock_librosa_for_crepe),
            patch("soundlab.transcription.crepe_backend.save_midi"),
        ):
            transcriber = CREPETranscriber()
            result = transcriber.transcribe(audio_file, output_dir)

            # Protocol interface should return Path
            assert isinstance(result, Path)

    def test_crepe_clamp_velocity(self) -> None:
        """Test CREPE velocity clamping function."""
        from soundlab.transcription.crepe_backend import _clamp_velocity

        assert _clamp_velocity(-10) == 0
        assert _clamp_velocity(0) == 0
        assert _clamp_velocity(64) == 64
        assert _clamp_velocity(127) == 127
        assert _clamp_velocity(200) == 127
        assert _clamp_velocity(100.7) == 101  # Rounds

    def test_crepe_to_mono_conversion(self) -> None:
        """Test audio mono conversion."""
        from soundlab.transcription.crepe_backend import _to_mono

        # Already mono
        mono = np.random.randn(1000).astype(np.float32)
        result = _to_mono(mono)
        assert result.ndim == 1
        assert len(result) == 1000

        # Stereo (channels first)
        stereo_cf = np.random.randn(2, 1000).astype(np.float32)
        result = _to_mono(stereo_cf)
        assert result.ndim == 1
        assert len(result) == 1000

        # Stereo (samples first)
        stereo_sf = np.random.randn(1000, 2).astype(np.float32)
        result = _to_mono(stereo_sf)
        assert result.ndim == 1
        assert len(result) == 1000


# --------------------------------------------------------------------------- #
# Shared Helper Tests
# --------------------------------------------------------------------------- #


class TestTranscriptionBackendHelpers:
    """Tests for shared helper functions."""

    def test_default_midi_path_crepe(self, tmp_path: Path) -> None:
        """Test CREPE default MIDI path generation."""
        from soundlab.transcription.crepe_backend import _default_midi_path

        audio = tmp_path / "test_audio.wav"
        output_dir = tmp_path / "output"

        result = _default_midi_path(audio, output_dir)

        assert result.parent == output_dir
        assert result.stem == "test_audio_crepe"
        assert result.suffix == ".mid"

    def test_default_midi_path_drums(self, tmp_path: Path) -> None:
        """Test drum default MIDI path generation."""
        from soundlab.transcription.drum_backend import _default_midi_path

        audio = tmp_path / "drum_loop.wav"
        output_dir = tmp_path / "output"

        result = _default_midi_path(audio, output_dir)

        assert result.parent == output_dir
        assert result.stem == "drum_loop_drums"
        assert result.suffix == ".mid"


# --------------------------------------------------------------------------- #
# Integration-Style Tests (with full mocking)
# --------------------------------------------------------------------------- #


class TestTranscriptionBackendIntegration:
    """Integration-style tests with fully mocked dependencies."""

    def test_drum_transcriber_full_pipeline(self, tmp_path: Path) -> None:
        """Test full drum transcription pipeline with mocks."""
        from soundlab.transcription.drum_backend import DrumConfig, DrumTranscriber, _normalize_strengths_to_velocities, _classify_drum_hit

        # Test the helper functions work correctly together
        config = DrumConfig()

        # Simulate onset strengths
        strengths = [0.3, 0.8, 0.5, 1.0]
        velocities = _normalize_strengths_to_velocities(strengths)

        assert len(velocities) == 4
        assert all(1 <= v <= 127 for v in velocities)
        assert velocities[3] == 127  # Strongest hit

        # Simulate classification
        centroids = [150.0, 500.0, 1500.0, 200.0]
        pitches = [_classify_drum_hit(c, config) for c in centroids]

        assert pitches[0] == config.kick_pitch  # Low = kick
        assert pitches[1] == config.snare_pitch  # Mid = snare
        assert pitches[2] == config.hihat_pitch  # High = hihat
        assert pitches[3] == config.kick_pitch  # Low = kick

    def test_crepe_transcriber_config_preserved(self, tmp_path: Path) -> None:
        """Test that CREPE transcriber preserves config in result."""
        from soundlab.transcription.crepe_backend import CREPETranscriber
        from soundlab.transcription.models import TranscriptionConfig

        config = TranscriptionConfig(
            onset_thresh=0.6,
            frame_thresh=0.4,
            min_note_length=0.1,
        )

        transcriber = CREPETranscriber(config=config)

        assert transcriber.config.onset_thresh == 0.6
        assert transcriber.config.frame_thresh == 0.4
        assert transcriber.config.min_note_length == 0.1

    def test_drum_transcriber_config_conversion(self) -> None:
        """Test DrumTranscriber config to TranscriptionConfig conversion."""
        from soundlab.transcription.drum_backend import DrumConfig, DrumTranscriber

        drum_config = DrumConfig(
            onset_thresh=0.4,
            note_duration=0.15,
        )

        transcriber = DrumTranscriber(config=drum_config)
        trans_config = transcriber._to_transcription_config()

        # Should map drum config to transcription config
        assert trans_config.onset_thresh == 0.4
        assert trans_config.min_note_length == 0.15
