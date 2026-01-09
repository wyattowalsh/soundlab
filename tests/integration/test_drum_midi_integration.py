"""Integration tests for drum-to-MIDI transcription pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")

from soundlab.analysis.onsets import OnsetResult
from soundlab.io.midi_io import MIDIData, MIDINote, load_midi
from soundlab.transcription.drum_backend import (
    DRUM_HIHAT_CLOSED,
    DRUM_KICK,
    DRUM_SNARE,
    DrumConfig,
    DrumTranscriber,
)
from soundlab.transcription.models import MIDIResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_onset_result() -> OnsetResult:
    """Create sample onset detection result with timing data."""
    return OnsetResult(
        timestamps=[0.0, 0.25, 0.5, 0.75, 1.0],
        count=5,
        strength=0.8,
    )


@pytest.fixture
def mock_librosa() -> MagicMock:
    """Create a mock librosa module."""
    mock = MagicMock()
    # Mock load to return stereo audio (2 channels, 1 second at 44100 Hz)
    mock.load.return_value = (
        np.random.randn(2, 44100).astype(np.float32) * 0.5,
        44100,
    )
    # Mock onset detection functions
    mock.onset.onset_strength.return_value = np.array([0.5, 0.8, 0.6, 0.9, 0.7])
    # time_to_frames should return a scalar when given a scalar onset_time
    # Use side_effect to return sequential frame indices
    frame_indices = iter([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])  # Extra for multiple calls
    mock.time_to_frames.side_effect = lambda onset_time, sr, hop_length: next(frame_indices, 0)
    # Mock spectral centroid to return different values for different calls
    # This allows testing classification into different drum types
    mock.feature.spectral_centroid.return_value = np.array([[500.0]])
    mock.frames_to_time.return_value = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    return mock


@pytest.fixture
def create_test_wav(tmp_path: Path):
    """Factory fixture to create test WAV files."""
    sf = pytest.importorskip("soundfile")

    def _create_wav(
        filename: str = "test_drums.wav",
        duration_samples: int = 44100,
        channels: int = 2,
        sample_rate: int = 44100,
    ) -> Path:
        audio_path = tmp_path / filename
        if channels == 1:
            audio_data = np.random.randn(duration_samples).astype(np.float32) * 0.5
        else:
            audio_data = np.random.randn(duration_samples, channels).astype(np.float32) * 0.5
        sf.write(str(audio_path), audio_data, sample_rate)
        return audio_path

    return _create_wav


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDrumTranscriptionIntegration:
    """End-to-end integration tests for drum-to-MIDI transcription."""

    def test_drum_transcription_end_to_end(
        self,
        tmp_path: Path,
        create_test_wav,
        sample_onset_result: OnsetResult,
        mock_librosa: MagicMock,
    ) -> None:
        """Test full drum transcription pipeline: load audio -> transcribe -> verify MIDI."""
        # Setup paths
        audio_path = create_test_wav("test_drums.wav")
        output_dir = tmp_path / "output"

        with (
            patch(
                "soundlab.transcription.drum_backend._load_librosa",
                return_value=mock_librosa,
            ),
            patch(
                "soundlab.transcription.drum_backend.detect_onsets",
                return_value=sample_onset_result,
            ),
        ):
            transcriber = DrumTranscriber()
            result = transcriber.transcribe(audio_path, output_dir)

        # Verify result type and structure
        assert isinstance(result, MIDIResult)
        assert result.path.exists()
        assert result.path.suffix == ".mid"
        assert result.processing_time >= 0

        # Verify notes were created (one per onset)
        assert len(result.notes) == sample_onset_result.count

        # Verify note timing is reasonable
        for note in result.notes:
            assert note.start >= 0.0
            assert note.end > note.start
            assert 0 <= note.pitch <= 127
            assert 0 <= note.velocity <= 127

    def test_drum_transcription_with_config(
        self,
        tmp_path: Path,
        create_test_wav,
        sample_onset_result: OnsetResult,
        mock_librosa: MagicMock,
    ) -> None:
        """Test drum transcription with custom DrumConfig settings."""
        audio_path = create_test_wav("test_drums.wav")
        output_dir = tmp_path / "output"

        # Custom configuration with different drum pitches
        custom_config = DrumConfig(
            onset_thresh=0.3,
            note_duration=0.15,
            kick_pitch=35,  # Custom kick pitch
            snare_pitch=40,  # Custom snare pitch
            hihat_pitch=44,  # Custom hi-hat pitch
            low_centroid_max=250.0,
            mid_centroid_max=800.0,
        )

        with (
            patch(
                "soundlab.transcription.drum_backend._load_librosa",
                return_value=mock_librosa,
            ),
            patch(
                "soundlab.transcription.drum_backend.detect_onsets",
                return_value=sample_onset_result,
            ),
        ):
            transcriber = DrumTranscriber(config=custom_config)
            result = transcriber.transcribe(audio_path, output_dir)

        # Verify config was applied
        assert transcriber.config.onset_thresh == 0.3
        assert transcriber.config.note_duration == 0.15

        # Verify notes have correct duration based on config
        for note in result.notes:
            expected_duration = custom_config.note_duration
            actual_duration = note.end - note.start
            assert abs(actual_duration - expected_duration) < 0.01

        # Verify pitches are from custom config (one of kick, snare, or hihat)
        valid_pitches = {custom_config.kick_pitch, custom_config.snare_pitch, custom_config.hihat_pitch}
        for note in result.notes:
            assert note.pitch in valid_pitches

    def test_drum_backend_produces_valid_midi(
        self,
        tmp_path: Path,
        create_test_wav,
        sample_onset_result: OnsetResult,
        mock_librosa: MagicMock,
    ) -> None:
        """Test that drum backend output can be loaded by mido and has valid MIDI values."""
        mido = pytest.importorskip("mido")

        audio_path = create_test_wav("test_drums.wav")
        output_dir = tmp_path / "output"

        with (
            patch(
                "soundlab.transcription.drum_backend._load_librosa",
                return_value=mock_librosa,
            ),
            patch(
                "soundlab.transcription.drum_backend.detect_onsets",
                return_value=sample_onset_result,
            ),
        ):
            transcriber = DrumTranscriber()
            result = transcriber.transcribe(audio_path, output_dir)

        # Verify MIDI file can be loaded by mido
        midi_file = mido.MidiFile(str(result.path))
        assert midi_file is not None
        assert len(midi_file.tracks) > 0

        # Collect all note events from MIDI file
        note_on_events = []
        note_off_events = []
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    note_on_events.append(msg)
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    note_off_events.append(msg)

        # Verify note events have valid MIDI values (0-127)
        for msg in note_on_events:
            assert 0 <= msg.note <= 127, f"Invalid note value: {msg.note}"
            assert 0 <= msg.velocity <= 127, f"Invalid velocity: {msg.velocity}"

        # Verify we have the expected number of notes
        assert len(note_on_events) == sample_onset_result.count

    def test_drum_transcription_empty_audio(
        self,
        tmp_path: Path,
        create_test_wav,
        mock_librosa: MagicMock,
    ) -> None:
        """Test drum transcription handles audio with no onsets gracefully."""
        audio_path = create_test_wav("silent.wav")
        output_dir = tmp_path / "output"

        # Empty onset result (no drum hits detected)
        empty_onset_result = OnsetResult(timestamps=[], count=0, strength=0.0)

        with (
            patch(
                "soundlab.transcription.drum_backend._load_librosa",
                return_value=mock_librosa,
            ),
            patch(
                "soundlab.transcription.drum_backend.detect_onsets",
                return_value=empty_onset_result,
            ),
        ):
            transcriber = DrumTranscriber()
            result = transcriber.transcribe(audio_path, output_dir)

        # Should still produce valid output with no notes
        assert isinstance(result, MIDIResult)
        assert result.path.exists()
        assert len(result.notes) == 0

        # MIDI file should still be valid and loadable
        midi_data = load_midi(result.path)
        assert isinstance(midi_data, MIDIData)
        assert len(midi_data.notes) == 0

    def test_drum_transcription_midi_can_be_reloaded(
        self,
        tmp_path: Path,
        create_test_wav,
        sample_onset_result: OnsetResult,
        mock_librosa: MagicMock,
    ) -> None:
        """Test that transcribed MIDI can be reloaded using soundlab's load_midi."""
        audio_path = create_test_wav("test_drums.wav")
        output_dir = tmp_path / "output"

        with (
            patch(
                "soundlab.transcription.drum_backend._load_librosa",
                return_value=mock_librosa,
            ),
            patch(
                "soundlab.transcription.drum_backend.detect_onsets",
                return_value=sample_onset_result,
            ),
        ):
            transcriber = DrumTranscriber()
            result = transcriber.transcribe(audio_path, output_dir)

        # Reload using soundlab's MIDI I/O
        midi_data = load_midi(result.path)

        assert isinstance(midi_data, MIDIData)
        assert len(midi_data.notes) == sample_onset_result.count
        assert midi_data.tempo > 0

        # Verify loaded notes match transcribed notes
        for loaded_note, original_note in zip(midi_data.notes, result.notes):
            assert loaded_note.pitch == original_note.pitch
            assert loaded_note.velocity == original_note.velocity
            # Timing may have slight float differences due to tick conversion
            assert abs(loaded_note.start_seconds - original_note.start) < 0.01


@pytest.mark.slow
class TestDrumClassification:
    """Tests for drum hit classification based on spectral characteristics."""

    def test_kick_classification_low_centroid(
        self,
        tmp_path: Path,
        create_test_wav,
        mock_librosa: MagicMock,
    ) -> None:
        """Test that low spectral centroid classifies as kick drum."""
        audio_path = create_test_wav("test_kick.wav")
        output_dir = tmp_path / "output"

        # Configure mock to return low centroid (kick territory)
        mock_librosa.feature.spectral_centroid.return_value = np.array([[200.0]])

        single_onset = OnsetResult(timestamps=[0.5], count=1, strength=0.8)

        with (
            patch(
                "soundlab.transcription.drum_backend._load_librosa",
                return_value=mock_librosa,
            ),
            patch(
                "soundlab.transcription.drum_backend.detect_onsets",
                return_value=single_onset,
            ),
        ):
            transcriber = DrumTranscriber()
            result = transcriber.transcribe(audio_path, output_dir)

        assert len(result.notes) == 1
        assert result.notes[0].pitch == DRUM_KICK

    def test_snare_classification_mid_centroid(
        self,
        tmp_path: Path,
        create_test_wav,
        mock_librosa: MagicMock,
    ) -> None:
        """Test that mid spectral centroid classifies as snare drum."""
        audio_path = create_test_wav("test_snare.wav")
        output_dir = tmp_path / "output"

        # Configure mock to return mid centroid (snare territory)
        mock_librosa.feature.spectral_centroid.return_value = np.array([[600.0]])

        single_onset = OnsetResult(timestamps=[0.5], count=1, strength=0.8)

        with (
            patch(
                "soundlab.transcription.drum_backend._load_librosa",
                return_value=mock_librosa,
            ),
            patch(
                "soundlab.transcription.drum_backend.detect_onsets",
                return_value=single_onset,
            ),
        ):
            transcriber = DrumTranscriber()
            result = transcriber.transcribe(audio_path, output_dir)

        assert len(result.notes) == 1
        assert result.notes[0].pitch == DRUM_SNARE

    def test_hihat_classification_high_centroid(
        self,
        tmp_path: Path,
        create_test_wav,
        mock_librosa: MagicMock,
    ) -> None:
        """Test that high spectral centroid classifies as hi-hat."""
        audio_path = create_test_wav("test_hihat.wav")
        output_dir = tmp_path / "output"

        # Configure mock to return high centroid (hi-hat territory)
        mock_librosa.feature.spectral_centroid.return_value = np.array([[2000.0]])

        single_onset = OnsetResult(timestamps=[0.5], count=1, strength=0.8)

        with (
            patch(
                "soundlab.transcription.drum_backend._load_librosa",
                return_value=mock_librosa,
            ),
            patch(
                "soundlab.transcription.drum_backend.detect_onsets",
                return_value=single_onset,
            ),
        ):
            transcriber = DrumTranscriber()
            result = transcriber.transcribe(audio_path, output_dir)

        assert len(result.notes) == 1
        assert result.notes[0].pitch == DRUM_HIHAT_CLOSED


class TestDrumConfigValidation:
    """Tests for DrumConfig parameter validation."""

    def test_drum_config_defaults(self) -> None:
        """Test DrumConfig has sensible defaults."""
        config = DrumConfig()

        assert config.onset_thresh == 0.5
        assert config.hop_length == 512
        assert config.note_duration == 0.1
        assert config.kick_pitch == DRUM_KICK
        assert config.snare_pitch == DRUM_SNARE
        assert config.hihat_pitch == DRUM_HIHAT_CLOSED

    def test_drum_config_validation_bounds(self) -> None:
        """Test DrumConfig validates parameter bounds."""
        import pydantic

        # onset_thresh out of bounds
        with pytest.raises(pydantic.ValidationError):
            DrumConfig(onset_thresh=1.5)

        with pytest.raises(pydantic.ValidationError):
            DrumConfig(onset_thresh=0.05)

        # note_duration out of bounds
        with pytest.raises(pydantic.ValidationError):
            DrumConfig(note_duration=0.005)

        with pytest.raises(pydantic.ValidationError):
            DrumConfig(note_duration=1.0)

        # pitch out of MIDI range
        with pytest.raises(pydantic.ValidationError):
            DrumConfig(kick_pitch=128)

        with pytest.raises(pydantic.ValidationError):
            DrumConfig(snare_pitch=-1)

    def test_drum_config_custom_values(self) -> None:
        """Test DrumConfig accepts valid custom values."""
        config = DrumConfig(
            onset_thresh=0.7,
            hop_length=256,
            note_duration=0.05,
            kick_pitch=35,
            snare_pitch=40,
            hihat_pitch=46,
            low_centroid_max=250.0,
            mid_centroid_max=900.0,
        )

        assert config.onset_thresh == 0.7
        assert config.hop_length == 256
        assert config.note_duration == 0.05
        assert config.kick_pitch == 35
        assert config.snare_pitch == 40
        assert config.hihat_pitch == 46


class TestDrumTranscriberUnit:
    """Unit tests for DrumTranscriber class."""

    def test_transcriber_uses_default_config(self) -> None:
        """Test DrumTranscriber uses default config when none provided."""
        transcriber = DrumTranscriber()
        assert transcriber.config is not None
        assert isinstance(transcriber.config, DrumConfig)

    def test_transcriber_uses_provided_config(self) -> None:
        """Test DrumTranscriber uses provided config."""
        custom_config = DrumConfig(onset_thresh=0.3, note_duration=0.2)
        transcriber = DrumTranscriber(config=custom_config)

        assert transcriber.config == custom_config
        assert transcriber.config.onset_thresh == 0.3
        assert transcriber.config.note_duration == 0.2

    def test_midi_output_path_naming(
        self,
        tmp_path: Path,
        create_test_wav,
        mock_librosa: MagicMock,
    ) -> None:
        """Test that MIDI output file is named correctly based on input."""
        audio_path = create_test_wav("my_drum_track.wav")
        output_dir = tmp_path / "output"

        empty_onset = OnsetResult(timestamps=[], count=0, strength=0.0)

        with (
            patch(
                "soundlab.transcription.drum_backend._load_librosa",
                return_value=mock_librosa,
            ),
            patch(
                "soundlab.transcription.drum_backend.detect_onsets",
                return_value=empty_onset,
            ),
        ):
            transcriber = DrumTranscriber()
            result = transcriber.transcribe(audio_path, output_dir)

        # Output should be named {input_stem}_drums.mid
        assert result.path.name == "my_drum_track_drums.mid"
        assert result.path.parent == output_dir
