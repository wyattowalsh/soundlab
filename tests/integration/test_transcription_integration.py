"""Integration tests for MIDI transcription pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")

from soundlab.transcription import MIDIResult, MIDITranscriber, TranscriptionConfig
from soundlab.transcription.models import NoteEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_note_events() -> list[tuple[float, float, int, float]]:
    """Create sample note events (start, end, pitch, velocity)."""
    return [
        (0.0, 0.5, 60, 100.0),  # C4
        (0.5, 1.0, 64, 90.0),  # E4
        (1.0, 1.5, 67, 80.0),  # G4
        (1.5, 2.0, 72, 110.0),  # C5
    ]


@pytest.fixture
def mock_midi_data() -> MagicMock:
    """Create a mock MIDI data object."""
    midi_data = MagicMock()
    midi_data.write = MagicMock()
    return midi_data


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTranscriptionIntegration:
    """End-to-end integration tests for MIDI transcription."""

    def test_transcribe_creates_midi_output(
        self,
        tmp_path: Path,
        sample_note_events: list[tuple[float, float, int, float]],
        mock_midi_data: MagicMock,
    ) -> None:
        """Test full transcription pipeline: load audio -> transcribe -> verify MIDI output."""
        # Setup paths
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        audio_path = input_dir / "test_audio.wav"

        # Create a simple test WAV file
        import soundfile as sf

        audio_data = np.random.randn(44100, 1).astype(np.float32) * 0.5
        sf.write(str(audio_path), audio_data, 44100)

        # Mock basic_pitch inference
        mock_inference = MagicMock()
        mock_inference.predict.return_value = (
            MagicMock(),  # model_output
            mock_midi_data,  # midi_data
            np.array(sample_note_events),  # note_events
        )

        with patch.dict(
            "sys.modules",
            {"basic_pitch": MagicMock(), "basic_pitch.inference": mock_inference},
        ):
            config = TranscriptionConfig()
            transcriber = MIDITranscriber(config=config)
            result = transcriber.transcribe(audio_path, output_dir)

        # Verify result structure
        assert isinstance(result, MIDIResult)
        assert result.config == config
        assert result.processing_time >= 0

        # Verify notes were extracted
        assert len(result.notes) == len(sample_note_events)

    def test_transcribe_with_custom_thresholds(
        self,
        tmp_path: Path,
        sample_note_events: list[tuple[float, float, int, float]],
        mock_midi_data: MagicMock,
    ) -> None:
        """Test transcription with custom onset/frame thresholds."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        audio_path = input_dir / "test_audio.wav"

        import soundfile as sf

        audio_data = np.random.randn(22050, 1).astype(np.float32) * 0.5
        sf.write(str(audio_path), audio_data, 44100)

        mock_inference = MagicMock()
        mock_inference.predict.return_value = (
            MagicMock(),
            mock_midi_data,
            np.array(sample_note_events),
        )

        with patch.dict(
            "sys.modules",
            {"basic_pitch": MagicMock(), "basic_pitch.inference": mock_inference},
        ):
            config = TranscriptionConfig(
                onset_thresh=0.6,
                frame_thresh=0.4,
                min_note_length=0.1,
            )
            transcriber = MIDITranscriber(config=config)
            result = transcriber.transcribe(audio_path, output_dir)

            # Verify thresholds were passed to inference
            mock_inference.predict.assert_called_once()
            call_kwargs = mock_inference.predict.call_args[1]
            assert call_kwargs["onset_threshold"] == 0.6
            assert call_kwargs["frame_threshold"] == 0.4
            assert call_kwargs["minimum_note_length"] == 0.1

        assert isinstance(result, MIDIResult)

    def test_transcribe_extracts_notes_correctly(
        self,
        tmp_path: Path,
        sample_note_events: list[tuple[float, float, int, float]],
        mock_midi_data: MagicMock,
    ) -> None:
        """Test that note events are correctly extracted from transcription output."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        audio_path = input_dir / "test_audio.wav"

        import soundfile as sf

        audio_data = np.random.randn(44100, 1).astype(np.float32) * 0.5
        sf.write(str(audio_path), audio_data, 44100)

        mock_inference = MagicMock()
        mock_inference.predict.return_value = (
            MagicMock(),
            mock_midi_data,
            np.array(sample_note_events),
        )

        with patch.dict(
            "sys.modules",
            {"basic_pitch": MagicMock(), "basic_pitch.inference": mock_inference},
        ):
            transcriber = MIDITranscriber()
            result = transcriber.transcribe(audio_path, output_dir)

        # Verify each note
        assert len(result.notes) == 4
        assert result.notes[0].pitch == 60
        assert result.notes[0].start == 0.0
        assert result.notes[0].end == 0.5
        assert result.notes[1].pitch == 64
        assert result.notes[2].pitch == 67
        assert result.notes[3].pitch == 72

    def test_midi_result_properties(self, tmp_path: Path) -> None:
        """Test MIDIResult fields and note access."""
        notes = [
            NoteEvent(start=0.0, end=0.5, pitch=60, velocity=100),
            NoteEvent(start=0.5, end=1.0, pitch=64, velocity=90),
            NoteEvent(start=1.0, end=1.5, pitch=67, velocity=80),
        ]

        result = MIDIResult(
            notes=notes,
            path=tmp_path / "output.mid",
            config=TranscriptionConfig(),
            processing_time=1.5,
        )

        # Verify notes list
        assert len(result.notes) == 3
        assert result.processing_time == 1.5
        assert result.path == tmp_path / "output.mid"

        # Verify note data access
        pitches = [n.pitch for n in result.notes]
        assert pitches == [60, 64, 67]

    def test_transcription_config_defaults(self) -> None:
        """Test TranscriptionConfig default values."""
        config = TranscriptionConfig()

        assert config.onset_thresh == 0.5
        assert config.frame_thresh == 0.3
        assert config.min_note_length == 0.058
        assert config.min_freq == 32.7
        assert config.max_freq == 2093.0
        assert config.max_freq > config.min_freq

    def test_transcription_config_validation(self) -> None:
        """Test TranscriptionConfig validates bounds correctly."""
        import pydantic

        # onset_thresh out of bounds
        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(onset_thresh=1.5)

        # frame_thresh out of bounds
        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(frame_thresh=-0.1)

        # min_note_length negative
        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(min_note_length=-1.0)

    def test_note_event_ordering(self) -> None:
        """Test NoteEvent enforces start < end constraint."""
        # Valid note
        note = NoteEvent(start=0.0, end=1.0, pitch=60, velocity=100)
        assert note.start == 0.0
        assert note.end == 1.0

        # Invalid: end before start
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            NoteEvent(start=1.0, end=0.5, pitch=60, velocity=100)

    def test_velocity_clamping(self) -> None:
        """Test that velocity values are clamped to valid MIDI range."""
        from soundlab.transcription.basic_pitch import _clamp_velocity

        assert _clamp_velocity(150.0) == 127
        assert _clamp_velocity(-10.0) == 0
        assert _clamp_velocity(64.0) == 64
        assert _clamp_velocity(64.6) == 65  # Rounds

    def test_transcribe_with_predict_and_save_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        """Test transcription using predict_and_save API fallback."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        audio_path = input_dir / "test_audio.wav"

        # Create test WAV
        import soundfile as sf

        audio_data = np.random.randn(22050, 1).astype(np.float32) * 0.5
        sf.write(str(audio_path), audio_data, 44100)

        # Create mock MIDI output file
        midi_path = output_dir / "test_audio.mid"
        midi_path.touch()

        # Mock inference without predict, only predict_and_save
        mock_inference = MagicMock(spec=["predict_and_save"])

        with patch.dict(
            "sys.modules",
            {"basic_pitch": MagicMock(), "basic_pitch.inference": mock_inference},
        ):
            transcriber = MIDITranscriber()
            result = transcriber.transcribe(audio_path, output_dir)

        assert isinstance(result, MIDIResult)
        mock_inference.predict_and_save.assert_called_once()
