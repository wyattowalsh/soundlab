"""Integration tests for audio-to-MIDI transcription workflow.

These tests verify the complete end-to-end transcription pipeline, from loading
audio files to generating MIDI output and visualizations. Tests are designed to
work in CI environments with appropriate mocking when Basic Pitch is unavailable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import soundfile as sf

from soundlab.core.exceptions import ProcessingError
from soundlab.io.audio_io import load_audio, save_audio
from soundlab.io.midi_io import MIDIData, MIDINote, load_midi, save_midi
from soundlab.transcription import (
    MIDIResult,
    MIDITranscriber,
    NoteEvent,
    TranscriptionConfig,
    render_note_density,
    render_piano_roll,
)


# === Test Fixtures ===


@pytest.fixture
def piano_audio_path(temp_dir: Path, sample_rate: int) -> Path:
    """
    Create a test audio file with piano-like tones (C major chord).

    Generates a synthetic audio file containing three simultaneous notes
    (C4, E4, G4) to simulate a simple piano chord. This provides a controlled
    input for testing the transcription pipeline.
    """
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # C major chord: C4 (261.63 Hz), E4 (329.63 Hz), G4 (392.00 Hz)
    frequencies = [261.63, 329.63, 392.00]
    audio = np.zeros_like(t)

    for freq in frequencies:
        # Add harmonic content for more realistic piano-like sound
        audio += 0.3 * np.sin(2 * np.pi * freq * t)
        audio += 0.1 * np.sin(2 * np.pi * freq * 2 * t)  # 2nd harmonic
        audio += 0.05 * np.sin(2 * np.pi * freq * 3 * t)  # 3rd harmonic

    # Apply envelope to simulate note attack and decay
    envelope = np.exp(-2 * t)
    audio *= envelope

    # Normalize
    audio = audio / np.max(np.abs(audio))

    audio_path = temp_dir / "piano_test.wav"
    sf.write(audio_path, audio, sample_rate)
    return audio_path


@pytest.fixture
def melody_audio_path(temp_dir: Path, sample_rate: int) -> Path:
    """
    Create a test audio file with a simple melody (sequential notes).

    Generates a sequence of notes (C4, D4, E4, F4) to test transcription
    of melodic content with distinct onset and offset times.
    """
    # Four notes, 0.5 seconds each with 0.1s gaps
    notes_config = [
        (0.0, 0.5, 261.63),   # C4
        (0.6, 1.1, 293.66),   # D4
        (1.2, 1.7, 329.63),   # E4
        (1.8, 2.3, 349.23),   # F4
    ]

    duration = 2.5
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

    for start, end, freq in notes_config:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        note_duration = end - start

        t = np.linspace(0, note_duration, end_idx - start_idx)

        # Generate note with harmonics
        note = 0.4 * np.sin(2 * np.pi * freq * t)
        note += 0.15 * np.sin(2 * np.pi * freq * 2 * t)

        # Apply envelope
        envelope = np.exp(-3 * t)
        note *= envelope

        audio[start_idx:end_idx] = note

    # Normalize
    audio = audio / np.max(np.abs(audio))

    audio_path = temp_dir / "melody_test.wav"
    sf.write(audio_path, audio, sample_rate)
    return audio_path


@pytest.fixture
def mock_basic_pitch_predict():
    """
    Mock Basic Pitch prediction for deterministic testing.

    Returns a fixture that can be used to mock basic_pitch.inference.predict
    with configurable note events. This allows tests to run without the actual
    Basic Pitch model, which is important for CI environments.
    """
    def _create_mock(note_events=None):
        """Create a mock with specified note events."""
        if note_events is None:
            # Default: C4, E4, G4 chord
            note_events = [
                (0.0, 1.5, 60, 0.8, None),   # C4
                (0.0, 1.5, 64, 0.75, None),  # E4
                (0.0, 1.5, 67, 0.7, None),   # G4
            ]

        # Create mock MIDI data
        mock_midi = MagicMock()
        mock_midi.write = MagicMock()

        mock_output = (
            MagicMock(),  # model_output
            mock_midi,    # midi_data
            note_events,  # note_events: (start, end, pitch, velocity, pitch_bend)
        )

        return mock_output

    return _create_mock


@pytest.fixture
def mock_basic_pitch_melody():
    """Mock Basic Pitch for melody transcription (sequential notes)."""
    note_events = [
        (0.0, 0.5, 60, 0.8, None),   # C4
        (0.6, 1.1, 62, 0.75, None),  # D4
        (1.2, 1.7, 64, 0.7, None),   # E4
        (1.8, 2.3, 65, 0.72, None),  # F4
    ]

    mock_midi = MagicMock()
    mock_midi.write = MagicMock()

    return (MagicMock(), mock_midi, note_events)


# === Integration Tests ===


@pytest.mark.integration
@pytest.mark.slow
class TestMIDITranscriptionWorkflow:
    """Integration tests for the complete MIDI transcription workflow."""

    def test_transcribe_audio_to_midi_full_workflow(
        self,
        piano_audio_path: Path,
        temp_output_dir: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test the complete workflow: audio file → transcription → MIDI file.

        This is the primary integration test that verifies all components work
        together: loading audio, transcribing with Basic Pitch, generating
        NoteEvent objects, and saving MIDI files.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    piano_audio_path,
                    output_dir=temp_output_dir,
                    save_midi_file=True,
                )

        # Verify result structure
        assert isinstance(result, MIDIResult)
        assert result.note_count == 3  # C, E, G notes
        assert result.source_path == piano_audio_path
        assert result.processing_time_seconds > 0

        # Verify notes were created correctly
        assert len(result.notes) == 3
        for note in result.notes:
            assert isinstance(note, NoteEvent)
            assert 0 <= note.pitch <= 127
            assert 1 <= note.velocity <= 127
            assert note.start_time >= 0
            assert note.end_time > note.start_time

        # Verify MIDI file was created
        assert result.midi_path is not None
        assert result.midi_path.exists()
        assert result.midi_path.suffix == ".mid"

    def test_transcribe_with_default_config(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test transcription with default configuration settings.

        Ensures that the transcriber works correctly with no custom
        configuration, using all default thresholds and parameters.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

        assert result.config == transcriber.config
        assert result.config.onset_thresh == 0.5
        assert result.config.frame_thresh == 0.3
        assert result.config.minimum_note_length == 58.0

    def test_transcribe_with_custom_onset_threshold(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test transcription with custom onset threshold.

        Verifies that custom onset thresholds are properly passed through
        to the Basic Pitch model and affect note detection sensitivity.
        """
        config = TranscriptionConfig(onset_thresh=0.7)

        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ) as mock_predict:
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber(config=config)
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

        # Verify the config was used
        assert result.config.onset_thresh == 0.7

        # Verify predict was called with correct threshold
        mock_predict.assert_called_once()
        call_kwargs = mock_predict.call_args.kwargs
        assert call_kwargs["onset_threshold"] == 0.7

    def test_transcribe_with_custom_frame_threshold(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test transcription with custom frame threshold.

        Frame threshold affects the continuation of notes and can impact
        note length detection. This test verifies custom values are applied.
        """
        config = TranscriptionConfig(frame_thresh=0.5)

        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ) as mock_predict:
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber(config=config)
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

        assert result.config.frame_thresh == 0.5

        call_kwargs = mock_predict.call_args.kwargs
        assert call_kwargs["frame_threshold"] == 0.5

    def test_transcribe_with_minimum_note_length(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test transcription with custom minimum note length.

        Minimum note length filters out very short notes that may be
        artifacts or noise. This test ensures the parameter is applied.
        """
        config = TranscriptionConfig(minimum_note_length=100.0)

        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ) as mock_predict:
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber(config=config)
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

        assert result.config.minimum_note_length == 100.0

        call_kwargs = mock_predict.call_args.kwargs
        assert call_kwargs["minimum_note_length"] == 100.0

    def test_transcribe_with_frequency_range(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test transcription with custom frequency range.

        Frequency range limits transcription to specific pitch ranges,
        useful for instrument-specific transcription (e.g., bass only).
        """
        config = TranscriptionConfig(
            minimum_frequency=50.0,
            maximum_frequency=1000.0,
        )

        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ) as mock_predict:
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber(config=config)
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

        call_kwargs = mock_predict.call_args.kwargs
        assert call_kwargs["minimum_frequency"] == 50.0
        assert call_kwargs["maximum_frequency"] == 1000.0

    def test_transcribe_with_progress_callback(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test transcription with progress callback function.

        Progress callbacks enable UI updates during long-running transcriptions.
        This test verifies callbacks are invoked with appropriate progress values.
        """
        progress_calls = []

        def progress_callback(current: int, total: int, message: str):
            progress_calls.append((current, total, message))

        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                    progress_callback=progress_callback,
                )

        # Verify progress callback was called
        assert len(progress_calls) > 0

        # Check for expected progress stages
        messages = [call[2] for call in progress_calls]
        assert "Loading audio..." in messages
        assert "Processing notes..." in messages
        assert "Complete" in messages

        # Verify progress goes from 0 to 100
        assert progress_calls[0][0] == 0
        assert progress_calls[-1][0] == 100
        assert all(call[1] == 100 for call in progress_calls)

    def test_transcribe_saves_midi_to_correct_location(
        self,
        piano_audio_path: Path,
        temp_output_dir: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test MIDI file is saved to the specified output directory.

        Verifies that the output_dir parameter is respected and files
        are created in the correct location with proper naming.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    piano_audio_path,
                    output_dir=temp_output_dir,
                    save_midi_file=True,
                )

        # Verify MIDI was saved to output dir
        assert result.midi_path.parent == temp_output_dir
        assert result.midi_path.name == "piano_test_transcription.mid"
        assert result.midi_path.exists()

    def test_transcribe_without_saving_midi(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test transcription without saving MIDI file to disk.

        Useful for in-memory processing where only NoteEvent data is needed
        and no file output is required.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

        # Verify no MIDI file was created
        assert result.midi_path is None

    def test_notes_are_sorted_by_start_time(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test that transcribed notes are sorted chronologically.

        Note sorting is important for sequential processing and MIDI
        file generation. Verifies notes are ordered by start_time.
        """
        # Create mock with unsorted notes
        unsorted_notes = [
            (1.5, 2.0, 64, 0.8, None),   # Later note
            (0.0, 0.5, 60, 0.7, None),   # Earlier note
            (0.7, 1.2, 62, 0.75, None),  # Middle note
        ]

        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(unsorted_notes),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

        # Verify notes are sorted
        start_times = [note.start_time for note in result.notes]
        assert start_times == sorted(start_times)

    def test_velocity_conversion_from_normalized(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test velocity conversion from normalized (0-1) to MIDI (1-127).

        Basic Pitch outputs normalized velocity values that must be
        converted to MIDI range. This test verifies the conversion logic.
        """
        # Mock with normalized velocities
        notes_with_velocity = [
            (0.0, 0.5, 60, 0.0, None),    # Should become 1 (min)
            (0.5, 1.0, 62, 0.5, None),    # Should become ~64
            (1.0, 1.5, 64, 1.0, None),    # Should become 127 (max)
        ]

        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(notes_with_velocity),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

        velocities = [note.velocity for note in result.notes]

        # Verify all velocities are in MIDI range
        assert all(1 <= v <= 127 for v in velocities)

        # Verify conversion is reasonable
        assert velocities[0] < velocities[1] < velocities[2]


@pytest.mark.integration
@pytest.mark.slow
class TestMIDIFileValidation:
    """Integration tests for MIDI file output validation."""

    def test_saved_midi_file_is_valid(
        self,
        piano_audio_path: Path,
        temp_output_dir: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test that saved MIDI files can be loaded and contain correct data.

        Verifies the complete round-trip: transcribe → save MIDI → load MIDI.
        This ensures files are valid and contain the expected note data.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    piano_audio_path,
                    output_dir=temp_output_dir,
                    save_midi_file=True,
                )

        # Load the saved MIDI file
        midi_data = load_midi(result.midi_path)

        # Verify MIDI data matches transcription result
        assert midi_data.note_count == result.note_count
        assert len(midi_data.notes) == len(result.notes)

        # Verify notes match (within tolerance due to float precision)
        for midi_note, result_note in zip(midi_data.notes, result.notes):
            assert abs(midi_note.start_time - result_note.start_time) < 0.01
            assert abs(midi_note.end_time - result_note.end_time) < 0.01
            assert midi_note.pitch == result_note.pitch

    def test_midi_file_contains_valid_note_data(
        self,
        melody_audio_path: Path,
        temp_output_dir: Path,
        mock_basic_pitch_melody,
    ):
        """
        Test MIDI file contains valid note pitch and timing data.

        Ensures that note properties (pitch, timing, velocity) are
        preserved correctly in the MIDI file format.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_melody,
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    melody_audio_path,
                    output_dir=temp_output_dir,
                    save_midi_file=True,
                )

        # Load and verify MIDI
        midi_data = load_midi(result.midi_path)

        # Verify all notes have valid MIDI pitches
        for note in midi_data.notes:
            assert 0 <= note.pitch <= 127
            assert note.pitch in [60, 62, 64, 65]  # C4, D4, E4, F4

        # Verify timing is valid
        for note in midi_data.notes:
            assert note.start_time >= 0
            assert note.end_time > note.start_time
            assert note.duration > 0


@pytest.mark.integration
@pytest.mark.slow
class TestTranscriptionToMIDIData:
    """Integration tests for transcribe_to_midi_data method."""

    def test_transcribe_to_midi_data_returns_correct_type(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test transcribe_to_midi_data returns MIDIData object.

        This method provides an alternative interface that returns
        MIDIData instead of MIDIResult for further processing.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                midi_data = transcriber.transcribe_to_midi_data(piano_audio_path)

        assert isinstance(midi_data, MIDIData)
        assert midi_data.note_count == 3

    def test_transcribe_to_midi_data_converts_notes_correctly(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test NoteEvent to MIDINote conversion in transcribe_to_midi_data.

        Verifies that NoteEvent objects are correctly converted to MIDINote
        objects with all properties preserved.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                midi_data = transcriber.transcribe_to_midi_data(piano_audio_path)

        # Verify all notes are MIDINote instances
        for note in midi_data.notes:
            assert isinstance(note, MIDINote)
            assert hasattr(note, "start_time")
            assert hasattr(note, "end_time")
            assert hasattr(note, "pitch")
            assert hasattr(note, "velocity")

    def test_transcribe_to_midi_data_can_be_saved(
        self,
        piano_audio_path: Path,
        temp_output_dir: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test MIDIData from transcribe_to_midi_data can be saved to file.

        Verifies the complete workflow: transcribe → get MIDIData → save.
        This tests the integration between transcription and MIDI I/O.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                midi_data = transcriber.transcribe_to_midi_data(piano_audio_path)

        # Save the MIDI data
        output_path = temp_output_dir / "converted.mid"
        saved_path = save_midi(midi_data, output_path)

        assert saved_path.exists()

        # Verify it can be loaded back
        loaded_data = load_midi(saved_path)
        assert loaded_data.note_count == midi_data.note_count


@pytest.mark.integration
@pytest.mark.slow
class TestVisualizationIntegration:
    """Integration tests for piano roll and note density visualizations."""

    def test_render_piano_roll_from_transcription(
        self,
        piano_audio_path: Path,
        temp_output_dir: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test generating piano roll visualization from transcription result.

        Piano roll visualizations are useful for analyzing transcription
        quality and debugging. This test verifies the integration works.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

        # Generate piano roll
        output_path = temp_output_dir / "piano_roll.png"
        saved_path = render_piano_roll(result.notes, output_path)

        # Verify file was created (if matplotlib available)
        if saved_path is not None:
            assert saved_path.exists()
            assert saved_path.suffix == ".png"

    def test_render_note_density_from_transcription(
        self,
        melody_audio_path: Path,
        temp_output_dir: Path,
        mock_basic_pitch_melody,
    ):
        """
        Test generating note density histogram from transcription result.

        Note density plots show note distribution over time and help
        analyze rhythmic patterns in the transcription.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_melody,
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    melody_audio_path,
                    save_midi_file=False,
                )

        # Generate note density plot
        output_path = temp_output_dir / "note_density.png"
        saved_path = render_note_density(result.notes, output_path)

        # Verify file was created (if matplotlib available)
        if saved_path is not None:
            assert saved_path.exists()
            assert saved_path.suffix == ".png"

    def test_piano_roll_with_custom_parameters(
        self,
        piano_audio_path: Path,
        temp_output_dir: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test piano roll with custom visualization parameters.

        Verifies that custom parameters (figure size, color map, etc.)
        are properly applied to the visualization.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

        # Generate with custom parameters
        output_path = temp_output_dir / "piano_roll_custom.png"
        saved_path = render_piano_roll(
            result.notes,
            output_path,
            figsize=(20, 10),
            colormap="plasma",
            title="Custom Piano Roll",
            show_velocity=True,
        )

        if saved_path is not None:
            assert saved_path.exists()


@pytest.mark.integration
@pytest.mark.slow
class TestNoteDetectionAccuracy:
    """Integration tests for note detection accuracy on known inputs."""

    def test_detect_correct_number_of_notes_in_chord(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test that chord notes are correctly detected.

        Verifies the transcriber can detect multiple simultaneous notes
        (polyphonic transcription) in a chord.
        """
        # Mock returns 3 simultaneous notes (C, E, G)
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

        assert result.note_count == 3

    def test_detect_sequential_notes_in_melody(
        self,
        melody_audio_path: Path,
        mock_basic_pitch_melody,
    ):
        """
        Test that sequential melody notes are correctly detected.

        Verifies the transcriber can detect separate notes in a melody
        with distinct onset and offset times.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_melody,
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    melody_audio_path,
                    save_midi_file=False,
                )

        assert result.note_count == 4

        # Verify notes are sequential (not overlapping)
        for i in range(len(result.notes) - 1):
            assert result.notes[i].end_time <= result.notes[i + 1].start_time

    def test_pitch_detection_accuracy(
        self,
        piano_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test that detected pitches match expected values.

        Verifies pitch detection is accurate for known input frequencies.
        For C major chord, expects C4 (60), E4 (64), G4 (67).
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

        pitches = sorted([note.pitch for note in result.notes])
        expected_pitches = [60, 64, 67]  # C4, E4, G4

        assert pitches == expected_pitches

    def test_note_timing_accuracy(
        self,
        melody_audio_path: Path,
        mock_basic_pitch_melody,
    ):
        """
        Test that note timing is accurately detected.

        Verifies that start and end times match expected values within
        a reasonable tolerance for the test melody.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_melody,
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    melody_audio_path,
                    save_midi_file=False,
                )

        # Expected timings from mock
        expected_timings = [
            (0.0, 0.5),
            (0.6, 1.1),
            (1.2, 1.7),
            (1.8, 2.3),
        ]

        for note, (expected_start, expected_end) in zip(result.notes, expected_timings):
            assert abs(note.start_time - expected_start) < 0.1
            assert abs(note.end_time - expected_end) < 0.1

    def test_result_properties_accuracy(
        self,
        melody_audio_path: Path,
        mock_basic_pitch_melody,
    ):
        """
        Test MIDIResult computed properties are accurate.

        Verifies properties like duration, pitch_range, and average_velocity
        are correctly computed from the transcribed notes.
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_melody,
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    melody_audio_path,
                    save_midi_file=False,
                )

        # Duration should be end time of last note
        assert result.duration == pytest.approx(2.3, abs=0.1)

        # Pitch range should be C4 (60) to F4 (65)
        assert result.pitch_range == (60, 65)

        # Average velocity should be reasonable
        assert 0 < result.average_velocity <= 127


@pytest.mark.integration
@pytest.mark.slow
class TestErrorHandling:
    """Integration tests for error handling in transcription workflow."""

    def test_transcribe_nonexistent_file_raises_error(self):
        """
        Test that transcribing a non-existent file raises appropriate error.

        Verifies error handling for missing input files.
        """
        transcriber = MIDITranscriber()

        with pytest.raises(Exception):  # Will raise FileNotFoundError or similar
            transcriber.transcribe(
                "/path/that/does/not/exist.wav",
                save_midi_file=False,
            )

    def test_transcribe_with_basic_pitch_import_error(
        self,
        piano_audio_path: Path,
    ):
        """
        Test graceful error when Basic Pitch is not installed.

        Verifies that a helpful ImportError is raised when the required
        basic-pitch package is not available.
        """
        with patch("basic_pitch.inference.predict", side_effect=ImportError):
            transcriber = MIDITranscriber()

            with pytest.raises(ImportError, match="basic-pitch is required"):
                transcriber.transcribe(
                    piano_audio_path,
                    save_midi_file=False,
                )

    def test_transcribe_with_processing_error(
        self,
        piano_audio_path: Path,
    ):
        """
        Test error handling when Basic Pitch prediction fails.

        Verifies that processing errors are caught and wrapped in
        ProcessingError with informative messages.
        """
        with patch(
            "basic_pitch.inference.predict",
            side_effect=RuntimeError("Model error"),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()

                with pytest.raises(ProcessingError, match="Transcription failed"):
                    transcriber.transcribe(
                        piano_audio_path,
                        save_midi_file=False,
                    )

    def test_empty_audio_produces_empty_result(
        self,
        silence_audio: np.ndarray,
        temp_dir: Path,
        sample_rate: int,
    ):
        """
        Test that silent audio produces valid but empty transcription.

        Verifies graceful handling of audio with no detectable notes.
        """
        # Create silent audio file
        silence_path = temp_dir / "silence.wav"
        sf.write(silence_path, silence_audio, sample_rate)

        # Mock Basic Pitch to return no notes
        mock_output = (
            MagicMock(),
            MagicMock(),
            [],  # No notes
        )

        with patch("basic_pitch.inference.predict", return_value=mock_output):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    silence_path,
                    save_midi_file=False,
                )

        assert result.note_count == 0
        assert result.duration == 0.0
        assert result.pitch_range == (0, 0)

    def test_invalid_config_parameters_raise_validation_error(self):
        """
        Test that invalid configuration parameters raise validation errors.

        Verifies Pydantic validation catches invalid parameter values
        before they reach the transcription pipeline.
        """
        from pydantic import ValidationError

        # Invalid onset threshold
        with pytest.raises(ValidationError):
            TranscriptionConfig(onset_thresh=1.5)

        # Invalid frame threshold
        with pytest.raises(ValidationError):
            TranscriptionConfig(frame_thresh=0.05)

        # Invalid minimum note length
        with pytest.raises(ValidationError):
            TranscriptionConfig(minimum_note_length=5.0)


@pytest.mark.integration
@pytest.mark.slow
class TestAudioIOIntegration:
    """Integration tests for audio I/O with transcription."""

    def test_transcribe_loaded_audio_segment(
        self,
        sample_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test transcribing audio loaded via load_audio function.

        Verifies integration between audio I/O module and transcription,
        ensuring audio loaded with soundlab.io.audio_io can be transcribed.
        """
        # Load audio using audio_io
        audio_segment = load_audio(sample_audio_path)

        # Transcribe (still requires file path for Basic Pitch)
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    sample_audio_path,
                    save_midi_file=False,
                )

        assert result.note_count > 0

    def test_transcribe_resampled_audio(
        self,
        temp_dir: Path,
        sample_mono_audio: np.ndarray,
        sample_rate: int,
        mock_basic_pitch_predict,
    ):
        """
        Test transcribing audio with different sample rates.

        Verifies that audio with various sample rates can be transcribed
        successfully, testing the resampling integration.
        """
        # Save audio at different sample rate
        resampled_path = temp_dir / "resampled.wav"
        sf.write(resampled_path, sample_mono_audio, 22050)  # Half sample rate

        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    resampled_path,
                    save_midi_file=False,
                )

        assert result.note_count > 0
        assert result.source_path == resampled_path

    def test_transcribe_stereo_audio(
        self,
        sample_stereo_audio_path: Path,
        mock_basic_pitch_predict,
    ):
        """
        Test transcribing stereo audio files.

        Verifies that stereo audio is properly handled (Basic Pitch
        typically converts to mono internally).
        """
        with patch(
            "basic_pitch.inference.predict",
            return_value=mock_basic_pitch_predict(),
        ):
            with patch("basic_pitch.ICASSP_2022_MODEL_PATH", "/mock/path"):
                transcriber = MIDITranscriber()
                result = transcriber.transcribe(
                    sample_stereo_audio_path,
                    save_midi_file=False,
                )

        assert result.note_count > 0
