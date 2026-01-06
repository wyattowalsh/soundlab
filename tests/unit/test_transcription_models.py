"""Tests for soundlab.transcription.models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from soundlab.transcription.models import MIDIResult, NoteEvent, TranscriptionConfig


class TestTranscriptionConfig:
    """Test TranscriptionConfig model."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = TranscriptionConfig()

        assert config.onset_thresh == 0.5
        assert config.frame_thresh == 0.3
        assert config.minimum_note_length == 58.0
        assert config.minimum_frequency == 32.7
        assert config.maximum_frequency == 2093.0

    def test_onset_thresh_bounds(self):
        """Onset threshold should be validated."""
        TranscriptionConfig(onset_thresh=0.1)
        TranscriptionConfig(onset_thresh=0.9)

        with pytest.raises(ValidationError):
            TranscriptionConfig(onset_thresh=0.05)
        with pytest.raises(ValidationError):
            TranscriptionConfig(onset_thresh=0.95)

    def test_frame_thresh_bounds(self):
        """Frame threshold should be validated."""
        TranscriptionConfig(frame_thresh=0.1)
        TranscriptionConfig(frame_thresh=0.9)

        with pytest.raises(ValidationError):
            TranscriptionConfig(frame_thresh=0.05)

    def test_minimum_note_length_bounds(self):
        """Minimum note length should be validated."""
        TranscriptionConfig(minimum_note_length=10.0)
        TranscriptionConfig(minimum_note_length=200.0)

        with pytest.raises(ValidationError):
            TranscriptionConfig(minimum_note_length=5.0)

    def test_frequency_bounds(self):
        """Frequency bounds should be validated."""
        TranscriptionConfig(minimum_frequency=20.0)
        TranscriptionConfig(maximum_frequency=8000.0)

        with pytest.raises(ValidationError):
            TranscriptionConfig(minimum_frequency=10.0)

    def test_config_is_frozen(self):
        """Config should be immutable."""
        config = TranscriptionConfig()
        with pytest.raises(ValidationError):
            config.onset_thresh = 0.8


class TestNoteEvent:
    """Test NoteEvent model."""

    def test_create_note(self):
        """Should create note with required fields."""
        note = NoteEvent(
            start_time=0.0,
            end_time=0.5,
            pitch=60,
            velocity=100,
        )

        assert note.start_time == 0.0
        assert note.end_time == 0.5
        assert note.pitch == 60
        assert note.velocity == 100

    def test_duration_property(self):
        """Duration should be calculated correctly."""
        note = NoteEvent(start_time=1.0, end_time=1.5, pitch=60)
        assert note.duration == 0.5

    def test_duration_ms_property(self):
        """Duration in ms should be calculated correctly."""
        note = NoteEvent(start_time=0.0, end_time=0.5, pitch=60)
        assert note.duration_ms == 500.0

    def test_pitch_name_c4(self):
        """C4 (MIDI 60) should return 'C4'."""
        note = NoteEvent(start_time=0.0, end_time=0.5, pitch=60)
        assert note.pitch_name == "C4"

    def test_pitch_name_a4(self):
        """A4 (MIDI 69) should return 'A4'."""
        note = NoteEvent(start_time=0.0, end_time=0.5, pitch=69)
        assert note.pitch_name == "A4"

    def test_pitch_name_sharp(self):
        """C#4 (MIDI 61) should return 'C#4'."""
        note = NoteEvent(start_time=0.0, end_time=0.5, pitch=61)
        assert note.pitch_name == "C#4"

    def test_frequency_a4(self):
        """A4 frequency should be 440 Hz."""
        note = NoteEvent(start_time=0.0, end_time=0.5, pitch=69)
        assert note.frequency == pytest.approx(440.0, rel=0.01)

    def test_frequency_a3(self):
        """A3 frequency should be 220 Hz."""
        note = NoteEvent(start_time=0.0, end_time=0.5, pitch=57)
        assert note.frequency == pytest.approx(220.0, rel=0.01)

    def test_pitch_bounds(self):
        """Pitch should be 0-127."""
        NoteEvent(start_time=0.0, end_time=0.5, pitch=0)
        NoteEvent(start_time=0.0, end_time=0.5, pitch=127)

        with pytest.raises(ValidationError):
            NoteEvent(start_time=0.0, end_time=0.5, pitch=-1)
        with pytest.raises(ValidationError):
            NoteEvent(start_time=0.0, end_time=0.5, pitch=128)

    def test_velocity_bounds(self):
        """Velocity should be 1-127."""
        NoteEvent(start_time=0.0, end_time=0.5, pitch=60, velocity=1)
        NoteEvent(start_time=0.0, end_time=0.5, pitch=60, velocity=127)

        with pytest.raises(ValidationError):
            NoteEvent(start_time=0.0, end_time=0.5, pitch=60, velocity=0)
        with pytest.raises(ValidationError):
            NoteEvent(start_time=0.0, end_time=0.5, pitch=60, velocity=128)

    def test_confidence_bounds(self):
        """Confidence should be 0-1."""
        NoteEvent(start_time=0.0, end_time=0.5, pitch=60, confidence=0.0)
        NoteEvent(start_time=0.0, end_time=0.5, pitch=60, confidence=1.0)

        with pytest.raises(ValidationError):
            NoteEvent(start_time=0.0, end_time=0.5, pitch=60, confidence=1.5)


class TestMIDIResult:
    """Test MIDIResult model."""

    def test_create_result(self):
        """Should create result with notes."""
        notes = [
            NoteEvent(start_time=0.0, end_time=0.5, pitch=60),
            NoteEvent(start_time=0.5, end_time=1.0, pitch=62),
        ]
        config = TranscriptionConfig()
        result = MIDIResult(
            notes=notes,
            config=config,
            processing_time_seconds=1.0,
        )

        assert len(result.notes) == 2

    def test_note_count_property(self):
        """note_count should return number of notes."""
        notes = [
            NoteEvent(start_time=0.0, end_time=0.5, pitch=60),
            NoteEvent(start_time=0.5, end_time=1.0, pitch=62),
            NoteEvent(start_time=1.0, end_time=1.5, pitch=64),
        ]
        result = MIDIResult(
            notes=notes,
            config=TranscriptionConfig(),
            processing_time_seconds=1.0,
        )

        assert result.note_count == 3

    def test_duration_property(self):
        """duration should return max end_time."""
        notes = [
            NoteEvent(start_time=0.0, end_time=0.5, pitch=60),
            NoteEvent(start_time=1.0, end_time=2.5, pitch=62),
        ]
        result = MIDIResult(
            notes=notes,
            config=TranscriptionConfig(),
            processing_time_seconds=1.0,
        )

        assert result.duration == 2.5

    def test_duration_empty_notes(self):
        """duration should be 0 for empty notes."""
        result = MIDIResult(
            notes=[],
            config=TranscriptionConfig(),
            processing_time_seconds=1.0,
        )

        assert result.duration == 0.0

    def test_pitch_range_property(self):
        """pitch_range should return min/max pitches."""
        notes = [
            NoteEvent(start_time=0.0, end_time=0.5, pitch=48),
            NoteEvent(start_time=0.5, end_time=1.0, pitch=72),
            NoteEvent(start_time=1.0, end_time=1.5, pitch=60),
        ]
        result = MIDIResult(
            notes=notes,
            config=TranscriptionConfig(),
            processing_time_seconds=1.0,
        )

        assert result.pitch_range == (48, 72)

    def test_get_notes_in_range(self):
        """Should filter notes by time range."""
        notes = [
            NoteEvent(start_time=0.0, end_time=0.5, pitch=60),
            NoteEvent(start_time=0.5, end_time=1.0, pitch=62),
            NoteEvent(start_time=1.5, end_time=2.0, pitch=64),
        ]
        result = MIDIResult(
            notes=notes,
            config=TranscriptionConfig(),
            processing_time_seconds=1.0,
        )

        filtered = result.get_notes_in_range(0.0, 1.0)
        assert len(filtered) == 2

        filtered = result.get_notes_in_range(1.0, 2.0)
        assert len(filtered) == 1
