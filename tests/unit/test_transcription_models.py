"""Tests for transcription models."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pydantic = pytest.importorskip("pydantic")

from soundlab.transcription.models import MIDIResult, NoteEvent, TranscriptionConfig

# --------------------------------------------------------------------------- #
# TranscriptionConfig Tests
# --------------------------------------------------------------------------- #


class TestTranscriptionConfig:
    """Tests for TranscriptionConfig model."""

    def test_default_values(self) -> None:
        """TranscriptionConfig should have sensible defaults."""
        config = TranscriptionConfig()

        assert config.onset_thresh == 0.5
        assert config.frame_thresh == 0.3
        assert config.min_note_length == 0.058
        assert config.min_freq == 32.7
        assert config.max_freq == 2093.0

    def test_custom_thresholds(self) -> None:
        """TranscriptionConfig should accept custom thresholds."""
        config = TranscriptionConfig(
            onset_thresh=0.6,
            frame_thresh=0.4,
            min_note_length=0.1,
            min_freq=50.0,
            max_freq=4000.0,
        )

        assert config.onset_thresh == 0.6
        assert config.frame_thresh == 0.4
        assert config.min_note_length == 0.1
        assert config.min_freq == 50.0
        assert config.max_freq == 4000.0

    def test_onset_thresh_bounds(self) -> None:
        """onset_thresh should be bounded [0.1, 0.9]."""
        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(onset_thresh=0.05)

        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(onset_thresh=0.95)

        config_min = TranscriptionConfig(onset_thresh=0.1)
        config_max = TranscriptionConfig(onset_thresh=0.9)

        assert config_min.onset_thresh == 0.1
        assert config_max.onset_thresh == 0.9

    def test_frame_thresh_bounds(self) -> None:
        """frame_thresh should be bounded [0.1, 0.9]."""
        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(frame_thresh=0.05)

        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(frame_thresh=0.95)

        config_min = TranscriptionConfig(frame_thresh=0.1)
        config_max = TranscriptionConfig(frame_thresh=0.9)

        assert config_min.frame_thresh == 0.1
        assert config_max.frame_thresh == 0.9

    def test_min_note_length_bounds(self) -> None:
        """min_note_length should be bounded [0.01, 0.2]."""
        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(min_note_length=0.005)

        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(min_note_length=0.3)

        config_min = TranscriptionConfig(min_note_length=0.01)
        config_max = TranscriptionConfig(min_note_length=0.2)

        assert config_min.min_note_length == 0.01
        assert config_max.min_note_length == 0.2

    def test_min_freq_bounds(self) -> None:
        """min_freq should be bounded [20.0, 500.0]."""
        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(min_freq=10.0)

        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(min_freq=600.0)

        config_min = TranscriptionConfig(min_freq=20.0)
        config_max = TranscriptionConfig(min_freq=500.0)

        assert config_min.min_freq == 20.0
        assert config_max.min_freq == 500.0

    def test_max_freq_bounds(self) -> None:
        """max_freq should be bounded [1000.0, 8000.0]."""
        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(max_freq=500.0)

        with pytest.raises(pydantic.ValidationError):
            TranscriptionConfig(max_freq=10000.0)

        config_min = TranscriptionConfig(max_freq=1000.0)
        config_max = TranscriptionConfig(max_freq=8000.0)

        assert config_min.max_freq == 1000.0
        assert config_max.max_freq == 8000.0

    def test_frozen_model(self) -> None:
        """TranscriptionConfig should be immutable."""
        config = TranscriptionConfig()

        with pytest.raises(pydantic.ValidationError):
            config.onset_thresh = 0.7  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# NoteEvent Tests
# --------------------------------------------------------------------------- #


class TestNoteEvent:
    """Tests for NoteEvent model."""

    def test_basic_note(self) -> None:
        """NoteEvent should accept basic note parameters."""
        note = NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80)

        assert note.start == 0.0
        assert note.end == 1.0
        assert note.pitch == 60
        assert note.velocity == 80

    def test_start_non_negative(self) -> None:
        """start should be non-negative."""
        with pytest.raises(pydantic.ValidationError):
            NoteEvent(start=-1.0, end=1.0, pitch=60, velocity=80)

        note = NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80)
        assert note.start == 0.0

    def test_end_non_negative(self) -> None:
        """end should be non-negative."""
        with pytest.raises(pydantic.ValidationError):
            NoteEvent(start=0.0, end=-1.0, pitch=60, velocity=80)

    def test_end_after_start_validation(self) -> None:
        """end must be >= start."""
        with pytest.raises(pydantic.ValidationError) as exc_info:
            NoteEvent(start=2.0, end=1.0, pitch=60, velocity=80)

        assert "end must be greater than or equal to start" in str(exc_info.value)

    def test_end_equals_start(self) -> None:
        """end can equal start (zero-length note)."""
        note = NoteEvent(start=1.0, end=1.0, pitch=60, velocity=80)

        assert note.start == note.end

    def test_pitch_bounds(self) -> None:
        """pitch should be bounded [0, 127]."""
        with pytest.raises(pydantic.ValidationError):
            NoteEvent(start=0.0, end=1.0, pitch=-1, velocity=80)

        with pytest.raises(pydantic.ValidationError):
            NoteEvent(start=0.0, end=1.0, pitch=128, velocity=80)

        note_min = NoteEvent(start=0.0, end=1.0, pitch=0, velocity=80)
        note_max = NoteEvent(start=0.0, end=1.0, pitch=127, velocity=80)

        assert note_min.pitch == 0
        assert note_max.pitch == 127

    def test_velocity_bounds(self) -> None:
        """velocity should be bounded [0, 127]."""
        with pytest.raises(pydantic.ValidationError):
            NoteEvent(start=0.0, end=1.0, pitch=60, velocity=-1)

        with pytest.raises(pydantic.ValidationError):
            NoteEvent(start=0.0, end=1.0, pitch=60, velocity=128)

        note_min = NoteEvent(start=0.0, end=1.0, pitch=60, velocity=0)
        note_max = NoteEvent(start=0.0, end=1.0, pitch=60, velocity=127)

        assert note_min.velocity == 0
        assert note_max.velocity == 127

    def test_frozen_model(self) -> None:
        """NoteEvent should be immutable."""
        note = NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80)

        with pytest.raises(pydantic.ValidationError):
            note.pitch = 72  # type: ignore[misc]

    def test_midi_note_range(self) -> None:
        """Common MIDI note values should be valid."""
        # Middle C (C4)
        middle_c = NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80)
        assert middle_c.pitch == 60

        # A4 (440 Hz reference)
        a4 = NoteEvent(start=0.0, end=1.0, pitch=69, velocity=80)
        assert a4.pitch == 69

        # Lowest MIDI note
        lowest = NoteEvent(start=0.0, end=1.0, pitch=0, velocity=80)
        assert lowest.pitch == 0

        # Highest MIDI note
        highest = NoteEvent(start=0.0, end=1.0, pitch=127, velocity=80)
        assert highest.pitch == 127


# --------------------------------------------------------------------------- #
# NoteEvent Ordering Tests
# --------------------------------------------------------------------------- #


class TestNoteEventOrdering:
    """Tests for NoteEvent temporal ordering."""

    def test_notes_can_overlap(self) -> None:
        """Multiple notes can have overlapping time ranges."""
        note1 = NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80)
        note2 = NoteEvent(start=0.5, end=1.5, pitch=64, velocity=75)

        # Both should be valid
        assert note1.start < note2.start
        assert note1.end > note2.start  # Overlapping

    def test_notes_can_be_simultaneous(self) -> None:
        """Multiple notes can start at the same time (chord)."""
        notes = [
            NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80),  # C
            NoteEvent(start=0.0, end=1.0, pitch=64, velocity=80),  # E
            NoteEvent(start=0.0, end=1.0, pitch=67, velocity=80),  # G
        ]

        # All start at same time
        assert all(n.start == 0.0 for n in notes)

    def test_notes_can_be_sequential(self) -> None:
        """Notes can be sequential (non-overlapping)."""
        notes = [
            NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80),
            NoteEvent(start=1.0, end=2.0, pitch=64, velocity=80),
            NoteEvent(start=2.0, end=3.0, pitch=67, velocity=80),
        ]

        for i in range(len(notes) - 1):
            assert notes[i].end <= notes[i + 1].start

    def test_sorting_by_start_time(self) -> None:
        """Notes should be sortable by start time."""
        notes = [
            NoteEvent(start=2.0, end=3.0, pitch=67, velocity=80),
            NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80),
            NoteEvent(start=1.0, end=2.0, pitch=64, velocity=80),
        ]

        sorted_notes = sorted(notes, key=lambda n: n.start)

        assert sorted_notes[0].start == 0.0
        assert sorted_notes[1].start == 1.0
        assert sorted_notes[2].start == 2.0


# --------------------------------------------------------------------------- #
# MIDIResult Tests
# --------------------------------------------------------------------------- #


class TestMIDIResult:
    """Tests for MIDIResult model."""

    def test_basic_result(self) -> None:
        """MIDIResult with basic fields."""
        notes = [NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80)]

        result = MIDIResult(
            notes=notes,
            path=Path("/output/result.mid"),
            config=TranscriptionConfig(),
            processing_time=5.0,
        )

        assert len(result.notes) == 1
        assert result.path == Path("/output/result.mid")
        assert result.processing_time == 5.0

    def test_empty_notes(self) -> None:
        """MIDIResult can have empty notes list."""
        result = MIDIResult(
            notes=[],
            path=Path("/output/empty.mid"),
            config=TranscriptionConfig(),
            processing_time=1.0,
        )

        assert result.notes == []

    def test_multiple_notes(self) -> None:
        """MIDIResult with multiple notes."""
        notes = [
            NoteEvent(start=0.0, end=0.5, pitch=60, velocity=80),
            NoteEvent(start=0.5, end=1.0, pitch=64, velocity=75),
            NoteEvent(start=1.0, end=1.5, pitch=67, velocity=85),
        ]

        result = MIDIResult(
            notes=notes,
            path=Path("/output/melody.mid"),
            config=TranscriptionConfig(),
            processing_time=3.0,
        )

        assert len(result.notes) == 3

    def test_nested_config(self) -> None:
        """MIDIResult preserves nested TranscriptionConfig."""
        config = TranscriptionConfig(
            onset_thresh=0.6,
            frame_thresh=0.4,
            min_note_length=0.1,
        )

        result = MIDIResult(
            notes=[],
            path=Path("/output/result.mid"),
            config=config,
            processing_time=2.0,
        )

        assert result.config.onset_thresh == 0.6
        assert result.config.frame_thresh == 0.4
        assert result.config.min_note_length == 0.1

    def test_frozen_model(self) -> None:
        """MIDIResult should be immutable."""
        result = MIDIResult(
            notes=[],
            path=Path("/output/result.mid"),
            config=TranscriptionConfig(),
            processing_time=5.0,
        )

        with pytest.raises(pydantic.ValidationError):
            result.processing_time = 10.0  # type: ignore[misc]

    def test_path_types(self) -> None:
        """path should accept Path objects."""
        result = MIDIResult(
            notes=[],
            path=Path("/output/result.mid"),
            config=TranscriptionConfig(),
            processing_time=1.0,
        )

        assert isinstance(result.path, Path)


# --------------------------------------------------------------------------- #
# Integration Tests
# --------------------------------------------------------------------------- #


class TestTranscriptionModelIntegration:
    """Integration tests for transcription models."""

    def test_config_and_notes_in_result(self) -> None:
        """Config and notes should work together in MIDIResult."""
        config = TranscriptionConfig(
            onset_thresh=0.55,
            frame_thresh=0.35,
        )

        notes = [
            NoteEvent(start=0.0, end=0.25, pitch=60, velocity=100),
            NoteEvent(start=0.25, end=0.5, pitch=62, velocity=95),
            NoteEvent(start=0.5, end=0.75, pitch=64, velocity=90),
            NoteEvent(start=0.75, end=1.0, pitch=65, velocity=85),
        ]

        result = MIDIResult(
            notes=notes,
            path=Path("/output/scale.mid"),
            config=config,
            processing_time=2.5,
        )

        assert len(result.notes) == 4
        assert result.config.onset_thresh == 0.55

    def test_chord_representation(self) -> None:
        """Chords should be representable as simultaneous notes."""
        # C major chord
        chord = [
            NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80),  # C
            NoteEvent(start=0.0, end=1.0, pitch=64, velocity=80),  # E
            NoteEvent(start=0.0, end=1.0, pitch=67, velocity=80),  # G
        ]

        result = MIDIResult(
            notes=chord,
            path=Path("/output/chord.mid"),
            config=TranscriptionConfig(),
            processing_time=1.0,
        )

        # All notes at same start time
        assert all(n.start == 0.0 for n in result.notes)
        # Different pitches
        pitches = [n.pitch for n in result.notes]
        assert len(set(pitches)) == 3

    def test_polyphonic_passage(self) -> None:
        """Complex polyphonic passages should be representable."""
        notes = [
            # Voice 1: Melody
            NoteEvent(start=0.0, end=0.5, pitch=72, velocity=90),
            NoteEvent(start=0.5, end=1.0, pitch=74, velocity=85),
            NoteEvent(start=1.0, end=1.5, pitch=76, velocity=80),
            # Voice 2: Bass
            NoteEvent(start=0.0, end=1.5, pitch=48, velocity=70),
            # Voice 3: Harmony
            NoteEvent(start=0.0, end=0.75, pitch=60, velocity=65),
            NoteEvent(start=0.75, end=1.5, pitch=62, velocity=65),
        ]

        result = MIDIResult(
            notes=notes,
            path=Path("/output/polyphonic.mid"),
            config=TranscriptionConfig(),
            processing_time=5.0,
        )

        assert len(result.notes) == 6

        # Check polyphony at t=0.25 (3 notes sounding)
        notes_at_time = [n for n in result.notes if n.start <= 0.25 < n.end]
        assert len(notes_at_time) == 3

    def test_serialization_roundtrip(self) -> None:
        """Models should survive serialization/deserialization."""
        config = TranscriptionConfig(onset_thresh=0.6)
        notes = [NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80)]

        result = MIDIResult(
            notes=notes,
            path=Path("/output/test.mid"),
            config=config,
            processing_time=1.0,
        )

        # Serialize to dict
        data = result.model_dump()

        # Verify serialized data
        assert data["notes"][0]["pitch"] == 60
        assert data["config"]["onset_thresh"] == 0.6
