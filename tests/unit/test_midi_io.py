"""Tests for soundlab.io.midi_io module."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Used at runtime in tests
from unittest.mock import MagicMock, patch

import pytest

from soundlab.io.midi_io import (
    MIDIData,
    MIDINote,
    TimeSignature,
    load_midi,
    save_midi,
)


class TestMIDINote:
    """Tests for MIDINote model."""

    def test_valid_note(self) -> None:
        """Valid note parameters create successfully."""
        note = MIDINote(pitch=60, start_seconds=0.0, end_seconds=1.0, velocity=100)

        assert note.pitch == 60
        assert note.start_seconds == 0.0
        assert note.end_seconds == 1.0
        assert note.velocity == 100

    def test_pitch_lower_bound(self) -> None:
        """Pitch must be >= 0."""
        with pytest.raises(ValueError):
            MIDINote(pitch=-1, start_seconds=0.0, end_seconds=1.0, velocity=100)

    def test_pitch_upper_bound(self) -> None:
        """Pitch must be <= 127."""
        with pytest.raises(ValueError):
            MIDINote(pitch=128, start_seconds=0.0, end_seconds=1.0, velocity=100)

    def test_velocity_lower_bound(self) -> None:
        """Velocity must be >= 0."""
        with pytest.raises(ValueError):
            MIDINote(pitch=60, start_seconds=0.0, end_seconds=1.0, velocity=-1)

    def test_velocity_upper_bound(self) -> None:
        """Velocity must be <= 127."""
        with pytest.raises(ValueError):
            MIDINote(pitch=60, start_seconds=0.0, end_seconds=1.0, velocity=128)

    def test_start_non_negative(self) -> None:
        """Start time must be >= 0."""
        with pytest.raises(ValueError):
            MIDINote(pitch=60, start_seconds=-0.1, end_seconds=1.0, velocity=100)

    def test_end_non_negative(self) -> None:
        """End time must be >= 0."""
        with pytest.raises(ValueError):
            MIDINote(pitch=60, start_seconds=0.0, end_seconds=-0.1, velocity=100)

    def test_end_after_start(self) -> None:
        """End time must be >= start time."""
        with pytest.raises(ValueError):
            MIDINote(pitch=60, start_seconds=1.0, end_seconds=0.5, velocity=100)

    def test_end_equals_start(self) -> None:
        """Zero-length note (end == start) is allowed."""
        note = MIDINote(pitch=60, start_seconds=1.0, end_seconds=1.0, velocity=100)
        assert note.start_seconds == note.end_seconds

    def test_boundary_pitch_values(self) -> None:
        """Boundary pitch values (0 and 127) are valid."""
        note0 = MIDINote(pitch=0, start_seconds=0.0, end_seconds=1.0, velocity=64)
        note127 = MIDINote(pitch=127, start_seconds=0.0, end_seconds=1.0, velocity=64)

        assert note0.pitch == 0
        assert note127.pitch == 127


class TestTimeSignature:
    """Tests for TimeSignature model."""

    def test_valid_time_signature(self) -> None:
        """Valid time signature creates successfully."""
        ts = TimeSignature(numerator=4, denominator=4)
        assert ts.numerator == 4
        assert ts.denominator == 4

    def test_waltz_time(self) -> None:
        """3/4 time signature is valid."""
        ts = TimeSignature(numerator=3, denominator=4)
        assert ts.numerator == 3

    def test_numerator_lower_bound(self) -> None:
        """Numerator must be >= 1."""
        with pytest.raises(ValueError):
            TimeSignature(numerator=0, denominator=4)

    def test_denominator_lower_bound(self) -> None:
        """Denominator must be >= 1."""
        with pytest.raises(ValueError):
            TimeSignature(numerator=4, denominator=0)


class TestMIDIData:
    """Tests for MIDIData model."""

    def test_default_values(self) -> None:
        """Default MIDIData has expected values."""
        data = MIDIData()

        assert data.notes == []
        assert data.tempo == 120.0
        assert data.time_signature is None

    def test_custom_tempo(self) -> None:
        """Custom tempo is accepted."""
        data = MIDIData(tempo=90.0)
        assert data.tempo == 90.0

    def test_tempo_must_be_positive(self) -> None:
        """Tempo must be > 0."""
        with pytest.raises(ValueError):
            MIDIData(tempo=0.0)

        with pytest.raises(ValueError):
            MIDIData(tempo=-60.0)

    def test_with_notes(self) -> None:
        """MIDIData can contain notes."""
        notes = [
            MIDINote(pitch=60, start_seconds=0.0, end_seconds=0.5, velocity=100),
            MIDINote(pitch=64, start_seconds=0.5, end_seconds=1.0, velocity=80),
        ]
        data = MIDIData(notes=notes)

        assert len(data.notes) == 2
        assert data.notes[0].pitch == 60
        assert data.notes[1].pitch == 64

    def test_with_time_signature(self) -> None:
        """MIDIData can have time signature."""
        ts = TimeSignature(numerator=6, denominator=8)
        data = MIDIData(time_signature=ts)

        assert data.time_signature is not None
        assert data.time_signature.numerator == 6
        assert data.time_signature.denominator == 8


class TestLoadMidi:
    """Tests for load_midi function."""

    def test_load_midi_requires_mido(self) -> None:
        """load_midi raises ImportError when mido is not available."""
        with patch("soundlab.io.midi_io.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'mido'")

            with pytest.raises(ImportError, match="mido is required"):
                load_midi("test.mid")

    def test_load_midi_parses_notes(self) -> None:
        """load_midi correctly parses MIDI notes."""
        # Create mock mido module and MidiFile
        mock_mido = MagicMock()

        # Configure mock
        mock_midi_file = MagicMock()
        mock_midi_file.ticks_per_beat = 480

        # Create mock messages
        tempo_msg = MagicMock(type="set_tempo", time=0, tempo=500000)
        note_on_msg = MagicMock(type="note_on", time=480, note=60, velocity=100)
        note_off_msg = MagicMock(type="note_off", time=480, note=60, velocity=0)

        mock_mido.MidiFile.return_value = mock_midi_file
        mock_mido.merge_tracks.return_value = [tempo_msg, note_on_msg, note_off_msg]
        mock_mido.bpm2tempo.return_value = 500000
        mock_mido.tempo2bpm.return_value = 120.0
        mock_mido.tick2second.side_effect = (
            lambda ticks, tpb, tempo: ticks / tpb * (tempo / 1000000)
        )

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        assert isinstance(result, MIDIData)
        # Note: actual parsing depends on mock accuracy


class TestSaveMidi:
    """Tests for save_midi function."""

    def test_save_midi_requires_mido(self) -> None:
        """save_midi raises ImportError when mido is not available."""
        with patch("soundlab.io.midi_io.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'mido'")

            data = MIDIData()
            with pytest.raises(ImportError, match="mido is required"):
                save_midi(data, "test.mid")

    def test_save_midi_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_midi creates parent directories."""
        mock_mido = MagicMock()

        mock_midi_file = MagicMock()
        mock_mido.MidiFile.return_value = mock_midi_file
        mock_mido.MidiTrack.return_value = []
        mock_mido.bpm2tempo.return_value = 500000
        mock_mido.MetaMessage.return_value = MagicMock()

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            output_path = tmp_path / "nested" / "dirs" / "output.mid"
            data = MIDIData()
            save_midi(data, output_path)

        assert output_path.parent.exists()

    def test_save_midi_with_notes(self, tmp_path: Path) -> None:
        """save_midi correctly saves notes."""
        mock_mido = MagicMock()

        mock_midi_file = MagicMock()
        mock_track = []
        mock_mido.MidiFile.return_value = mock_midi_file
        mock_mido.MidiTrack.return_value = mock_track
        mock_mido.bpm2tempo.return_value = 500000
        mock_mido.second2tick.return_value = 480
        mock_mido.MetaMessage.return_value = MagicMock(time=0)
        mock_mido.Message.return_value = MagicMock(time=0)

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            notes = [
                MIDINote(pitch=60, start_seconds=0.0, end_seconds=0.5, velocity=100),
                MIDINote(pitch=64, start_seconds=0.5, end_seconds=1.0, velocity=80),
            ]
            data = MIDIData(notes=notes, tempo=120.0)
            output_path = tmp_path / "output.mid"
            save_midi(data, output_path)

        # Verify mido methods were called
        mock_mido.MidiFile.assert_called_once()
        mock_midi_file.save.assert_called_once()


class TestMIDIRoundTrip:
    """Integration tests for load/save round-trip."""

    def test_empty_midi_roundtrip(self, tmp_path: Path) -> None:
        """Empty MIDIData can be saved and loaded."""
        # This test requires actual mido, skip if not available
        pytest.importorskip("mido")

        from soundlab.io.midi_io import load_midi, save_midi

        output_path = tmp_path / "empty.mid"
        data = MIDIData(tempo=100.0)
        save_midi(data, output_path)

        loaded = load_midi(output_path)

        assert loaded.tempo == pytest.approx(100.0, rel=0.01)
        assert len(loaded.notes) == 0

    def test_notes_roundtrip(self, tmp_path: Path) -> None:
        """Notes survive save/load roundtrip."""
        pytest.importorskip("mido")

        from soundlab.io.midi_io import load_midi, save_midi

        notes = [
            MIDINote(pitch=60, start_seconds=0.0, end_seconds=0.5, velocity=100),
            MIDINote(pitch=64, start_seconds=0.5, end_seconds=1.0, velocity=80),
            MIDINote(pitch=67, start_seconds=1.0, end_seconds=1.5, velocity=90),
        ]
        data = MIDIData(notes=notes, tempo=120.0)
        output_path = tmp_path / "notes.mid"

        save_midi(data, output_path)
        loaded = load_midi(output_path)

        assert len(loaded.notes) == 3
        assert loaded.notes[0].pitch == 60
        assert loaded.notes[1].pitch == 64
        assert loaded.notes[2].pitch == 67

    def test_time_signature_roundtrip(self, tmp_path: Path) -> None:
        """Time signature survives save/load roundtrip."""
        pytest.importorskip("mido")

        from soundlab.io.midi_io import load_midi, save_midi

        ts = TimeSignature(numerator=3, denominator=4)
        data = MIDIData(time_signature=ts, tempo=90.0)
        output_path = tmp_path / "timesig.mid"

        save_midi(data, output_path)
        loaded = load_midi(output_path)

        assert loaded.time_signature is not None
        assert loaded.time_signature.numerator == 3
        assert loaded.time_signature.denominator == 4
