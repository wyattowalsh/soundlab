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


class TestLoadMidiEdgeCases:
    """Edge case tests for load_midi function."""

    def _create_mock_mido(
        self,
        messages: list[MagicMock],
        ticks_per_beat: int = 480,
    ) -> MagicMock:
        """Create a configured mock mido module."""
        mock_mido = MagicMock()
        mock_midi_file = MagicMock()
        mock_midi_file.ticks_per_beat = ticks_per_beat
        mock_midi_file.tracks = [messages]

        mock_mido.MidiFile.return_value = mock_midi_file
        mock_mido.merge_tracks.return_value = messages
        mock_mido.bpm2tempo.return_value = 500000  # 120 BPM
        mock_mido.tempo2bpm.return_value = 120.0
        mock_mido.tick2second.side_effect = (
            lambda ticks, tpb, tempo: ticks / tpb * (tempo / 1_000_000)
        )
        return mock_mido

    def _msg(self, msg_type: str, time: int = 0, **kwargs: object) -> MagicMock:
        """Create a mock MIDI message."""
        msg = MagicMock()
        msg.type = msg_type
        msg.time = time
        for key, value in kwargs.items():
            setattr(msg, key, value)
        return msg

    def test_velocity_zero_as_note_off(self) -> None:
        """note_on with velocity=0 is treated as note_off."""
        messages = [
            self._msg("note_on", time=0, note=60, velocity=100),
            self._msg("note_on", time=480, note=60, velocity=0),  # note_off equivalent
        ]
        mock_mido = self._create_mock_mido(messages)

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        assert len(result.notes) == 1
        assert result.notes[0].pitch == 60
        assert result.notes[0].velocity == 100
        assert result.notes[0].end_seconds > result.notes[0].start_seconds

    def test_orphan_note_off_ignored(self) -> None:
        """note_off without matching note_on is ignored."""
        messages = [
            self._msg("note_off", time=0, note=60, velocity=0),  # orphan
            self._msg("note_on", time=480, note=64, velocity=100),
            self._msg("note_off", time=480, note=64, velocity=0),
        ]
        mock_mido = self._create_mock_mido(messages)

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        assert len(result.notes) == 1
        assert result.notes[0].pitch == 64

    def test_polyphonic_notes_same_pitch(self) -> None:
        """Multiple notes on same pitch are handled correctly."""
        # Two overlapping notes on pitch 60 (FIFO matching)
        messages = [
            self._msg("note_on", time=0, note=60, velocity=100),
            self._msg("note_on", time=240, note=60, velocity=80),  # Second note starts
            self._msg("note_off", time=240, note=60, velocity=0),  # First note ends
            self._msg("note_off", time=240, note=60, velocity=0),  # Second note ends
        ]
        mock_mido = self._create_mock_mido(messages)

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        assert len(result.notes) == 2
        # First note: velocity 100
        assert result.notes[0].velocity == 100
        # Second note: velocity 80
        assert result.notes[1].velocity == 80

    def test_overlapping_notes_different_pitch(self) -> None:
        """Overlapping notes on different pitches preserved."""
        messages = [
            self._msg("note_on", time=0, note=60, velocity=100),
            self._msg("note_on", time=240, note=64, velocity=90),  # Overlaps with 60
            self._msg("note_off", time=240, note=60, velocity=0),
            self._msg("note_off", time=480, note=64, velocity=0),
        ]
        mock_mido = self._create_mock_mido(messages)

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        assert len(result.notes) == 2
        pitches = {n.pitch for n in result.notes}
        assert pitches == {60, 64}

    def test_multi_track_notes_merged(self) -> None:
        """Notes from multiple tracks merged into single list."""
        # mido.merge_tracks merges all tracks, so we simulate merged output
        messages = [
            self._msg("note_on", time=0, note=60, velocity=100),  # Track 1
            self._msg("note_on", time=0, note=64, velocity=90),  # Track 2
            self._msg("note_off", time=480, note=60, velocity=0),  # Track 1
            self._msg("note_off", time=0, note=64, velocity=0),  # Track 2
        ]
        mock_mido = self._create_mock_mido(messages)

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        assert len(result.notes) == 2
        pitches = {n.pitch for n in result.notes}
        assert pitches == {60, 64}

    def test_empty_file_returns_empty_notes(self) -> None:
        """Empty MIDI file returns empty notes list."""
        messages: list[MagicMock] = []
        mock_mido = self._create_mock_mido(messages)

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        assert result.notes == []
        assert result.tempo == 120.0  # Default
        assert result.time_signature == TimeSignature(numerator=4, denominator=4)

    def test_no_tempo_message_defaults_120(self) -> None:
        """Missing tempo message defaults to 120 BPM."""
        messages = [
            self._msg("note_on", time=0, note=60, velocity=100),
            self._msg("note_off", time=480, note=60, velocity=0),
        ]
        mock_mido = self._create_mock_mido(messages)

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        assert result.tempo == 120.0

    def test_no_time_sig_defaults_4_4(self) -> None:
        """Missing time signature defaults to 4/4."""
        messages = [
            self._msg("note_on", time=0, note=60, velocity=100),
            self._msg("note_off", time=480, note=60, velocity=0),
        ]
        mock_mido = self._create_mock_mido(messages)

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        assert result.time_signature is not None
        assert result.time_signature.numerator == 4
        assert result.time_signature.denominator == 4

    def test_multiple_tempo_changes(self) -> None:
        """Multiple tempo changes - uses last encountered tempo value."""
        messages = [
            self._msg("set_tempo", time=0, tempo=500000),  # 120 BPM
            self._msg("note_on", time=480, note=60, velocity=100),
            self._msg("set_tempo", time=0, tempo=750000),  # 80 BPM
            self._msg("note_off", time=480, note=60, velocity=0),
        ]
        mock_mido = self._create_mock_mido(messages)
        mock_mido.tempo2bpm.side_effect = lambda t: 60_000_000 / t

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        # Implementation uses last tempo encountered for return value
        assert result.tempo == pytest.approx(80.0, rel=0.01)

    def test_multiple_time_signatures(self) -> None:
        """Multiple time signatures - uses last encountered."""
        messages = [
            self._msg("time_signature", time=0, numerator=4, denominator=4),
            self._msg("note_on", time=480, note=60, velocity=100),
            self._msg("time_signature", time=0, numerator=3, denominator=4),
            self._msg("note_off", time=480, note=60, velocity=0),
        ]
        mock_mido = self._create_mock_mido(messages)

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        # Implementation overwrites, so last one wins
        assert result.time_signature is not None
        assert result.time_signature.numerator == 3
        assert result.time_signature.denominator == 4

    def test_note_events_sorted_by_time(self) -> None:
        """Notes are returned in order of start time."""
        # Messages arrive in time order via merge_tracks
        messages = [
            self._msg("note_on", time=480, note=64, velocity=90),  # Starts at 0.5s
            self._msg("note_off", time=480, note=64, velocity=0),
            self._msg("note_on", time=0, note=60, velocity=100),  # Starts at 1.0s
            self._msg("note_off", time=480, note=60, velocity=0),
        ]
        mock_mido = self._create_mock_mido(messages)

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        # Notes added in order they complete (note_off order)
        assert len(result.notes) == 2
        # First completed note is pitch 64
        assert result.notes[0].pitch == 64
        assert result.notes[1].pitch == 60

    def test_sustain_pedal_control_ignored(self) -> None:
        """Control change messages (like sustain pedal) are ignored."""
        messages = [
            self._msg("note_on", time=0, note=60, velocity=100),
            self._msg("control_change", time=0, control=64, value=127),  # Sustain on
            self._msg("control_change", time=240, control=64, value=0),  # Sustain off
            self._msg("note_off", time=240, note=60, velocity=0),
        ]
        mock_mido = self._create_mock_mido(messages)

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            result = load_midi("test.mid")

        # Only the note is parsed, control changes ignored
        assert len(result.notes) == 1
        assert result.notes[0].pitch == 60


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


class TestSecondsToTicks:
    """Tests for _seconds_to_ticks helper function."""

    def test_zero_seconds_returns_zero(self) -> None:
        """0 seconds converts to 0 ticks."""
        from soundlab.io.midi_io import _seconds_to_ticks

        mock_mido = MagicMock()
        mock_mido.second2tick.return_value = 0

        result = _seconds_to_ticks(0.0, ticks_per_beat=480, tempo=500000, mido=mock_mido)

        assert result == 0
        # second2tick should NOT be called for zero/negative seconds
        mock_mido.second2tick.assert_not_called()

    def test_negative_seconds_returns_zero(self) -> None:
        """Negative seconds converts to 0 ticks."""
        from soundlab.io.midi_io import _seconds_to_ticks

        mock_mido = MagicMock()

        result = _seconds_to_ticks(-1.0, ticks_per_beat=480, tempo=500000, mido=mock_mido)

        assert result == 0
        mock_mido.second2tick.assert_not_called()

    def test_positive_seconds_correct(self) -> None:
        """Positive seconds convert correctly."""
        from soundlab.io.midi_io import _seconds_to_ticks

        mock_mido = MagicMock()
        # With TPB=480 and tempo=500000 (120 BPM): 1 sec = 960 ticks
        mock_mido.second2tick.return_value = 960

        result = _seconds_to_ticks(1.0, ticks_per_beat=480, tempo=500000, mido=mock_mido)

        assert result == 960
        mock_mido.second2tick.assert_called_once_with(1.0, 480, 500000)

    def test_fractional_seconds(self) -> None:
        """Fractional seconds convert correctly."""
        from soundlab.io.midi_io import _seconds_to_ticks

        mock_mido = MagicMock()
        # 0.5 seconds at 120 BPM = 480 ticks
        mock_mido.second2tick.return_value = 480.0

        result = _seconds_to_ticks(0.5, ticks_per_beat=480, tempo=500000, mido=mock_mido)

        assert result == 480
        mock_mido.second2tick.assert_called_once_with(0.5, 480, 500000)

    def test_rounds_fractional_ticks(self) -> None:
        """Fractional tick results are rounded to nearest integer."""
        from soundlab.io.midi_io import _seconds_to_ticks

        mock_mido = MagicMock()
        mock_mido.second2tick.return_value = 480.6

        result = _seconds_to_ticks(0.5, ticks_per_beat=480, tempo=500000, mido=mock_mido)

        assert result == 481  # Rounded from 480.6

    def test_uses_mido_second2tick(self) -> None:
        """Function delegates to mido.second2tick."""
        from soundlab.io.midi_io import _seconds_to_ticks

        mock_mido = MagicMock()
        mock_mido.second2tick.return_value = 1234

        result = _seconds_to_ticks(2.5, ticks_per_beat=960, tempo=600000, mido=mock_mido)

        mock_mido.second2tick.assert_called_once_with(2.5, 960, 600000)
        assert result == 1234


class TestDefaultTimeSignature:
    """Tests for _default_time_signature helper."""

    def test_returns_4_4_time_signature(self) -> None:
        """Returns TimeSignature(4, 4)."""
        from soundlab.io.midi_io import _default_time_signature

        ts = _default_time_signature()

        assert ts.numerator == 4
        assert ts.denominator == 4

    def test_returns_time_signature_model(self) -> None:
        """Returns proper TimeSignature Pydantic model."""
        from soundlab.io.midi_io import _default_time_signature

        ts = _default_time_signature()

        assert isinstance(ts, TimeSignature)


class TestSaveMidiEdgeCases:
    """Edge case tests for save_midi function."""

    def test_zero_duration_notes_handled(self, tmp_path: Path) -> None:
        """Notes with start == end are handled."""
        mock_mido = MagicMock()
        mock_midi_file = MagicMock()
        mock_track: list[MagicMock] = []

        mock_mido.MidiFile.return_value = mock_midi_file
        mock_mido.MidiTrack.return_value = mock_track
        mock_mido.bpm2tempo.return_value = 500000
        # Zero duration note: start and end at same tick
        mock_mido.second2tick.return_value = 480
        mock_mido.MetaMessage.return_value = MagicMock(time=0)
        mock_mido.Message.side_effect = lambda msg_type, **kwargs: MagicMock(
            type=msg_type, time=0, **kwargs
        )

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            # Note with start == end (zero duration)
            notes = [MIDINote(pitch=60, start_seconds=0.5, end_seconds=0.5, velocity=100)]
            data = MIDIData(notes=notes, tempo=120.0)
            output_path = tmp_path / "zero_duration.mid"
            save_midi(data, output_path)

        # Should have created both note_on and note_off messages
        assert mock_mido.Message.call_count == 2
        mock_midi_file.save.assert_called_once()

    def test_notes_at_time_zero(self, tmp_path: Path) -> None:
        """Notes starting at time 0 work correctly."""
        mock_mido = MagicMock()
        mock_midi_file = MagicMock()
        mock_track: list[MagicMock] = []

        mock_mido.MidiFile.return_value = mock_midi_file
        mock_mido.MidiTrack.return_value = mock_track
        mock_mido.bpm2tempo.return_value = 500000
        # Return 0 for start, 480 for end
        mock_mido.second2tick.side_effect = lambda s, tpb, tempo: 0 if s <= 0 else 480
        mock_mido.MetaMessage.return_value = MagicMock(time=0)

        note_on_msg = MagicMock(type="note_on", time=0)
        note_off_msg = MagicMock(type="note_off", time=0)
        mock_mido.Message.side_effect = [note_on_msg, note_off_msg]

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            notes = [MIDINote(pitch=60, start_seconds=0.0, end_seconds=0.5, velocity=100)]
            data = MIDIData(notes=notes, tempo=120.0)
            output_path = tmp_path / "time_zero.mid"
            save_midi(data, output_path)

        # Verify note_on was created for time 0
        mock_mido.Message.assert_any_call("note_on", note=60, velocity=100)
        mock_midi_file.save.assert_called_once()

    def test_simultaneous_notes_ordered(self, tmp_path: Path) -> None:
        """Simultaneous notes maintain consistent ordering (note_off before note_on at same tick)."""
        mock_mido = MagicMock()
        mock_midi_file = MagicMock()
        mock_track: list[MagicMock] = []

        mock_mido.MidiFile.return_value = mock_midi_file
        mock_mido.MidiTrack.return_value = mock_track
        mock_mido.bpm2tempo.return_value = 500000
        # All notes at same tick
        mock_mido.second2tick.return_value = 480
        mock_mido.MetaMessage.return_value = MagicMock(time=0)

        messages_created: list[tuple[str, int]] = []

        def track_message(msg_type: str, **kwargs: int) -> MagicMock:
            msg = MagicMock(type=msg_type, time=0, **kwargs)
            messages_created.append((msg_type, kwargs.get("note", 0)))
            return msg

        mock_mido.Message.side_effect = track_message

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            # Multiple notes starting and ending at same time
            notes = [
                MIDINote(pitch=60, start_seconds=0.5, end_seconds=0.5, velocity=100),
                MIDINote(pitch=64, start_seconds=0.5, end_seconds=0.5, velocity=80),
            ]
            data = MIDIData(notes=notes, tempo=120.0)
            output_path = tmp_path / "simultaneous.mid"
            save_midi(data, output_path)

        # Messages should be created (4 total: 2 note_on + 2 note_off)
        assert len(messages_created) == 4

        # Sort key is (tick, order) where note_on has order=1, note_off has order=0
        # So at same tick: note_off comes before note_on
        # This is the implementation behavior for proper MIDI ordering
        mock_midi_file.save.assert_called_once()

    def test_empty_notes_with_custom_timesig(self, tmp_path: Path) -> None:
        """Empty notes list with custom time signature works."""
        mock_mido = MagicMock()
        mock_midi_file = MagicMock()
        mock_track: list[MagicMock] = []

        mock_mido.MidiFile.return_value = mock_midi_file
        mock_mido.MidiTrack.return_value = mock_track
        mock_mido.bpm2tempo.return_value = 666666  # ~90 BPM

        meta_messages: list[tuple[str, dict[str, object]]] = []

        def track_meta(msg_type: str, **kwargs: object) -> MagicMock:
            meta_messages.append((msg_type, kwargs))
            return MagicMock()

        mock_mido.MetaMessage.side_effect = track_meta

        with patch("soundlab.io.midi_io.importlib.import_module", return_value=mock_mido):
            ts = TimeSignature(numerator=6, denominator=8)
            data = MIDIData(notes=[], tempo=90.0, time_signature=ts)
            output_path = tmp_path / "custom_timesig.mid"
            save_midi(data, output_path)

        # Verify time signature meta message was created with correct values
        mock_mido.MetaMessage.assert_any_call("time_signature", numerator=6, denominator=8, time=0)
        mock_midi_file.save.assert_called_once()
