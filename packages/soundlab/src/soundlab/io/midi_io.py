"""MIDI file I/O operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from soundlab.core.exceptions import AudioLoadError
from soundlab.core.types import PathLike
from soundlab.utils.retry import io_retry

if TYPE_CHECKING:
    pass


__all__ = ["MIDINote", "MIDIData", "load_midi", "save_midi"]


@dataclass
class MIDINote:
    """A single MIDI note event."""

    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    pitch: int         # MIDI pitch (0-127)
    velocity: int      # MIDI velocity (0-127)

    @property
    def duration(self) -> float:
        """Note duration in seconds."""
        return self.end_time - self.start_time

    @property
    def pitch_name(self) -> str:
        """Note name (e.g., 'C4', 'A#3')."""
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (self.pitch // 12) - 1
        note = note_names[self.pitch % 12]
        return f"{note}{octave}"


@dataclass
class MIDIData:
    """Container for MIDI data."""

    notes: list[MIDINote] = field(default_factory=list)
    tempo: float = 120.0  # BPM
    time_signature: tuple[int, int] = (4, 4)  # (numerator, denominator)
    duration: float = 0.0  # Total duration in seconds

    @property
    def note_count(self) -> int:
        """Number of notes."""
        return len(self.notes)

    @property
    def pitch_range(self) -> tuple[int, int]:
        """Pitch range (min, max)."""
        if not self.notes:
            return (0, 0)
        pitches = [n.pitch for n in self.notes]
        return (min(pitches), max(pitches))

    def get_notes_in_range(
        self,
        start_time: float,
        end_time: float,
    ) -> list[MIDINote]:
        """Get notes within a time range."""
        return [
            n for n in self.notes
            if n.start_time >= start_time and n.start_time < end_time
        ]


@io_retry
def load_midi(path: PathLike) -> MIDIData:
    """
    Load a MIDI file.

    Parameters
    ----------
    path
        Path to MIDI file.

    Returns
    -------
    MIDIData
        Loaded MIDI data.

    Raises
    ------
    AudioLoadError
        If the file cannot be loaded.
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi is required for MIDI I/O. Install with: pip install pretty_midi")

    path = Path(path)

    if not path.exists():
        raise AudioLoadError(f"MIDI file not found: {path}")

    logger.debug(f"Loading MIDI: {path}")

    try:
        pm = pretty_midi.PrettyMIDI(str(path))

        notes = []
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue  # Skip drum tracks for now
            for note in instrument.notes:
                notes.append(MIDINote(
                    start_time=note.start,
                    end_time=note.end,
                    pitch=note.pitch,
                    velocity=note.velocity,
                ))

        # Sort by start time
        notes.sort(key=lambda n: n.start_time)

        # Get tempo (first tempo change, or default)
        tempo = 120.0
        tempo_changes = pm.get_tempo_changes()
        if len(tempo_changes[1]) > 0:
            tempo = tempo_changes[1][0]

        # Get time signature
        time_sig = (4, 4)
        if pm.time_signature_changes:
            ts = pm.time_signature_changes[0]
            time_sig = (ts.numerator, ts.denominator)

        return MIDIData(
            notes=notes,
            tempo=tempo,
            time_signature=time_sig,
            duration=pm.get_end_time(),
        )

    except Exception as e:
        raise AudioLoadError(f"Failed to load MIDI: {path}. Error: {e}") from e


@io_retry
def save_midi(
    data: MIDIData,
    path: PathLike,
    *,
    instrument_name: str = "Acoustic Grand Piano",
    program: int = 0,
) -> Path:
    """
    Save MIDI data to file.

    Parameters
    ----------
    data
        MIDI data to save.
    path
        Output path.
    instrument_name
        Name of the instrument.
    program
        MIDI program number (0-127).

    Returns
    -------
    Path
        Path to saved file.
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi is required for MIDI I/O. Install with: pip install pretty_midi")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Saving MIDI: {path}")

    # Create PrettyMIDI object
    pm = pretty_midi.PrettyMIDI(initial_tempo=data.tempo)

    # Create instrument
    instrument = pretty_midi.Instrument(program=program, name=instrument_name)

    # Add notes
    for note in data.notes:
        pm_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=note.start_time,
            end=note.end_time,
        )
        instrument.notes.append(pm_note)

    pm.instruments.append(instrument)

    # Add time signature
    ts = pretty_midi.TimeSignature(
        numerator=data.time_signature[0],
        denominator=data.time_signature[1],
        time=0,
    )
    pm.time_signature_changes.append(ts)

    # Write file
    pm.write(str(path))

    return path
