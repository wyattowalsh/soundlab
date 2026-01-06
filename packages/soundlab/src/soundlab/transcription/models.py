"""Audio-to-MIDI transcription configuration and result models."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "TranscriptionConfig",
    "NoteEvent",
    "MIDIResult",
]


class TranscriptionConfig(BaseModel):
    """Configuration for audio-to-MIDI transcription."""

    model_config = ConfigDict(frozen=True)

    # Detection thresholds
    onset_thresh: Annotated[float, Field(ge=0.1, le=0.9)] = 0.5
    frame_thresh: Annotated[float, Field(ge=0.1, le=0.9)] = 0.3

    # Note parameters
    minimum_note_length: Annotated[float, Field(ge=10.0, le=200.0)] = 58.0  # milliseconds

    # Frequency range (Hz)
    minimum_frequency: Annotated[float, Field(ge=20.0, le=500.0)] = 32.7  # C1
    maximum_frequency: Annotated[float, Field(ge=1000.0, le=8000.0)] = 2093.0  # C7

    # Output options
    include_pitch_bends: bool = False
    melodia_trick: bool = True

    # Processing
    device: str = "auto"


class NoteEvent(BaseModel):
    """A single note event from transcription."""

    model_config = ConfigDict(frozen=True)

    start_time: Annotated[float, Field(ge=0.0)]  # seconds
    end_time: Annotated[float, Field(ge=0.0)]    # seconds
    pitch: Annotated[int, Field(ge=0, le=127)]   # MIDI pitch
    velocity: Annotated[int, Field(ge=1, le=127)] = 100
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    @property
    def duration(self) -> float:
        """Note duration in seconds."""
        return self.end_time - self.start_time

    @property
    def duration_ms(self) -> float:
        """Note duration in milliseconds."""
        return self.duration * 1000

    @property
    def pitch_name(self) -> str:
        """Human-readable pitch name (e.g., 'C4', 'A#3')."""
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (self.pitch // 12) - 1
        note = note_names[self.pitch % 12]
        return f"{note}{octave}"

    @property
    def frequency(self) -> float:
        """Frequency in Hz."""
        return 440.0 * (2 ** ((self.pitch - 69) / 12))


class MIDIResult(BaseModel):
    """Result from audio-to-MIDI transcription."""

    model_config = ConfigDict(frozen=True)

    notes: list[NoteEvent]
    midi_path: Path | None = None
    source_path: Path | None = None
    config: TranscriptionConfig
    processing_time_seconds: float

    @property
    def note_count(self) -> int:
        """Total number of notes transcribed."""
        return len(self.notes)

    @property
    def duration(self) -> float:
        """Duration of the transcription in seconds."""
        if not self.notes:
            return 0.0
        return max(note.end_time for note in self.notes)

    @property
    def pitch_range(self) -> tuple[int, int]:
        """Pitch range as (min, max) MIDI pitches."""
        if not self.notes:
            return (0, 0)
        pitches = [note.pitch for note in self.notes]
        return (min(pitches), max(pitches))

    @property
    def average_velocity(self) -> float:
        """Average note velocity."""
        if not self.notes:
            return 0.0
        return sum(note.velocity for note in self.notes) / len(self.notes)

    def get_notes_in_range(
        self,
        start_time: float,
        end_time: float,
    ) -> list[NoteEvent]:
        """Get notes within a time range."""
        return [
            note for note in self.notes
            if note.start_time >= start_time and note.start_time < end_time
        ]
