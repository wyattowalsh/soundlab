"""Audio-to-MIDI transcription module for SoundLab."""

from soundlab.transcription.basic_pitch import MIDITranscriber
from soundlab.transcription.models import MIDIResult, NoteEvent, TranscriptionConfig
from soundlab.transcription.visualization import render_note_density, render_piano_roll

__all__ = [
    # Main class
    "MIDITranscriber",
    # Models
    "MIDIResult",
    "NoteEvent",
    "TranscriptionConfig",
    # Visualization
    "render_note_density",
    "render_piano_roll",
]
