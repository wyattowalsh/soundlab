"""Transcription configuration and result models."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class TranscriptionConfig(BaseModel):
    """Configuration for audio-to-MIDI transcription."""

    model_config = ConfigDict(frozen=True)

    onset_thresh: Annotated[float, Field(ge=0.1, le=0.9)] = 0.5
    frame_thresh: Annotated[float, Field(ge=0.1, le=0.9)] = 0.3
    min_note_length: Annotated[float, Field(ge=0.01, le=0.2)] = 0.058
    min_freq: Annotated[float, Field(ge=20.0, le=500.0)] = 32.7
    max_freq: Annotated[float, Field(ge=1000.0, le=8000.0)] = 2093.0


class DrumTranscriptionConfig(BaseModel):
    """Configuration optimized for drum transcription.

    This configuration provides settings specifically tuned for detecting and
    classifying drum hits from audio. It uses spectral centroid thresholds to
    differentiate between kick, snare, and hi-hat sounds, and maps them to
    General MIDI drum notes.

    Attributes:
        onset_threshold: Sensitivity for detecting drum hit onsets. Lower values
            detect more hits but may include noise.
        min_note_length: Minimum duration for a detected hit in seconds. Short
            default (0.02s) accommodates fast drum rolls.
        velocity_scale: Multiplier applied to detected velocities. Values > 1.0
            boost perceived dynamics.
        kick_note: General MIDI note number for kick drum (default: 36/C1).
        snare_note: General MIDI note number for snare drum (default: 38/D1).
        hihat_closed_note: General MIDI note number for closed hi-hat (default: 42/F#1).
        hihat_open_note: General MIDI note number for open hi-hat (default: 46/A#1).
        tom_low_note: General MIDI note number for low tom (default: 45/A1).
        tom_mid_note: General MIDI note number for mid tom (default: 47/B1).
        tom_high_note: General MIDI note number for high tom (default: 50/D2).
        kick_max_centroid: Maximum spectral centroid (Hz) to classify a hit as kick.
            Sounds with centroid below this are considered kick drums.
        snare_max_centroid: Maximum spectral centroid (Hz) to classify a hit as snare.
            Sounds with centroid between kick_max and this value are snares.
            Sounds above this threshold are classified as hi-hats.
    """

    model_config = ConfigDict(frozen=True)

    # Onset detection parameters
    onset_threshold: Annotated[float, Field(ge=0.1, le=0.9)] = 0.3
    min_note_length: Annotated[float, Field(ge=0.01, le=0.2)] = 0.02
    velocity_scale: Annotated[float, Field(ge=0.5, le=2.0)] = 1.2

    # General MIDI drum mapping
    kick_note: Annotated[int, Field(ge=0, le=127)] = 36
    snare_note: Annotated[int, Field(ge=0, le=127)] = 38
    hihat_closed_note: Annotated[int, Field(ge=0, le=127)] = 42
    hihat_open_note: Annotated[int, Field(ge=0, le=127)] = 46
    tom_low_note: Annotated[int, Field(ge=0, le=127)] = 45
    tom_mid_note: Annotated[int, Field(ge=0, le=127)] = 47
    tom_high_note: Annotated[int, Field(ge=0, le=127)] = 50

    # Spectral thresholds for classification (Hz)
    kick_max_centroid: float = 300.0
    snare_max_centroid: float = 1000.0


class NoteEvent(BaseModel):
    """Single transcribed note event."""

    model_config = ConfigDict(frozen=True)

    start: Annotated[float, Field(ge=0.0)]
    end: Annotated[float, Field(ge=0.0)]
    pitch: Annotated[int, Field(ge=0, le=127)]
    velocity: Annotated[int, Field(ge=0, le=127)]

    @field_validator("end")
    @classmethod
    def end_after_start(cls, value: float, info: ValidationInfo) -> float:
        start = info.data.get("start")
        if start is not None and value < start:
            raise ValueError("end must be greater than or equal to start")
        return value


class MIDIResult(BaseModel):
    """Transcription output container."""

    model_config = ConfigDict(frozen=True)

    notes: list[NoteEvent]
    path: Path
    config: TranscriptionConfig
    processing_time: float
