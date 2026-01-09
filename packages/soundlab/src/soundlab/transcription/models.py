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
