"""MIDI input/output helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MIDINote(BaseModel):
    """Representation of a single MIDI note event."""

    pitch: int = Field(ge=0, le=127)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    velocity: int = Field(ge=0, le=127)

    @model_validator(mode="after")
    def _validate_times(self) -> MIDINote:
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must be >= start_seconds")
        return self


class TimeSignature(BaseModel):
    """Time signature for a MIDI file."""

    numerator: int = Field(ge=1)
    denominator: int = Field(ge=1)


class MIDIData(BaseModel):
    """Container for MIDI data."""

    notes: list[MIDINote] = Field(default_factory=list)
    tempo: float = Field(default=120.0, gt=0)
    time_signature: TimeSignature | None = None


def _load_mido() -> Any:
    try:
        return importlib.import_module("mido")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("mido is required for MIDI I/O") from exc


def _default_time_signature() -> TimeSignature:
    return TimeSignature(numerator=4, denominator=4)


def load_midi(path: str | Path) -> MIDIData:
    """Load a MIDI file into a MIDIData structure."""
    mido = _load_mido()
    midi = mido.MidiFile(str(path))
    ticks_per_beat = midi.ticks_per_beat
    tempo = mido.bpm2tempo(120.0)
    tempo_bpm = 120.0
    time_signature = _default_time_signature()
    notes: list[MIDINote] = []
    active: dict[int, list[tuple[float, int]]] = {}
    current_time = 0.0

    for message in mido.merge_tracks(midi.tracks):
        if message.time:
            current_time += mido.tick2second(message.time, ticks_per_beat, tempo)

        if message.type == "set_tempo":
            tempo = message.tempo
            tempo_bpm = float(mido.tempo2bpm(tempo))
            continue

        if message.type == "time_signature":
            time_signature = TimeSignature(
                numerator=message.numerator,
                denominator=message.denominator,
            )
            continue

        if message.type == "note_on" and message.velocity > 0:
            active.setdefault(message.note, []).append((current_time, message.velocity))
            continue

        if message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            starts = active.get(message.note)
            if not starts:
                continue
            start_time, velocity = starts.pop(0)
            notes.append(
                MIDINote(
                    pitch=message.note,
                    start_seconds=start_time,
                    end_seconds=current_time,
                    velocity=velocity,
                )
            )

    return MIDIData(notes=notes, tempo=tempo_bpm, time_signature=time_signature)


def _seconds_to_ticks(seconds: float, ticks_per_beat: int, tempo: int, mido: Any) -> int:
    if seconds <= 0:
        return 0
    ticks = mido.second2tick(seconds, ticks_per_beat, tempo)
    return round(ticks)


def save_midi(data: MIDIData, path: str | Path) -> None:
    """Save MIDIData to a MIDI file."""
    mido = _load_mido()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ticks_per_beat = 480
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    tempo = mido.bpm2tempo(data.tempo)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    time_signature = data.time_signature or _default_time_signature()
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=time_signature.numerator,
            denominator=time_signature.denominator,
            time=0,
        )
    )

    events: list[tuple[int, int, Any]] = []
    for note in data.notes:
        start_tick = _seconds_to_ticks(note.start_seconds, ticks_per_beat, tempo, mido)
        end_tick = _seconds_to_ticks(note.end_seconds, ticks_per_beat, tempo, mido)
        events.append(
            (start_tick, 1, mido.Message("note_on", note=note.pitch, velocity=note.velocity))
        )
        events.append((end_tick, 0, mido.Message("note_off", note=note.pitch, velocity=0)))

    events.sort(key=lambda item: (item[0], item[1]))

    last_tick = 0
    for tick, _order, message in events:
        delta = max(0, tick - last_tick)
        message.time = delta
        track.append(message)
        last_tick = tick

    midi.save(output)


__all__ = ["MIDIData", "MIDINote", "TimeSignature", "load_midi", "save_midi"]
