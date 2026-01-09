"""Basic Pitch transcription wrapper."""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from soundlab.transcription.models import MIDIResult, NoteEvent, TranscriptionConfig


def _load_basic_pitch() -> Any:
    try:
        return importlib.import_module("basic_pitch.inference")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("basic-pitch is required for MIDI transcription") from exc


def _default_midi_path(audio_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{audio_path.stem}.mid"


def _pick_midi_file(output_dir: Path, fallback: Path) -> Path:
    candidates = sorted(output_dir.glob("*.mid"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else fallback


def _clamp_velocity(value: float) -> int:
    return int(min(127, max(0, round(value))))


def _notes_from_basic_pitch(note_events: Any) -> list[NoteEvent]:
    if note_events is None:
        return []

    try:
        array = np.asarray(note_events)
    except Exception:
        array = None

    notes: list[NoteEvent] = []

    if array is not None and array.ndim >= 2 and array.shape[1] >= 4:
        for start, end, pitch, velocity in array:
            notes.append(
                NoteEvent(
                    start=float(start),
                    end=float(end),
                    pitch=round(pitch),
                    velocity=_clamp_velocity(float(velocity)),
                )
            )
        return notes

    for event in note_events:
        if isinstance(event, dict):
            start = event.get("start_time") or event.get("start") or 0.0
            end = event.get("end_time") or event.get("end") or 0.0
            pitch = event.get("pitch") or event.get("note") or 0
            velocity = event.get("velocity") or event.get("amplitude") or 0
        else:
            try:
                start, end, pitch, velocity = event
            except Exception:
                continue

        notes.append(
            NoteEvent(
                start=float(start),
                end=float(end),
                pitch=round(pitch),
                velocity=_clamp_velocity(float(velocity)),
            )
        )

    return notes


def _notes_from_midi(midi_path: Path) -> list[NoteEvent]:
    try:
        from soundlab.io.midi_io import load_midi
    except Exception:
        return []

    try:
        midi_data = load_midi(midi_path)
    except Exception:
        return []

    return [
        NoteEvent(
            start=note.start_seconds,
            end=note.end_seconds,
            pitch=note.pitch,
            velocity=note.velocity,
        )
        for note in midi_data.notes
    ]


class MIDITranscriber:
    """High-level wrapper for Basic Pitch transcription."""

    def __init__(self, config: TranscriptionConfig | None = None) -> None:
        self.config = config or TranscriptionConfig()

    def transcribe(self, audio_path: str | Path, output_dir: str | Path) -> MIDIResult:
        """Transcribe an audio file to MIDI using Basic Pitch."""
        source = Path(audio_path)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        inference = _load_basic_pitch()
        midi_path = _default_midi_path(source, output)
        note_events: Any = None

        start_time = time.perf_counter()

        if hasattr(inference, "predict"):
            _model_output, midi_data, note_events = inference.predict(
                str(source),
                onset_threshold=self.config.onset_thresh,
                frame_threshold=self.config.frame_thresh,
                minimum_note_length=self.config.min_note_length,
                minimum_frequency=self.config.min_freq,
                maximum_frequency=self.config.max_freq,
            )

            if midi_data is not None:
                try:
                    midi_data.write(str(midi_path))
                except Exception:
                    midi_path = _pick_midi_file(output, midi_path)
            else:
                midi_path = _pick_midi_file(output, midi_path)
        elif hasattr(inference, "predict_and_save"):
            inference.predict_and_save(
                [str(source)],
                str(output),
                save_midi=True,
                onset_threshold=self.config.onset_thresh,
                frame_threshold=self.config.frame_thresh,
                minimum_note_length=self.config.min_note_length,
                minimum_frequency=self.config.min_freq,
                maximum_frequency=self.config.max_freq,
            )
            midi_path = _pick_midi_file(output, midi_path)
        else:
            raise AttributeError("basic_pitch.inference lacks predict/predict_and_save")

        processing_time = time.perf_counter() - start_time

        notes = _notes_from_basic_pitch(note_events)
        if not notes:
            notes = _notes_from_midi(midi_path)

        return MIDIResult(
            notes=notes,
            path=midi_path,
            config=self.config,
            processing_time=processing_time,
        )


__all__ = ["MIDITranscriber"]
