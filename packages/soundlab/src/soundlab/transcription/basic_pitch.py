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


def _basic_pitch_model_path(inference: Any) -> Any:
    """Resolve ``model_or_model_path`` for basic-pitch 0.3.

    Coded against basic-pitch 0.3.1 ``basic_pitch.inference.predict`` (the 0.3.0
    signature matches; 0.3.2 only adds a default):

        predict(
            audio_path,
            model_or_model_path,  # required in 0.3.0 and 0.3.1
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=127.70,  # milliseconds; divided by 1000
            minimum_frequency=None,
            maximum_frequency=None,
            multiple_pitch_bends=False,
            melodia_trick=True,
            debug_file=None,
            midi_tempo=120,
        )

    ``ICASSP_2022_MODEL_PATH`` lives on the ``basic_pitch`` package in 0.3.0 and
    0.3.1. 0.3.2 also imports it into ``basic_pitch.inference``.
    """
    package_path = None
    try:
        package = importlib.import_module("basic_pitch")
    except Exception:
        package = None
    else:
        package_path = getattr(package, "ICASSP_2022_MODEL_PATH", None)
    if package_path is not None:
        return package_path

    inference_path = getattr(inference, "ICASSP_2022_MODEL_PATH", None)
    if inference_path is not None:
        return inference_path

    raise ImportError(
        "basic-pitch 0.3 predict requires model_or_model_path, "
        "but ICASSP_2022_MODEL_PATH was not found."
    )


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
        for start, end, pitch, amplitude in array[:, :4]:
            velocity = float(amplitude)
            if 0.0 <= velocity <= 1.0:
                velocity *= 127.0
            notes.append(
                NoteEvent(
                    start=float(start),
                    end=float(end),
                    pitch=round(float(pitch)),
                    velocity=_clamp_velocity(velocity),
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
            model_path = _basic_pitch_model_path(inference)
            minimum_note_length_ms = self.config.minimum_note_length_ms()
            _model_output, midi_data, note_events = inference.predict(
                str(source),
                model_or_model_path=model_path,
                onset_threshold=self.config.onset_thresh,
                frame_threshold=self.config.frame_thresh,
                minimum_note_length=minimum_note_length_ms,
                minimum_frequency=self.config.min_freq,
                maximum_frequency=self.config.max_freq,
            )

            if midi_data is None:
                raise RuntimeError(f"basic-pitch did not return MIDI for {source}")
            # A failed write must surface. Do not substitute another .mid in the directory.
            midi_data.write(str(midi_path))
        elif hasattr(inference, "predict_and_save"):
            model_path = _basic_pitch_model_path(inference)
            minimum_note_length_ms = self.config.minimum_note_length_ms()
            # 0.3.0/0.3.1 require the model path and the save flags with no defaults.
            inference.predict_and_save(
                [str(source)],
                str(output),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=model_path,
                onset_threshold=self.config.onset_thresh,
                frame_threshold=self.config.frame_thresh,
                minimum_note_length=minimum_note_length_ms,
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
