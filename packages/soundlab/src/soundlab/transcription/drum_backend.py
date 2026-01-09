"""Drum transcription backend using onset detection and spectral analysis."""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from soundlab.analysis.onsets import detect_onsets
from soundlab.io.midi_io import MIDIData, MIDINote, save_midi
from soundlab.transcription.models import MIDIResult, NoteEvent, TranscriptionConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

# General MIDI drum map (channel 10)
DRUM_KICK = 36
DRUM_SNARE = 38
DRUM_HIHAT_CLOSED = 42

# Spectral centroid thresholds (Hz)
CENTROID_LOW_MAX = 300.0
CENTROID_MID_MAX = 1000.0

# Default note duration for drum hits (seconds)
DEFAULT_DRUM_DURATION = 0.1


class DrumConfig(BaseModel):
    """Configuration for drum transcription."""

    model_config = ConfigDict(frozen=True)

    onset_thresh: Annotated[float, Field(ge=0.1, le=0.9)] = 0.5
    hop_length: int = 512
    centroid_window: Annotated[float, Field(ge=0.01, le=0.2)] = 0.05
    note_duration: Annotated[float, Field(ge=0.01, le=0.5)] = DEFAULT_DRUM_DURATION
    kick_pitch: Annotated[int, Field(ge=0, le=127)] = DRUM_KICK
    snare_pitch: Annotated[int, Field(ge=0, le=127)] = DRUM_SNARE
    hihat_pitch: Annotated[int, Field(ge=0, le=127)] = DRUM_HIHAT_CLOSED
    low_centroid_max: Annotated[float, Field(ge=50.0, le=500.0)] = CENTROID_LOW_MAX
    mid_centroid_max: Annotated[float, Field(ge=200.0, le=2000.0)] = CENTROID_MID_MAX


def _load_librosa():
    try:
        return importlib.import_module("librosa")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("librosa is required for drum transcription") from exc


def _to_mono(y: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert stereo audio to mono."""
    if y.ndim == 1:
        return y
    if y.shape[0] <= y.shape[-1]:
        return np.mean(y, axis=0).astype(np.float32)
    return np.mean(y, axis=1).astype(np.float32)


def _clamp_velocity(value: float) -> int:
    """Clamp a velocity value to valid MIDI range."""
    return int(min(127, max(1, round(value))))


def _compute_spectral_centroid(
    y: NDArray[np.float32],
    sr: int,
    onset_time: float,
    window_size: float,
) -> float:
    """Compute spectral centroid around an onset time."""
    librosa = _load_librosa()

    # Calculate sample indices for the analysis window
    center_sample = int(onset_time * sr)
    half_window = int(window_size * sr / 2)
    start_sample = max(0, center_sample - half_window)
    end_sample = min(len(y), center_sample + half_window)

    # Extract the window
    window = y[start_sample:end_sample]

    if len(window) < 256:
        # Window too small for meaningful analysis
        return 500.0  # Default to mid-range

    # Compute spectral centroid
    centroid = librosa.feature.spectral_centroid(y=window, sr=sr, n_fft=min(len(window), 2048))

    if centroid.size == 0:
        return 500.0

    return float(np.mean(centroid))


def _classify_drum_hit(centroid: float, config: DrumConfig) -> int:
    """Classify drum hit based on spectral centroid."""
    if centroid < config.low_centroid_max:
        return config.kick_pitch
    elif centroid < config.mid_centroid_max:
        return config.snare_pitch
    else:
        return config.hihat_pitch


def _compute_onset_strengths(
    y: NDArray[np.float32],
    sr: int,
    onset_times: list[float],
    hop_length: int,
) -> list[float]:
    """Compute onset strength at each onset time."""
    librosa = _load_librosa()

    # Get full onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    strengths = []
    for onset_time in onset_times:
        # Convert time to frame index
        frame = librosa.time_to_frames(onset_time, sr=sr, hop_length=hop_length)
        if 0 <= frame < len(onset_env):
            strengths.append(float(onset_env[frame]))
        else:
            strengths.append(0.0)

    return strengths


def _normalize_strengths_to_velocities(strengths: list[float]) -> list[int]:
    """Normalize onset strengths to MIDI velocities (1-127)."""
    if not strengths:
        return []

    max_strength = max(strengths) if strengths else 1.0
    if max_strength <= 0:
        return [64] * len(strengths)

    velocities = []
    for strength in strengths:
        # Scale to 1-127 range (avoid velocity 0)
        normalized = (strength / max_strength) * 126 + 1
        velocities.append(_clamp_velocity(normalized))

    return velocities


def _default_midi_path(audio_path: Path, output_dir: Path) -> Path:
    """Generate default MIDI output path."""
    return output_dir / f"{audio_path.stem}_drums.mid"


class DrumTranscriber:
    """Drum transcription using onset detection and spectral classification."""

    def __init__(self, config: DrumConfig | None = None) -> None:
        self.config = config or DrumConfig()

    def transcribe(self, audio_path: str | Path, output_dir: str | Path) -> MIDIResult:
        """Transcribe drum hits from an audio file to MIDI.

        Args:
            audio_path: Path to the input audio file.
            output_dir: Directory where MIDI file will be saved.

        Returns:
            MIDIResult containing the transcribed drum notes and file path.
        """
        librosa = _load_librosa()

        source = Path(audio_path)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()

        # Load audio
        y, sr = librosa.load(str(source), sr=None, mono=False)
        y = _to_mono(y)

        # Detect onsets using existing analysis module
        onset_result = detect_onsets(y, sr, hop_length=self.config.hop_length)
        onset_times = onset_result.timestamps

        if not onset_times:
            # No onsets detected, return empty result
            midi_path = _default_midi_path(source, output)
            midi_data = MIDIData(notes=[], tempo=120.0)
            save_midi(midi_data, midi_path)

            processing_time = time.perf_counter() - start_time
            return MIDIResult(
                notes=[],
                path=midi_path,
                config=self._to_transcription_config(),
                processing_time=processing_time,
            )

        # Compute onset strengths for velocity mapping
        strengths = _compute_onset_strengths(y, sr, onset_times, self.config.hop_length)
        velocities = _normalize_strengths_to_velocities(strengths)

        # Classify each onset by spectral centroid
        notes: list[NoteEvent] = []
        midi_notes: list[MIDINote] = []

        for i, onset_time in enumerate(onset_times):
            centroid = _compute_spectral_centroid(
                y, sr, onset_time, self.config.centroid_window
            )
            pitch = _classify_drum_hit(centroid, self.config)
            velocity = velocities[i] if i < len(velocities) else 64

            end_time = onset_time + self.config.note_duration

            notes.append(
                NoteEvent(
                    start=onset_time,
                    end=end_time,
                    pitch=pitch,
                    velocity=velocity,
                )
            )

            midi_notes.append(
                MIDINote(
                    pitch=pitch,
                    start_seconds=onset_time,
                    end_seconds=end_time,
                    velocity=velocity,
                )
            )

        # Save MIDI file
        midi_path = _default_midi_path(source, output)
        midi_data = MIDIData(notes=midi_notes, tempo=120.0)
        save_midi(midi_data, midi_path)

        processing_time = time.perf_counter() - start_time

        return MIDIResult(
            notes=notes,
            path=midi_path,
            config=self._to_transcription_config(),
            processing_time=processing_time,
        )

    def _to_transcription_config(self) -> TranscriptionConfig:
        """Convert DrumConfig to TranscriptionConfig for MIDIResult compatibility."""
        return TranscriptionConfig(
            onset_thresh=self.config.onset_thresh,
            frame_thresh=0.3,  # Not used for drums
            min_note_length=self.config.note_duration,
            min_freq=32.7,  # Not used for drums
            max_freq=2093.0,  # Not used for drums
        )


__all__ = ["DrumConfig", "DrumTranscriber"]
