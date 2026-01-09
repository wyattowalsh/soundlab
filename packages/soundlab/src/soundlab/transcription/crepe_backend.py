"""CREPE-based pitch transcription backend."""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from soundlab.io.midi_io import MIDIData, MIDINote, save_midi
from soundlab.transcription.models import MIDIResult, NoteEvent, TranscriptionConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _load_crepe() -> Any:
    """Load the CREPE module, raising a helpful error if unavailable."""
    try:
        return importlib.import_module("crepe")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "crepe is required for CREPE-based transcription. "
            "Install it with: pip install crepe"
        ) from exc


def _load_librosa() -> Any:
    """Load librosa for onset detection."""
    try:
        return importlib.import_module("librosa")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "librosa is required for onset detection. "
            "Install it with: pip install librosa"
        ) from exc


def _to_mono(y: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert audio to mono if necessary."""
    if y.ndim == 1:
        return y
    if y.shape[0] <= y.shape[-1]:
        return np.mean(y, axis=0).astype(np.float32)
    return np.mean(y, axis=1).astype(np.float32)


def _freq_to_midi(frequency: float) -> int:
    """Convert frequency in Hz to MIDI pitch number.

    Uses the standard formula: midi = 69 + 12 * log2(f / 440)
    Returns a clamped value between 0 and 127.
    """
    if frequency <= 0:
        return 0
    midi_pitch = 69 + 12 * np.log2(frequency / 440.0)
    return int(min(127, max(0, round(midi_pitch))))


def _clamp_velocity(value: float) -> int:
    """Clamp velocity value to valid MIDI range [0, 127]."""
    return int(min(127, max(0, round(value))))


def _default_midi_path(audio_path: Path, output_dir: Path) -> Path:
    """Generate default MIDI output path from audio source."""
    return output_dir / f"{audio_path.stem}_crepe.mid"


class CREPETranscriber:
    """CREPE-based pitch transcription backend.

    This transcriber uses librosa for onset detection to segment audio into
    note regions, then uses CREPE for accurate pitch estimation within each
    region. Results are converted to MIDI format.

    Implements the TranscriberBackend protocol.
    """

    def __init__(self, config: TranscriptionConfig | None = None) -> None:
        """Initialize the CREPE transcriber.

        Args:
            config: Transcription configuration. Defaults to TranscriptionConfig().
        """
        self.config = config or TranscriptionConfig()

    def _detect_onsets(
        self,
        y: NDArray[np.float32],
        sr: int,
        hop_length: int = 512,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Detect onset times and their strengths.

        Args:
            y: Audio samples (mono).
            sr: Sample rate.
            hop_length: Hop length for onset detection.

        Returns:
            Tuple of (onset_times, onset_strengths) arrays.
        """
        librosa = _load_librosa()

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            backtrack=True,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

        # Get onset strengths at detected onset frames
        onset_strengths = onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])

        return onset_times, onset_strengths

    def _estimate_pitch_for_region(
        self,
        y: NDArray[np.float32],
        sr: int,
        start_time: float,
        end_time: float,
        crepe_module: Any,
    ) -> tuple[float, float]:
        """Estimate pitch for a note region using CREPE.

        Args:
            y: Full audio samples (mono).
            sr: Sample rate.
            start_time: Start time of the region in seconds.
            end_time: End time of the region in seconds.
            crepe_module: The loaded CREPE module.

        Returns:
            Tuple of (frequency_hz, confidence) for the region.
        """
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Ensure we have valid sample range
        start_sample = max(0, start_sample)
        end_sample = min(len(y), end_sample)

        if end_sample <= start_sample:
            return 0.0, 0.0

        region = y[start_sample:end_sample]

        # CREPE needs at least some samples to work with
        if len(region) < sr // 20:  # At least 50ms of audio
            return 0.0, 0.0

        # Run CREPE prediction on the region
        try:
            _time, frequency, confidence, _activation = crepe_module.predict(
                region, sr, viterbi=True
            )
        except Exception:
            return 0.0, 0.0

        if len(frequency) == 0:
            return 0.0, 0.0

        # Weight frequencies by confidence and compute weighted median
        valid_mask = (frequency > 0) & (confidence > 0.3)
        if not np.any(valid_mask):
            return 0.0, 0.0

        valid_freqs = frequency[valid_mask]
        valid_conf = confidence[valid_mask]

        # Use confidence-weighted mean frequency
        weighted_freq = float(np.average(valid_freqs, weights=valid_conf))
        mean_confidence = float(np.mean(valid_conf))

        return weighted_freq, mean_confidence

    def transcribe(self, stem_wav: Path, output_dir: Path) -> Path:
        """Transcribe audio to MIDI using CREPE pitch estimation.

        This method implements the TranscriberBackend protocol interface.

        Args:
            stem_wav: Path to the input audio file.
            output_dir: Directory for output MIDI file.

        Returns:
            Path to the generated MIDI file.
        """
        result = self.transcribe_full(stem_wav, output_dir)
        return result.path

    def transcribe_full(
        self, audio_path: str | Path, output_dir: str | Path
    ) -> MIDIResult:
        """Transcribe audio to MIDI with full result details.

        Args:
            audio_path: Path to the input audio file.
            output_dir: Directory for output MIDI file.

        Returns:
            MIDIResult containing transcribed notes, output path, and metadata.
        """
        source = Path(audio_path)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        crepe = _load_crepe()
        librosa = _load_librosa()

        start_time = time.perf_counter()

        # Load audio
        y, sr = librosa.load(str(source), sr=None, mono=True)
        y = y.astype(np.float32)
        duration = len(y) / sr

        # Detect onsets
        onset_times, onset_strengths = self._detect_onsets(y, sr)

        if len(onset_times) == 0:
            # No onsets detected, return empty result
            midi_path = _default_midi_path(source, output)
            midi_data = MIDIData(notes=[], tempo=120.0)
            save_midi(midi_data, midi_path)
            processing_time = time.perf_counter() - start_time
            return MIDIResult(
                notes=[],
                path=midi_path,
                config=self.config,
                processing_time=processing_time,
            )

        # Normalize onset strengths to velocity range
        if len(onset_strengths) > 0 and np.max(onset_strengths) > 0:
            normalized_strengths = onset_strengths / np.max(onset_strengths)
        else:
            normalized_strengths = np.ones_like(onset_times)

        # Create note regions: each onset to the next onset (or end of audio)
        notes: list[NoteEvent] = []
        midi_notes: list[MIDINote] = []

        for i, onset in enumerate(onset_times):
            # Determine end time (next onset or end of audio)
            if i + 1 < len(onset_times):
                note_end = float(onset_times[i + 1])
            else:
                note_end = duration

            # Apply minimum note length constraint
            if note_end - float(onset) < self.config.min_note_length:
                continue

            # Estimate pitch for this region
            freq, confidence = self._estimate_pitch_for_region(
                y, sr, float(onset), note_end, crepe
            )

            # Skip if no valid pitch or low confidence
            if freq <= 0 or confidence < 0.3:
                continue

            # Filter by frequency range
            if freq < self.config.min_freq or freq > self.config.max_freq:
                continue

            # Convert to MIDI pitch
            pitch = _freq_to_midi(freq)
            if pitch < 0 or pitch > 127:
                continue

            # Map onset strength to velocity (scale to 40-127 range for musicality)
            strength = normalized_strengths[i] if i < len(normalized_strengths) else 0.5
            velocity = _clamp_velocity(40 + strength * 87)

            note_event = NoteEvent(
                start=float(onset),
                end=note_end,
                pitch=pitch,
                velocity=velocity,
            )
            notes.append(note_event)

            midi_note = MIDINote(
                pitch=pitch,
                start_seconds=float(onset),
                end_seconds=note_end,
                velocity=velocity,
            )
            midi_notes.append(midi_note)

        # Save MIDI file
        midi_path = _default_midi_path(source, output)
        midi_data = MIDIData(notes=midi_notes, tempo=120.0)
        save_midi(midi_data, midi_path)

        processing_time = time.perf_counter() - start_time

        return MIDIResult(
            notes=notes,
            path=midi_path,
            config=self.config,
            processing_time=processing_time,
        )


__all__ = ["CREPETranscriber"]
