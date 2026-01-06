"""Basic Pitch audio-to-MIDI transcription wrapper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from soundlab.core.exceptions import ProcessingError
from soundlab.io.midi_io import MIDIData, MIDINote, save_midi
from soundlab.transcription.models import MIDIResult, NoteEvent, TranscriptionConfig
from soundlab.utils.gpu import get_device
from soundlab.utils.retry import model_retry

if TYPE_CHECKING:
    from soundlab.core.types import PathLike, ProgressCallback


__all__ = ["MIDITranscriber"]


class MIDITranscriber:
    """High-level interface for audio-to-MIDI transcription using Basic Pitch."""

    def __init__(self, config: TranscriptionConfig | None = None) -> None:
        """
        Initialize the MIDI transcriber.

        Parameters
        ----------
        config
            Transcription configuration. Uses defaults if not provided.
        """
        self.config = config or TranscriptionConfig()
        self._model = None

    @model_retry
    def transcribe(
        self,
        audio_path: PathLike,
        output_dir: PathLike | None = None,
        *,
        save_midi_file: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> MIDIResult:
        """
        Transcribe audio to MIDI.

        Parameters
        ----------
        audio_path
            Path to input audio file.
        output_dir
            Directory to save MIDI output. If None, uses audio file directory.
        save_midi_file
            Whether to save MIDI file to disk.
        progress_callback
            Optional callback for progress updates.

        Returns
        -------
        MIDIResult
            Transcription result with notes and optional MIDI file path.
        """
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
        except ImportError:
            raise ImportError(
                "basic-pitch is required for transcription. "
                "Install with: pip install basic-pitch"
            )

        audio_path = Path(audio_path)

        if output_dir is None:
            output_dir = audio_path.parent
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()

        if progress_callback:
            progress_callback(0, 100, "Loading audio...")

        logger.info(f"Transcribing: {audio_path}")

        try:
            # Run Basic Pitch prediction
            model_output, midi_data, note_events = predict(
                str(audio_path),
                onset_threshold=self.config.onset_thresh,
                frame_threshold=self.config.frame_thresh,
                minimum_note_length=self.config.minimum_note_length,
                minimum_frequency=self.config.minimum_frequency,
                maximum_frequency=self.config.maximum_frequency,
                include_pitch_bends=self.config.include_pitch_bends,
                melodia_trick=self.config.melodia_trick,
            )

            if progress_callback:
                progress_callback(70, 100, "Processing notes...")

            # Convert note events to our format
            notes = []
            for start_time_s, end_time_s, pitch, velocity, pitch_bend in note_events:
                notes.append(NoteEvent(
                    start_time=float(start_time_s),
                    end_time=float(end_time_s),
                    pitch=int(pitch),
                    velocity=int(min(127, max(1, velocity * 127))),
                    confidence=1.0,  # Basic Pitch doesn't provide per-note confidence
                ))

            # Sort by start time
            notes.sort(key=lambda n: n.start_time)

            # Save MIDI file if requested
            midi_path = None
            if save_midi_file and midi_data is not None:
                midi_path = output_dir / f"{audio_path.stem}_transcription.mid"
                midi_data.write(str(midi_path))
                logger.debug(f"Saved MIDI: {midi_path}")

            if progress_callback:
                progress_callback(100, 100, "Complete")

            processing_time = time.perf_counter() - start_time

            logger.info(
                f"Transcription complete: {len(notes)} notes in {processing_time:.1f}s"
            )

            return MIDIResult(
                notes=notes,
                midi_path=midi_path,
                source_path=audio_path,
                config=self.config,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            raise ProcessingError(f"Transcription failed: {e}") from e

    def transcribe_to_midi_data(
        self,
        audio_path: PathLike,
    ) -> MIDIData:
        """
        Transcribe audio and return MIDIData for further processing.

        Parameters
        ----------
        audio_path
            Path to input audio file.

        Returns
        -------
        MIDIData
            MIDI data container with notes.
        """
        result = self.transcribe(audio_path, save_midi_file=False)

        # Convert NoteEvents to MIDINotes
        midi_notes = [
            MIDINote(
                start_time=note.start_time,
                end_time=note.end_time,
                pitch=note.pitch,
                velocity=note.velocity,
            )
            for note in result.notes
        ]

        return MIDIData(
            notes=midi_notes,
            duration=result.duration,
        )
