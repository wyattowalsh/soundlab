"""Pipeline backend interfaces."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Protocol


class SeparatorBackend(Protocol):
    """Backend for stem separation."""

    def separate(self, mix_wav: Path, output_dir: Path) -> dict[str, Path]:
        """Separate a mix into stems."""


class TranscriberBackend(Protocol):
    """Backend for MIDI transcription."""

    def transcribe(self, stem_wav: Path, output_dir: Path) -> Path:
        """Transcribe audio into a MIDI file."""


class StemPostProcessor(Protocol):
    """Post-process separated stems."""

    def process(self, stems: dict[str, Path], output_dir: Path) -> dict[str, Path]:
        """Process stem files and return updated paths."""


class MidiPostProcessor(Protocol):
    """Post-process MIDI outputs."""

    def process(self, midi_path: Path, output_dir: Path) -> Path:
        """Process a MIDI file and return the updated path."""


class QAEvaluator(Protocol):
    """Evaluate QA metrics for stems."""

    def score(self, mix_wav: Path, stems: dict[str, Path]) -> dict[str, float]:
        """Return QA metric scores for a stem set."""


__all__ = [
    "MidiPostProcessor",
    "QAEvaluator",
    "SeparatorBackend",
    "StemPostProcessor",
    "TranscriberBackend",
]
