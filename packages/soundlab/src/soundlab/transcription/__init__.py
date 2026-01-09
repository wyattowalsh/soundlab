"""Transcription utilities and models."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal, Union

from soundlab.transcription.basic_pitch import MIDITranscriber
from soundlab.transcription.crepe_backend import CREPETranscriber
from soundlab.transcription.drum_backend import DrumTranscriber
from soundlab.transcription.models import (
    DrumTranscriptionConfig,
    MIDIResult,
    TranscriptionConfig,
)
from soundlab.transcription.visualization import render_piano_roll

if TYPE_CHECKING:
    from soundlab.transcription.drum_backend import DrumConfig

# Type alias for transcriber backends
TranscriberBackend = Union[MIDITranscriber, CREPETranscriber, DrumTranscriber]

# Python version check for backend selection
_PYTHON_312_PLUS = sys.version_info >= (3, 12)


def _is_basic_pitch_available() -> bool:
    """Check if basic_pitch is available for import."""
    try:
        import importlib

        importlib.import_module("basic_pitch")
        return True
    except ImportError:
        return False


def _is_crepe_available() -> bool:
    """Check if CREPE is available for import."""
    try:
        import importlib

        importlib.import_module("crepe")
        return True
    except ImportError:
        return False


def get_transcriber(
    backend: Literal["auto", "basic_pitch", "crepe", "drum"] = "auto",
    for_drums: bool = False,
    config: TranscriptionConfig | None = None,
    drum_config: "DrumConfig | None" = None,
) -> TranscriberBackend:
    """Get appropriate transcriber backend.

    This factory function provides smart backend selection based on Python version
    and available dependencies. Basic Pitch requires Python <3.12, while CREPE and
    drum backends work with Python >=3.12.

    Args:
        backend: Backend selection strategy:
            - "auto": Auto-select based on Python version and task type
            - "basic_pitch": Use Basic Pitch (requires Python <3.12)
            - "crepe": Use CREPE pitch estimation
            - "drum": Use drum transcription backend
        for_drums: If True and backend="auto", use drum backend. Ignored when
            backend is explicitly specified.
        config: Configuration for melodic transcribers (MIDITranscriber, CREPETranscriber).
        drum_config: Configuration for DrumTranscriber. If None, defaults are used.

    Returns:
        Transcriber instance appropriate for the selected backend.

    Raises:
        ImportError: If the requested backend dependencies are not available.
        ValueError: If an invalid backend is specified.

    Examples:
        >>> # Auto-select based on Python version
        >>> transcriber = get_transcriber()

        >>> # Force CREPE backend for melodic transcription
        >>> transcriber = get_transcriber(backend="crepe")

        >>> # Get drum transcriber
        >>> transcriber = get_transcriber(for_drums=True)
        >>> # Or explicitly:
        >>> transcriber = get_transcriber(backend="drum")

        >>> # With custom configuration
        >>> from soundlab.transcription import TranscriptionConfig
        >>> config = TranscriptionConfig(onset_thresh=0.4)
        >>> transcriber = get_transcriber(config=config)
    """
    # Handle explicit drum backend request
    if backend == "drum":
        return DrumTranscriber(config=drum_config)

    # Handle auto selection with for_drums flag
    if backend == "auto" and for_drums:
        return DrumTranscriber(config=drum_config)

    # Handle explicit basic_pitch request
    if backend == "basic_pitch":
        if _PYTHON_312_PLUS:
            raise ImportError(
                "Basic Pitch is not supported on Python 3.12+. "
                "Use backend='crepe' or backend='auto' for automatic fallback."
            )
        if not _is_basic_pitch_available():
            raise ImportError(
                "basic-pitch is not installed. Install it with: pip install basic-pitch"
            )
        return MIDITranscriber(config=config)

    # Handle explicit CREPE request
    if backend == "crepe":
        if not _is_crepe_available():
            raise ImportError(
                "CREPE is not installed. Install it with: pip install crepe"
            )
        return CREPETranscriber(config=config)

    # Auto selection logic
    if backend == "auto":
        # On Python <3.12, prefer Basic Pitch if available
        if not _PYTHON_312_PLUS:
            if _is_basic_pitch_available():
                return MIDITranscriber(config=config)
            # Fall back to CREPE if basic_pitch unavailable
            if _is_crepe_available():
                return CREPETranscriber(config=config)
            raise ImportError(
                "No transcription backend available. Install either:\n"
                "  - basic-pitch: pip install basic-pitch\n"
                "  - crepe: pip install crepe"
            )

        # On Python >=3.12, use CREPE (Basic Pitch not supported)
        if _is_crepe_available():
            return CREPETranscriber(config=config)
        raise ImportError(
            "CREPE is required for transcription on Python 3.12+. "
            "Install it with: pip install crepe"
        )

    raise ValueError(
        f"Invalid backend: {backend!r}. "
        "Must be one of: 'auto', 'basic_pitch', 'crepe', 'drum'"
    )


__all__ = [
    "CREPETranscriber",
    "DrumTranscriber",
    "DrumTranscriptionConfig",
    "MIDIResult",
    "MIDITranscriber",
    "TranscriptionConfig",
    "TranscriberBackend",
    "get_transcriber",
    "render_piano_roll",
]
