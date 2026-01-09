"""Custom exception hierarchy for SoundLab."""

from __future__ import annotations


class SoundLabError(Exception):
    """Base exception for all SoundLab errors."""


class AudioLoadError(SoundLabError):
    """Failed to load audio file."""


class AudioFormatError(SoundLabError):
    """Unsupported or invalid audio format."""


class ModelNotFoundError(SoundLabError):
    """Required model not available."""


class GPUMemoryError(SoundLabError):
    """Insufficient GPU memory for operation."""


class ProcessingError(SoundLabError):
    """Error during audio processing."""


class ConfigurationError(SoundLabError):
    """Invalid configuration."""


class VoiceConversionError(SoundLabError):
    """Error in voice conversion pipeline."""
