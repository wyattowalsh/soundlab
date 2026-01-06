"""Custom exception hierarchy for SoundLab."""

from __future__ import annotations


__all__ = [
    "SoundLabError",
    "AudioLoadError",
    "AudioFormatError",
    "ModelNotFoundError",
    "GPUMemoryError",
    "ProcessingError",
    "ConfigurationError",
    "VoiceConversionError",
]


class SoundLabError(Exception):
    """Base exception for all SoundLab errors."""
    pass


class AudioLoadError(SoundLabError):
    """Failed to load audio file."""
    pass


class AudioFormatError(SoundLabError):
    """Unsupported or invalid audio format."""
    pass


class ModelNotFoundError(SoundLabError):
    """Required model not available."""
    pass


class GPUMemoryError(SoundLabError):
    """Insufficient GPU memory for operation."""
    pass


class ProcessingError(SoundLabError):
    """Error during audio processing."""
    pass


class ConfigurationError(SoundLabError):
    """Invalid configuration."""
    pass


class VoiceConversionError(SoundLabError):
    """Error in voice conversion pipeline."""
    pass
