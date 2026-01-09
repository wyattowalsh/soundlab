"""Core SoundLab types and configuration."""

from __future__ import annotations

from soundlab.core.audio import AudioFormat, AudioMetadata, AudioSegment
from soundlab.core.config import SoundLabConfig
from soundlab.core.exceptions import (
    AudioFormatError,
    AudioLoadError,
    ConfigurationError,
    GPUMemoryError,
    ModelNotFoundError,
    ProcessingError,
    SoundLabError,
    VoiceConversionError,
)

__all__ = [
    "AudioFormat",
    "AudioFormatError",
    "AudioLoadError",
    "AudioMetadata",
    "AudioSegment",
    "ConfigurationError",
    "GPUMemoryError",
    "ModelNotFoundError",
    "ProcessingError",
    "SoundLabConfig",
    "SoundLabError",
    "VoiceConversionError",
]
