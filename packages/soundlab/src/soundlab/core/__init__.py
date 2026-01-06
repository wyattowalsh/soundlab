"""Core abstractions for SoundLab."""

from soundlab.core.audio import (
    AudioFormat,
    AudioMetadata,
    AudioSegment,
    BitDepth,
    SampleRate,
)
from soundlab.core.config import SoundLabConfig, get_config, reset_config
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
from soundlab.core.types import (
    AudioArray,
    AudioProcessor,
    PathLike,
    ProgressCallback,
    SampleRateHz,
)

__all__ = [
    # Audio models
    "AudioFormat",
    "AudioMetadata",
    "AudioSegment",
    "BitDepth",
    "SampleRate",
    # Config
    "SoundLabConfig",
    "get_config",
    "reset_config",
    # Exceptions
    "AudioFormatError",
    "AudioLoadError",
    "ConfigurationError",
    "GPUMemoryError",
    "ModelNotFoundError",
    "ProcessingError",
    "SoundLabError",
    "VoiceConversionError",
    # Types
    "AudioArray",
    "AudioProcessor",
    "PathLike",
    "ProgressCallback",
    "SampleRateHz",
]