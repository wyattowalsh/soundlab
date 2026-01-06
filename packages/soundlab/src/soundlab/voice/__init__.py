"""Voice generation module for SoundLab."""

from soundlab.voice.models import (
    F0Method,
    SVCConfig,
    SVCResult,
    TTSConfig,
    TTSLanguage,
    TTSResult,
    VoiceCloningRequirements,
)
from soundlab.voice.svc import VoiceConverter
from soundlab.voice.tts import TTSGenerator

__all__ = [
    # TTS
    "TTSGenerator",
    "TTSConfig",
    "TTSLanguage",
    "TTSResult",
    # SVC
    "VoiceConverter",
    "SVCConfig",
    "SVCResult",
    "F0Method",
    # Utilities
    "VoiceCloningRequirements",
]
