"""Voice generation configuration and result models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


__all__ = [
    "TTSLanguage",
    "F0Method",
    "TTSConfig",
    "TTSResult",
    "SVCConfig",
    "SVCResult",
    "VoiceCloningRequirements",
]


class TTSLanguage(StrEnum):
    """Supported TTS languages for XTTS-v2."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    POLISH = "pl"
    TURKISH = "tr"
    RUSSIAN = "ru"
    DUTCH = "nl"
    CZECH = "cs"
    ARABIC = "ar"
    CHINESE = "zh-cn"
    JAPANESE = "ja"
    HUNGARIAN = "hu"
    KOREAN = "ko"
    HINDI = "hi"


class F0Method(StrEnum):
    """Pitch detection methods for voice conversion."""

    RMVPE = "rmvpe"      # Recommended - robust multi-channel pitch estimation
    CREPE = "crepe"      # Neural network based
    HARVEST = "harvest"  # High-quality traditional
    PM = "pm"           # Praat-style pitch marks
    DIO = "dio"         # Fast traditional method


class TTSConfig(BaseModel):
    """Configuration for text-to-speech generation."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, max_length=10000)
    language: TTSLanguage = TTSLanguage.ENGLISH

    # Voice cloning
    speaker_wav: Path | None = None  # Reference audio for voice cloning (6-30s)

    # Generation parameters
    temperature: Annotated[float, Field(ge=0.1, le=1.0)] = 0.7
    length_penalty: Annotated[float, Field(ge=0.5, le=2.0)] = 1.0
    repetition_penalty: Annotated[float, Field(ge=1.0, le=10.0)] = 2.0
    top_k: Annotated[int, Field(ge=1, le=100)] = 50
    top_p: Annotated[float, Field(ge=0.0, le=1.0)] = 0.85

    # Speed adjustment
    speed: Annotated[float, Field(ge=0.5, le=2.0)] = 1.0

    # Processing
    device: str = "auto"


class TTSResult(BaseModel):
    """Result from TTS generation."""

    model_config = ConfigDict(frozen=True)

    audio_path: Path
    text: str
    language: TTSLanguage
    duration_seconds: float
    processing_time_seconds: float

    @property
    def words_per_minute(self) -> float:
        """Estimated speaking rate."""
        word_count = len(self.text.split())
        if self.duration_seconds <= 0:
            return 0.0
        return (word_count / self.duration_seconds) * 60


class SVCConfig(BaseModel):
    """Configuration for singing voice conversion (RVC)."""

    model_config = ConfigDict(frozen=True)

    # Model selection
    model_path: Path | None = None  # Path to RVC model
    index_path: Path | None = None  # Path to feature index file

    # Pitch parameters
    pitch_shift_semitones: Annotated[int, Field(ge=-12, le=12)] = 0
    f0_method: F0Method = F0Method.RMVPE

    # Voice conversion parameters
    index_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75
    filter_radius: Annotated[int, Field(ge=0, le=7)] = 3
    resample_sr: Annotated[int, Field(ge=0, le=48000)] = 0  # 0 = no resampling
    rms_mix_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25

    # Protection
    protect_rate: Annotated[float, Field(ge=0.0, le=0.5)] = 0.33

    # Processing
    device: str = "auto"


class SVCResult(BaseModel):
    """Result from singing voice conversion."""

    model_config = ConfigDict(frozen=True)

    audio_path: Path
    source_path: Path
    model_name: str
    pitch_shift: int
    processing_time_seconds: float
    duration_seconds: float


class VoiceCloningRequirements(BaseModel):
    """Requirements for voice cloning reference audio."""

    model_config = ConfigDict(frozen=True)

    min_duration_seconds: float = 6.0
    max_duration_seconds: float = 30.0
    recommended_duration_seconds: float = 15.0
    max_speakers: int = 1
    max_background_noise_db: float = -40.0

    @property
    def guidelines(self) -> list[str]:
        """Get voice cloning guidelines."""
        return [
            f"Duration: {self.min_duration_seconds}-{self.max_duration_seconds} seconds (recommended: {self.recommended_duration_seconds}s)",
            "Single speaker only",
            "Clear speech without music",
            "Minimal background noise",
            "Consistent microphone distance",
            "Natural speaking pace",
        ]
