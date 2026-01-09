"""Voice synthesis and conversion models."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class TTSConfig(BaseModel):
    """Text-to-speech configuration."""

    model_config = ConfigDict(frozen=True)

    text: str
    language: str = "en"
    speaker_wav: Path | None = None
    temperature: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    speed: Annotated[float, Field(ge=0.5, le=2.0)] = 1.0


class TTSResult(BaseModel):
    """Text-to-speech output."""

    model_config = ConfigDict(frozen=True)

    audio_path: Path
    processing_time: Annotated[float, Field(ge=0.0)]


class SVCConfig(BaseModel):
    """Singing voice conversion configuration."""

    model_config = ConfigDict(frozen=True)

    pitch_shift: Annotated[float, Field(ge=-24.0, le=24.0)] = 0.0
    f0_method: str = "rmvpe"
    index_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    protect_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5


class SVCResult(BaseModel):
    """Singing voice conversion output."""

    model_config = ConfigDict(frozen=True)

    audio_path: Path
    processing_time: Annotated[float, Field(ge=0.0)]


__all__ = ["SVCConfig", "SVCResult", "TTSConfig", "TTSResult"]
