"""Stem separation configuration and result models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class DemucsModel(StrEnum):
    """Available Demucs models."""

    HTDEMUCS = "htdemucs"
    HTDEMUCS_FT = "htdemucs_ft"
    HTDEMUCS_6S = "htdemucs_6s"
    MDX_EXTRA = "mdx_extra"
    MDX_EXTRA_Q = "mdx_extra_q"

    @property
    def stem_count(self) -> int:
        return 6 if self == DemucsModel.HTDEMUCS_6S else 4

    @property
    def stems(self) -> list[str]:
        base = ["vocals", "drums", "bass", "other"]
        if self == DemucsModel.HTDEMUCS_6S:
            return [*base, "piano", "guitar"]
        return base


class SeparationConfig(BaseModel):
    """Configuration for stem separation."""

    model_config = ConfigDict(frozen=True)

    model: DemucsModel = DemucsModel.HTDEMUCS_FT

    segment_length: Annotated[float, Field(ge=1.0, le=30.0)] = 7.8
    overlap: Annotated[float, Field(ge=0.1, le=0.9)] = 0.25
    shifts: Annotated[int, Field(ge=0, le=5)] = 1

    two_stems: str | None = None
    float32: bool = False
    int24: bool = True
    mp3_bitrate: Annotated[int, Field(ge=128, le=320)] = 320

    device: str = "auto"
    split: bool = True


class StemResult(BaseModel):
    """Result from stem separation."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    stems: dict[str, Path]
    source_path: Path
    config: SeparationConfig
    processing_time_seconds: float

    @property
    def vocals(self) -> Path | None:
        return self.stems.get("vocals")

    @property
    def instrumental(self) -> Path | None:
        """Combined non-vocal stems (computed on demand)."""
        return None
