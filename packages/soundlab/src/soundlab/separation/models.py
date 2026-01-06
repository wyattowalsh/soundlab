"""Stem separation configuration and result models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


__all__ = [
    "DemucsModel",
    "SeparationConfig",
    "StemResult",
]


class DemucsModel(StrEnum):
    """Available Demucs models."""

    HTDEMUCS = "htdemucs"           # Hybrid Transformer, 4 stems
    HTDEMUCS_FT = "htdemucs_ft"     # Fine-tuned, best quality
    HTDEMUCS_6S = "htdemucs_6s"     # 6 stems (piano unreliable)
    MDX_EXTRA = "mdx_extra"          # MDX architecture
    MDX_EXTRA_Q = "mdx_extra_q"      # Quantized MDX

    @property
    def stem_count(self) -> int:
        """Number of stems produced by this model."""
        return 6 if self == DemucsModel.HTDEMUCS_6S else 4

    @property
    def stems(self) -> list[str]:
        """Names of stems produced by this model."""
        base = ["vocals", "drums", "bass", "other"]
        if self == DemucsModel.HTDEMUCS_6S:
            return base + ["piano", "guitar"]
        return base

    @property
    def description(self) -> str:
        """Human-readable description of the model."""
        descriptions = {
            DemucsModel.HTDEMUCS: "Hybrid Transformer Demucs - Fast, good quality",
            DemucsModel.HTDEMUCS_FT: "Fine-tuned HT-Demucs - Best quality, slower",
            DemucsModel.HTDEMUCS_6S: "6-stem Demucs - Includes piano/guitar (experimental)",
            DemucsModel.MDX_EXTRA: "MDX architecture - Alternative approach",
            DemucsModel.MDX_EXTRA_Q: "Quantized MDX - Faster, slightly lower quality",
        }
        return descriptions.get(self, "Unknown model")


class SeparationConfig(BaseModel):
    """Configuration for stem separation."""

    model_config = ConfigDict(frozen=True)

    # Model selection
    model: DemucsModel = DemucsModel.HTDEMUCS_FT

    # Processing parameters
    segment_length: Annotated[float, Field(ge=1.0, le=30.0)] = 7.8
    overlap: Annotated[float, Field(ge=0.1, le=0.9)] = 0.25
    shifts: Annotated[int, Field(ge=0, le=5)] = 1

    # Output options
    two_stems: str | None = None  # Extract only one stem: "vocals", "drums", "bass", "other"
    float32: bool = False
    int24: bool = True
    mp3_bitrate: Annotated[int, Field(ge=128, le=320)] = 320

    # Resource management
    device: str = "auto"  # "auto", "cuda", "cpu"
    split: bool = True    # Enable segment-based processing for long audio


class StemResult(BaseModel):
    """Result from stem separation."""

    model_config = ConfigDict(frozen=True)

    stems: dict[str, Path]           # stem_name -> file_path
    source_path: Path
    config: SeparationConfig
    processing_time_seconds: float

    @property
    def vocals(self) -> Path | None:
        """Path to vocals stem if available."""
        return self.stems.get("vocals")

    @property
    def drums(self) -> Path | None:
        """Path to drums stem if available."""
        return self.stems.get("drums")

    @property
    def bass(self) -> Path | None:
        """Path to bass stem if available."""
        return self.stems.get("bass")

    @property
    def other(self) -> Path | None:
        """Path to other/instrumental stem if available."""
        return self.stems.get("other")

    @property
    def stem_names(self) -> list[str]:
        """List of available stem names."""
        return list(self.stems.keys())
