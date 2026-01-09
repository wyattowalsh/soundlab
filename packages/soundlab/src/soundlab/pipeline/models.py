"""Pipeline orchestration models."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from datetime import datetime

    from soundlab.separation.models import SeparationConfig
    from soundlab.transcription.models import TranscriptionConfig


class QAConfig(BaseModel):
    """Configuration thresholds for QA scoring."""

    model_config = ConfigDict(frozen=True)

    min_overall_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    max_reconstruction_error: Annotated[float, Field(ge=0.0)] = 0.15
    max_clipping_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.01
    min_stereo_coherence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    min_spectral_flatness: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    max_leakage_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    min_midi_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6


class CandidatePlan(BaseModel):
    """Plan describing a candidate separation/transcription strategy."""

    model_config = ConfigDict(frozen=True)

    name: str
    separation: SeparationConfig
    transcription: dict[str, TranscriptionConfig] = Field(default_factory=dict)
    postprocess: bool = True
    notes: str | None = None


class CandidateScore(BaseModel):
    """Score for a candidate plan."""

    model_config = ConfigDict(frozen=True)

    name: str
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    metrics: dict[str, float] = Field(default_factory=dict)
    passed: bool = True


class StageCheckpoint(BaseModel):
    """Checkpoint metadata for a pipeline stage."""

    model_config = ConfigDict(frozen=True)

    stage: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    artifacts: dict[str, Path] = Field(default_factory=dict)
    notes: str | None = None


class StemQAResult(BaseModel):
    """QA metrics for stem separation."""

    model_config = ConfigDict(frozen=True)

    score: Annotated[float, Field(ge=0.0, le=1.0)]
    metrics: dict[str, float] = Field(default_factory=dict)
    stem_scores: dict[str, float] = Field(default_factory=dict)
    passed: bool = True


class MIDIQAResult(BaseModel):
    """QA metrics for transcription output."""

    model_config = ConfigDict(frozen=True)

    score: Annotated[float, Field(ge=0.0, le=1.0)]
    metrics: dict[str, float] = Field(default_factory=dict)
    passed: bool = True


class RunArtifacts(BaseModel):
    """Artifact paths tracked for a pipeline run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    output_dir: Path
    cache_dir: Path | None = None
    reports_dir: Path | None = None
    stems: dict[str, Path] = Field(default_factory=dict)
    midi: dict[str, Path] = Field(default_factory=dict)
    checkpoints: list[StageCheckpoint] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    run_id: str | None = None
    output_dir: Path | None = None
    cache_dir: Path | None = None
    excerpt_start: Annotated[float, Field(ge=0.0)] = 0.0
    excerpt_duration: Annotated[float, Field(ge=5.0, le=120.0)] = 30.0
    max_candidates: Annotated[int, Field(ge=1, le=5)] = 3
    candidate_plans: list[CandidatePlan] = Field(default_factory=list)
    qa: QAConfig = Field(default_factory=QAConfig)
    resume: bool = True
    strict: bool = False


__all__ = [
    "CandidatePlan",
    "CandidateScore",
    "MIDIQAResult",
    "PipelineConfig",
    "QAConfig",
    "RunArtifacts",
    "StageCheckpoint",
    "StemQAResult",
]
