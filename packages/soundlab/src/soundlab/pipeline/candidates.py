"""Candidate plan utilities for the pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from soundlab.pipeline.models import CandidatePlan, CandidateScore, PipelineConfig, QAConfig

if TYPE_CHECKING:
    from collections.abc import Iterable
from soundlab.separation.models import SeparationConfig


def build_candidate_plans(
    config: PipelineConfig,
    *,
    base: SeparationConfig | None = None,
) -> list[CandidatePlan]:
    """Return candidate plans for excerpt trials and full runs."""
    if config.candidate_plans:
        return list(config.candidate_plans)

    base_config = base or SeparationConfig()

    plans = [
        CandidatePlan(name="default", separation=base_config, notes="baseline settings"),
        CandidatePlan(
            name="chunked",
            separation=SeparationConfig(
                model=base_config.model,
                segment_length=min(base_config.segment_length, 10.0),
                overlap=base_config.overlap,
                shifts=base_config.shifts,
                two_stems=base_config.two_stems,
                float32=base_config.float32,
                int24=base_config.int24,
                mp3_bitrate=base_config.mp3_bitrate,
                device=base_config.device,
                split=True,
            ),
            notes="favor chunked separation for stability",
        ),
        CandidatePlan(
            name="high_quality",
            separation=SeparationConfig(
                model=base_config.model,
                segment_length=base_config.segment_length,
                overlap=min(0.5, max(base_config.overlap, 0.3)),
                shifts=min(3, max(base_config.shifts, 2)),
                two_stems=base_config.two_stems,
                float32=base_config.float32,
                int24=base_config.int24,
                mp3_bitrate=base_config.mp3_bitrate,
                device=base_config.device,
                split=base_config.split,
            ),
            notes="higher overlap + shifts",
        ),
        CandidatePlan(
            name="vocals_first",
            separation=SeparationConfig(
                model=base_config.model,
                segment_length=base_config.segment_length,
                overlap=base_config.overlap,
                shifts=base_config.shifts,
                two_stems="vocals",
                float32=base_config.float32,
                int24=base_config.int24,
                mp3_bitrate=base_config.mp3_bitrate,
                device=base_config.device,
                split=base_config.split,
            ),
            notes="stage 1: vocals vs instrumental",
        ),
    ]

    max_candidates = config.max_candidates
    return plans[:max_candidates]


def choose_best_candidate(
    scores: Iterable[CandidateScore],
    *,
    qa: QAConfig | None = None,
) -> CandidateScore | None:
    """Select the best candidate score given QA thresholds."""
    qa_config = qa or QAConfig()
    eligible = [score for score in scores if score.score >= qa_config.min_overall_score]
    pool = eligible or list(scores)
    if not pool:
        return None
    return max(pool, key=lambda item: item.score)


__all__ = ["build_candidate_plans", "choose_best_candidate"]
