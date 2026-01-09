"""Tests for pipeline orchestration models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")

from soundlab.pipeline.models import (
    CandidatePlan,
    CandidateScore,
    MIDIQAResult,
    PipelineConfig,
    QAConfig,
    RunArtifacts,
    StageCheckpoint,
    StemQAResult,
)
from soundlab.separation.models import SeparationConfig
from soundlab.transcription.models import TranscriptionConfig

# Rebuild models that have forward references in TYPE_CHECKING blocks
CandidatePlan.model_rebuild()
PipelineConfig.model_rebuild()
StageCheckpoint.model_rebuild()

# Suppress unused import warning - needed for model_rebuild
_ = TranscriptionConfig


# --------------------------------------------------------------------------- #
# QAConfig Tests
# --------------------------------------------------------------------------- #


class TestQAConfig:
    """Tests for QAConfig defaults and validation."""

    def test_default_values(self) -> None:
        """QAConfig should initialize with sensible defaults."""
        config = QAConfig()

        assert config.min_overall_score == 0.7
        assert config.max_reconstruction_error == 0.15
        assert config.max_clipping_ratio == 0.01
        assert config.min_stereo_coherence == 0.2
        assert config.min_spectral_flatness == 0.1
        assert config.max_leakage_ratio == 0.2
        assert config.min_midi_score == 0.6

    def test_custom_thresholds(self) -> None:
        """QAConfig should accept custom threshold values."""
        config = QAConfig(
            min_overall_score=0.8,
            max_reconstruction_error=0.1,
            max_clipping_ratio=0.005,
            min_stereo_coherence=0.3,
            min_spectral_flatness=0.15,
            max_leakage_ratio=0.15,
            min_midi_score=0.7,
        )

        assert config.min_overall_score == 0.8
        assert config.max_reconstruction_error == 0.1
        assert config.max_clipping_ratio == 0.005
        assert config.min_stereo_coherence == 0.3
        assert config.min_spectral_flatness == 0.15
        assert config.max_leakage_ratio == 0.15
        assert config.min_midi_score == 0.7

    def test_score_bounds_validation(self) -> None:
        """QAConfig should validate score bounds [0.0, 1.0]."""
        # Below minimum
        with pytest.raises(pydantic.ValidationError):
            QAConfig(min_overall_score=-0.1)

        # Above maximum
        with pytest.raises(pydantic.ValidationError):
            QAConfig(min_overall_score=1.5)

        # Boundary values should be valid
        config_min = QAConfig(min_overall_score=0.0)
        config_max = QAConfig(min_overall_score=1.0)

        assert config_min.min_overall_score == 0.0
        assert config_max.min_overall_score == 1.0

    def test_clipping_ratio_bounds(self) -> None:
        """Clipping ratio should be bounded [0.0, 1.0]."""
        with pytest.raises(pydantic.ValidationError):
            QAConfig(max_clipping_ratio=-0.01)

        with pytest.raises(pydantic.ValidationError):
            QAConfig(max_clipping_ratio=1.1)

    def test_reconstruction_error_non_negative(self) -> None:
        """Reconstruction error threshold should be non-negative."""
        with pytest.raises(pydantic.ValidationError):
            QAConfig(max_reconstruction_error=-0.05)

        # Zero should be valid (strict check)
        config = QAConfig(max_reconstruction_error=0.0)
        assert config.max_reconstruction_error == 0.0

    def test_frozen_model(self) -> None:
        """QAConfig should be immutable."""
        config = QAConfig()

        with pytest.raises(pydantic.ValidationError):
            config.min_overall_score = 0.9  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# CandidatePlan Tests
# --------------------------------------------------------------------------- #


class TestCandidatePlan:
    """Tests for CandidatePlan schema."""

    def test_minimal_plan(self) -> None:
        """CandidatePlan with only required fields."""
        plan = CandidatePlan(name="test", separation=SeparationConfig())

        assert plan.name == "test"
        assert plan.transcription == {}
        assert plan.postprocess is True
        assert plan.notes is None

    def test_full_plan(self) -> None:
        """CandidatePlan with all fields populated."""
        separation_config = SeparationConfig(
            segment_length=10.0,
            overlap=0.3,
            shifts=2,
        )

        plan = CandidatePlan(
            name="high_quality",
            separation=separation_config,
            transcription={},
            postprocess=False,
            notes="Custom configuration for complex audio",
        )

        assert plan.name == "high_quality"
        assert plan.separation.segment_length == 10.0
        assert plan.separation.overlap == 0.3
        assert plan.separation.shifts == 2
        assert plan.postprocess is False
        assert plan.notes == "Custom configuration for complex audio"

    def test_plan_immutability(self) -> None:
        """CandidatePlan should be frozen."""
        plan = CandidatePlan(name="test", separation=SeparationConfig())

        with pytest.raises(pydantic.ValidationError):
            plan.name = "modified"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# CandidateScore Tests
# --------------------------------------------------------------------------- #


class TestCandidateScore:
    """Tests for CandidateScore model."""

    def test_default_score(self) -> None:
        """CandidateScore with minimal fields."""
        score = CandidateScore(name="baseline", score=0.85)

        assert score.name == "baseline"
        assert score.score == 0.85
        assert score.metrics == {}
        assert score.passed is True

    def test_score_with_metrics(self) -> None:
        """CandidateScore with detailed metrics."""
        score = CandidateScore(
            name="optimized",
            score=0.92,
            metrics={
                "reconstruction_error": 0.05,
                "spectral_flatness": 0.12,
                "leakage_ratio": 0.08,
            },
            passed=True,
        )

        assert score.name == "optimized"
        assert score.score == 0.92
        assert score.metrics["reconstruction_error"] == 0.05
        assert len(score.metrics) == 3

    def test_failing_score(self) -> None:
        """CandidateScore can represent failed evaluations."""
        score = CandidateScore(name="failed", score=0.4, passed=False)

        assert score.score == 0.4
        assert score.passed is False

    def test_score_bounds_validation(self) -> None:
        """Score should be bounded [0.0, 1.0]."""
        with pytest.raises(pydantic.ValidationError):
            CandidateScore(name="invalid", score=-0.1)

        with pytest.raises(pydantic.ValidationError):
            CandidateScore(name="invalid", score=1.5)

        # Boundary values
        min_score = CandidateScore(name="min", score=0.0)
        max_score = CandidateScore(name="max", score=1.0)

        assert min_score.score == 0.0
        assert max_score.score == 1.0


# --------------------------------------------------------------------------- #
# StemQAResult Tests
# --------------------------------------------------------------------------- #


class TestStemQAResult:
    """Tests for StemQAResult model."""

    def test_minimal_result(self) -> None:
        """StemQAResult with minimal fields."""
        result = StemQAResult(score=0.75)

        assert result.score == 0.75
        assert result.metrics == {}
        assert result.stem_scores == {}
        assert result.passed is True

    def test_full_result(self) -> None:
        """StemQAResult with all fields."""
        result = StemQAResult(
            score=0.88,
            metrics={
                "reconstruction_error": 0.08,
                "clipping_ratio": 0.002,
            },
            stem_scores={
                "vocals": 0.9,
                "drums": 0.85,
                "bass": 0.87,
                "other": 0.8,
            },
            passed=True,
        )

        assert result.score == 0.88
        assert len(result.metrics) == 2
        assert len(result.stem_scores) == 4
        assert result.stem_scores["vocals"] == 0.9


# --------------------------------------------------------------------------- #
# MIDIQAResult Tests
# --------------------------------------------------------------------------- #


class TestMIDIQAResult:
    """Tests for MIDIQAResult model."""

    def test_minimal_result(self) -> None:
        """MIDIQAResult with minimal fields."""
        result = MIDIQAResult(score=0.65)

        assert result.score == 0.65
        assert result.metrics == {}
        assert result.passed is True

    def test_result_with_metrics(self) -> None:
        """MIDIQAResult with detailed metrics."""
        result = MIDIQAResult(
            score=0.8,
            metrics={
                "notes": 150.0,
                "notes_per_second": 5.0,
                "max_polyphony": 4.0,
                "pitch_range": 36.0,
            },
            passed=True,
        )

        assert result.score == 0.8
        assert result.metrics["notes"] == 150.0
        assert result.metrics["max_polyphony"] == 4.0


# --------------------------------------------------------------------------- #
# StageCheckpoint Tests
# --------------------------------------------------------------------------- #


class TestStageCheckpoint:
    """Tests for StageCheckpoint model."""

    def test_minimal_checkpoint(self) -> None:
        """StageCheckpoint with minimal fields."""
        checkpoint = StageCheckpoint(stage="separation")

        assert checkpoint.stage == "separation"
        assert checkpoint.started_at is None
        assert checkpoint.completed_at is None
        assert checkpoint.artifacts == {}
        assert checkpoint.notes is None

    def test_full_checkpoint(self) -> None:
        """StageCheckpoint with all fields."""

        checkpoint = StageCheckpoint(
            stage="transcription",
            started_at=datetime(2026, 1, 9, 10, 0, 0),
            completed_at=datetime(2026, 1, 9, 10, 5, 30),
            artifacts={"vocals_midi": Path("/output/vocals.mid")},
            notes="Transcription completed successfully",
        )

        assert checkpoint.stage == "transcription"
        assert checkpoint.started_at is not None
        assert checkpoint.completed_at is not None
        assert "vocals_midi" in checkpoint.artifacts


# --------------------------------------------------------------------------- #
# RunArtifacts Tests
# --------------------------------------------------------------------------- #


class TestRunArtifacts:
    """Tests for RunArtifacts model."""

    def test_minimal_artifacts(self) -> None:
        """RunArtifacts with only required fields."""
        artifacts = RunArtifacts(
            run_id="test_run_001",
            output_dir=Path("/outputs"),
        )

        assert artifacts.run_id == "test_run_001"
        assert artifacts.output_dir == Path("/outputs")
        assert artifacts.cache_dir is None
        assert artifacts.reports_dir is None
        assert artifacts.stems == {}
        assert artifacts.midi == {}
        assert artifacts.checkpoints == []

    def test_full_artifacts(self) -> None:
        """RunArtifacts with all fields populated."""
        artifacts = RunArtifacts(
            run_id="full_run_001",
            output_dir=Path("/outputs"),
            cache_dir=Path("/cache"),
            reports_dir=Path("/reports"),
            stems={"vocals": Path("/outputs/vocals.wav")},
            midi={"vocals": Path("/outputs/vocals.mid")},
            checkpoints=[StageCheckpoint(stage="init")],
        )

        assert artifacts.run_id == "full_run_001"
        assert artifacts.cache_dir == Path("/cache")
        assert len(artifacts.stems) == 1
        assert len(artifacts.midi) == 1
        assert len(artifacts.checkpoints) == 1


# --------------------------------------------------------------------------- #
# PipelineConfig Tests
# --------------------------------------------------------------------------- #


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    def test_default_values(self) -> None:
        """PipelineConfig should have sensible defaults."""
        config = PipelineConfig()

        assert config.run_id is None
        assert config.output_dir is None
        assert config.cache_dir is None
        assert config.excerpt_start == 0.0
        assert config.excerpt_duration == 30.0
        assert config.max_candidates == 3
        assert config.candidate_plans == []
        assert config.resume is True
        assert config.strict is False

        # Nested QAConfig defaults
        assert config.qa.min_overall_score == 0.7

    def test_custom_config(self) -> None:
        """PipelineConfig with custom values."""
        config = PipelineConfig(
            run_id="custom_run",
            output_dir=Path("/custom/output"),
            excerpt_start=10.0,
            excerpt_duration=60.0,
            max_candidates=2,
            resume=False,
            strict=True,
        )

        assert config.run_id == "custom_run"
        assert config.output_dir == Path("/custom/output")
        assert config.excerpt_start == 10.0
        assert config.excerpt_duration == 60.0
        assert config.max_candidates == 2
        assert config.resume is False
        assert config.strict is True

    def test_excerpt_duration_bounds(self) -> None:
        """Excerpt duration should be bounded [5.0, 120.0]."""
        with pytest.raises(pydantic.ValidationError):
            PipelineConfig(excerpt_duration=3.0)

        with pytest.raises(pydantic.ValidationError):
            PipelineConfig(excerpt_duration=150.0)

        # Boundary values
        config_min = PipelineConfig(excerpt_duration=5.0)
        config_max = PipelineConfig(excerpt_duration=120.0)

        assert config_min.excerpt_duration == 5.0
        assert config_max.excerpt_duration == 120.0

    def test_max_candidates_bounds(self) -> None:
        """Max candidates should be bounded [1, 5]."""
        with pytest.raises(pydantic.ValidationError):
            PipelineConfig(max_candidates=0)

        with pytest.raises(pydantic.ValidationError):
            PipelineConfig(max_candidates=6)

        # Boundary values
        config_min = PipelineConfig(max_candidates=1)
        config_max = PipelineConfig(max_candidates=5)

        assert config_min.max_candidates == 1
        assert config_max.max_candidates == 5

    def test_excerpt_start_non_negative(self) -> None:
        """Excerpt start should be non-negative."""
        with pytest.raises(pydantic.ValidationError):
            PipelineConfig(excerpt_start=-5.0)

        config = PipelineConfig(excerpt_start=0.0)
        assert config.excerpt_start == 0.0

    def test_custom_qa_config(self) -> None:
        """PipelineConfig can have custom QA thresholds."""
        config = PipelineConfig(
            qa=QAConfig(
                min_overall_score=0.85,
                max_reconstruction_error=0.1,
            )
        )

        assert config.qa.min_overall_score == 0.85
        assert config.qa.max_reconstruction_error == 0.1

    def test_with_candidate_plans(self) -> None:
        """PipelineConfig can include pre-defined candidate plans."""
        plans = [
            CandidatePlan(name="plan_a", separation=SeparationConfig()),
            CandidatePlan(name="plan_b", separation=SeparationConfig(shifts=2)),
        ]

        config = PipelineConfig(candidate_plans=plans)

        assert len(config.candidate_plans) == 2
        assert config.candidate_plans[0].name == "plan_a"
        assert config.candidate_plans[1].name == "plan_b"

    def test_frozen_model(self) -> None:
        """PipelineConfig should be immutable."""
        config = PipelineConfig()

        with pytest.raises(pydantic.ValidationError):
            config.resume = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Integration Tests
# --------------------------------------------------------------------------- #


class TestPipelineModelIntegration:
    """Integration tests for pipeline models working together."""

    def test_qa_config_in_pipeline(self) -> None:
        """QAConfig nested in PipelineConfig works correctly."""
        qa = QAConfig(
            min_overall_score=0.8,
            max_leakage_ratio=0.1,
        )

        pipeline = PipelineConfig(qa=qa, strict=True)

        assert pipeline.qa.min_overall_score == 0.8
        assert pipeline.qa.max_leakage_ratio == 0.1

    def test_candidate_plan_with_separation_config(self) -> None:
        """CandidatePlan properly contains SeparationConfig."""
        from soundlab.separation.models import DemucsModel

        sep_config = SeparationConfig(
            model=DemucsModel.HTDEMUCS_FT,
            segment_length=15.0,
            overlap=0.4,
        )

        plan = CandidatePlan(
            name="custom",
            separation=sep_config,
            notes="Testing integration",
        )

        assert plan.separation.model == DemucsModel.HTDEMUCS_FT
        assert plan.separation.segment_length == 15.0

    def test_run_artifacts_with_checkpoints(self) -> None:
        """RunArtifacts tracks multiple checkpoints."""
        checkpoints = [
            StageCheckpoint(stage="init"),
            StageCheckpoint(stage="separation"),
            StageCheckpoint(stage="transcription"),
        ]

        artifacts = RunArtifacts(
            run_id="multi_stage_run",
            output_dir=Path("/outputs"),
            checkpoints=checkpoints,
        )

        assert len(artifacts.checkpoints) == 3
        assert artifacts.checkpoints[0].stage == "init"
        assert artifacts.checkpoints[2].stage == "transcription"
