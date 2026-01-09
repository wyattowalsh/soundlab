"""Integration tests for pipeline QA selection and candidate scoring."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

# Import models first to ensure Pydantic deferred types are available
from soundlab.separation.models import DemucsModel, SeparationConfig
from soundlab.transcription.models import TranscriptionConfig

from soundlab.pipeline import (
    CandidatePlan,
    CandidateScore,
    PipelineConfig,
    QAConfig,
    StageCheckpoint,
    build_candidate_plans,
    choose_best_candidate,
    list_checkpoints,
    read_checkpoint,
    write_checkpoint,
)

# Rebuild models to resolve deferred annotations
PipelineConfig.model_rebuild()
CandidatePlan.model_rebuild()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Create a sample pipeline config."""
    return PipelineConfig(
        excerpt_duration=10.0,
        max_candidates=3,
    )


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    """Create a root directory for pipeline runs."""
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def sample_candidate_scores() -> list[CandidateScore]:
    """Create sample candidate scores."""
    return [
        CandidateScore(
            name="default",
            score=0.75,
            metrics={"reconstruction": 0.95, "clipping": 0.01},
            passed=True,
        ),
        CandidateScore(
            name="chunked",
            score=0.85,
            metrics={"reconstruction": 0.92, "clipping": 0.005},
            passed=True,
        ),
        CandidateScore(
            name="high_quality",
            score=0.90,
            metrics={"reconstruction": 0.98, "clipping": 0.002},
            passed=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Candidate Plan Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCandidatePlans:
    """Tests for candidate plan generation."""

    def test_build_candidate_plans_default(self, pipeline_config: PipelineConfig) -> None:
        """Test build_candidate_plans generates default plans."""
        plans = build_candidate_plans(pipeline_config)

        assert len(plans) == 3  # max_candidates = 3
        assert all(isinstance(p, CandidatePlan) for p in plans)

        # Check plan names
        plan_names = [p.name for p in plans]
        assert "default" in plan_names
        assert "chunked" in plan_names
        assert "high_quality" in plan_names

    def test_build_candidate_plans_respects_max(self) -> None:
        """Test build_candidate_plans respects max_candidates."""
        config = PipelineConfig(max_candidates=2)
        plans = build_candidate_plans(config)

        assert len(plans) == 2

    def test_build_candidate_plans_with_custom_plans(self) -> None:
        """Test build_candidate_plans returns custom plans if provided."""
        custom_plan = CandidatePlan(
            name="custom",
            separation=SeparationConfig(model=DemucsModel.MDX_EXTRA),
            notes="Custom plan",
        )
        config = PipelineConfig(candidate_plans=[custom_plan])
        plans = build_candidate_plans(config)

        assert len(plans) == 1
        assert plans[0].name == "custom"

    def test_build_candidate_plans_with_base_config(self, pipeline_config: PipelineConfig) -> None:
        """Test build_candidate_plans uses base config."""
        base = SeparationConfig(model=DemucsModel.MDX_EXTRA_Q)
        plans = build_candidate_plans(pipeline_config, base=base)

        # All plans should use the base model
        for plan in plans:
            assert plan.separation.model == DemucsModel.MDX_EXTRA_Q

    def test_candidate_plan_structure(self) -> None:
        """Test CandidatePlan has expected fields."""
        plan = CandidatePlan(
            name="test_plan",
            separation=SeparationConfig(),
            postprocess=True,
            notes="Test notes",
        )

        assert plan.name == "test_plan"
        assert isinstance(plan.separation, SeparationConfig)
        assert plan.postprocess is True
        assert plan.notes == "Test notes"
        assert plan.transcription == {}

    def test_candidate_plan_frozen(self) -> None:
        """Test CandidatePlan is immutable."""
        import pydantic

        plan = CandidatePlan(name="test", separation=SeparationConfig())

        with pytest.raises(pydantic.ValidationError):
            plan.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Candidate Scoring Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCandidateScoring:
    """Tests for candidate scoring and selection."""

    def test_choose_best_candidate_selects_highest(
        self, sample_candidate_scores: list[CandidateScore]
    ) -> None:
        """Test choose_best_candidate selects highest scoring candidate."""
        best = choose_best_candidate(sample_candidate_scores)

        assert best is not None
        assert best.name == "high_quality"
        assert best.score == 0.90

    def test_choose_best_candidate_respects_threshold(self) -> None:
        """Test choose_best_candidate respects QA threshold."""
        scores = [
            CandidateScore(name="low", score=0.5, passed=True),
            CandidateScore(name="medium", score=0.65, passed=True),
            CandidateScore(name="high", score=0.75, passed=True),
        ]
        qa = QAConfig(min_overall_score=0.7)

        best = choose_best_candidate(scores, qa=qa)

        # Only "high" meets threshold
        assert best is not None
        assert best.name == "high"

    def test_choose_best_candidate_fallback_when_none_pass(self) -> None:
        """Test choose_best_candidate falls back when none pass threshold."""
        scores = [
            CandidateScore(name="low", score=0.3, passed=True),
            CandidateScore(name="medium", score=0.4, passed=True),
        ]
        qa = QAConfig(min_overall_score=0.9)

        # Falls back to best available
        best = choose_best_candidate(scores, qa=qa)

        assert best is not None
        assert best.name == "medium"  # Highest available

    def test_choose_best_candidate_empty_scores(self) -> None:
        """Test choose_best_candidate returns None for empty scores."""
        best = choose_best_candidate([])
        assert best is None

    def test_candidate_score_structure(self) -> None:
        """Test CandidateScore has expected fields."""
        score = CandidateScore(
            name="test_candidate",
            score=0.85,
            metrics={"reconstruction": 0.95},
            passed=True,
        )

        assert score.name == "test_candidate"
        assert score.score == 0.85
        assert score.metrics["reconstruction"] == 0.95
        assert score.passed is True

    def test_candidate_score_validation(self) -> None:
        """Test CandidateScore validation."""
        import pydantic

        # Score must be 0-1
        with pytest.raises(pydantic.ValidationError):
            CandidateScore(name="invalid", score=1.5)

        with pytest.raises(pydantic.ValidationError):
            CandidateScore(name="invalid", score=-0.1)


# ---------------------------------------------------------------------------
# Resume/Checkpoint Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPipelineResume:
    """Tests for pipeline resume functionality."""

    def test_resume_skips_completed_stages(self, run_root: Path) -> None:
        """Test that resume uses checkpoints to skip completed stages."""
        run_id = "test_resume_run"

        # Simulate completed separation stage
        separation_checkpoint = StageCheckpoint(
            stage="separation",
            artifacts={"vocals": Path("/tmp/vocals.wav")},
            notes="Separation completed",
        )
        write_checkpoint(run_root, run_id, separation_checkpoint)

        # Load checkpoints
        checkpoints = list_checkpoints(run_root, run_id)
        completed_stages = {cp.stage for cp in checkpoints}

        assert "separation" in completed_stages
        # "transcription" not in completed stages, should be run
        assert "transcription" not in completed_stages

    def test_resume_with_multiple_stages(self, run_root: Path) -> None:
        """Test resume with multiple completed stages."""
        run_id = "test_multi_stage"

        # Complete multiple stages
        stages = ["separation", "transcription", "analysis"]
        for stage in stages:
            checkpoint = StageCheckpoint(stage=stage)
            write_checkpoint(run_root, run_id, checkpoint)

        checkpoints = list_checkpoints(run_root, run_id)
        completed_stages = {cp.stage for cp in checkpoints}

        assert completed_stages == {"separation", "transcription", "analysis"}

    def test_resume_from_specific_checkpoint(self, run_root: Path) -> None:
        """Test reading a specific checkpoint for resume."""
        run_id = "test_specific"

        # Write checkpoints
        separation = StageCheckpoint(
            stage="separation",
            artifacts={"vocals": Path("/tmp/vocals.wav")},
        )
        transcription = StageCheckpoint(
            stage="transcription",
            artifacts={"piano": Path("/tmp/piano.mid")},
        )

        write_checkpoint(run_root, run_id, separation)
        write_checkpoint(run_root, run_id, transcription)

        # Read specific checkpoint
        sep = read_checkpoint(run_root, run_id, "separation")
        trans = read_checkpoint(run_root, run_id, "transcription")

        assert sep is not None
        assert trans is not None
        assert "vocals" in sep.artifacts
        assert "piano" in trans.artifacts

    def test_checkpoint_artifacts_persist(self, run_root: Path) -> None:
        """Test checkpoint artifacts are persisted correctly."""
        run_id = "test_artifacts"

        checkpoint = StageCheckpoint(
            stage="effects",
            artifacts={
                "compressed": Path("/tmp/compressed.wav"),
                "reverbed": Path("/tmp/reverbed.wav"),
            },
        )
        write_checkpoint(run_root, run_id, checkpoint)

        loaded = read_checkpoint(run_root, run_id, "effects")

        assert loaded is not None
        assert len(loaded.artifacts) == 2
        assert "compressed" in loaded.artifacts
        assert "reverbed" in loaded.artifacts

    def test_pipeline_config_resume_flag(self) -> None:
        """Test PipelineConfig resume flag."""
        config_resume = PipelineConfig(resume=True)
        config_fresh = PipelineConfig(resume=False)

        assert config_resume.resume is True
        assert config_fresh.resume is False

    def test_qa_config_in_pipeline_config(self) -> None:
        """Test QAConfig embedded in PipelineConfig."""
        custom_qa = QAConfig(min_overall_score=0.8, min_midi_score=0.7)
        config = PipelineConfig(qa=custom_qa)

        assert config.qa.min_overall_score == 0.8
        assert config.qa.min_midi_score == 0.7

    def test_candidate_plan_with_transcription_config(self) -> None:
        """Test CandidatePlan with transcription configs."""
        from soundlab.transcription.models import TranscriptionConfig

        plan = CandidatePlan(
            name="with_transcription",
            separation=SeparationConfig(),
            transcription={
                "vocals": TranscriptionConfig(onset_thresh=0.6),
                "bass": TranscriptionConfig(min_freq=40.0),
            },
        )

        assert len(plan.transcription) == 2
        assert "vocals" in plan.transcription
        assert plan.transcription["vocals"].onset_thresh == 0.6
