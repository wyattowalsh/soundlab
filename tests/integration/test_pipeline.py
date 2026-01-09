"""Integration tests for pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from datetime import UTC, datetime  # noqa: F401

# Import models first to ensure Pydantic deferred types are available
from soundlab.separation.models import SeparationConfig  # noqa: F401
from soundlab.transcription.models import TranscriptionConfig  # noqa: F401

from soundlab.pipeline import (
    PipelineConfig,
    QAConfig,
    RunArtifacts,
    StageCheckpoint,
    compute_run_id,
    ensure_run_paths,
    init_run,
    list_checkpoints,
    read_checkpoint,
    run_paths,
    write_checkpoint,
)
from soundlab.pipeline.models import CandidatePlan

# Rebuild models to resolve deferred annotations - need datetime in scope
StageCheckpoint.model_rebuild()
RunArtifacts.model_rebuild()
CandidatePlan.model_rebuild()
PipelineConfig.model_rebuild()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_audio_file(tmp_path: Path) -> Path:
    """Create a sample audio file."""
    import soundfile as sf

    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    audio_path = tmp_path / "test_audio.wav"
    sf.write(str(audio_path), audio, sample_rate)
    return audio_path


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Create a sample pipeline config."""
    return PipelineConfig(
        excerpt_duration=10.0,
        max_candidates=2,
    )


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    """Create a root directory for pipeline runs."""
    root = tmp_path / "runs"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPipelineIntegration:
    """End-to-end pipeline integration tests."""

    def test_init_run_creates_artifacts(
        self,
        sample_audio_file: Path,
        pipeline_config: PipelineConfig,
        run_root: Path,
    ) -> None:
        """Test init_run creates RunArtifacts with all directories."""
        artifacts = init_run(pipeline_config, sample_audio_file, root=run_root)

        assert isinstance(artifacts, RunArtifacts)
        assert artifacts.run_id is not None
        assert artifacts.output_dir is not None
        assert artifacts.cache_dir is not None
        assert artifacts.reports_dir is not None

        # Verify directories exist
        assert artifacts.output_dir.exists()
        assert artifacts.cache_dir.exists()
        assert artifacts.reports_dir.exists()

    def test_compute_run_id_deterministic(
        self,
        sample_audio_file: Path,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test run ID is deterministic for same inputs."""
        run_id_1 = compute_run_id(sample_audio_file, pipeline_config)
        run_id_2 = compute_run_id(sample_audio_file, pipeline_config)

        assert run_id_1 == run_id_2
        assert len(run_id_1) == 64  # SHA256 hex

    def test_compute_run_id_changes_with_config(
        self,
        sample_audio_file: Path,
    ) -> None:
        """Test run ID changes with different config."""
        config_1 = PipelineConfig(excerpt_duration=10.0)
        config_2 = PipelineConfig(excerpt_duration=20.0)

        run_id_1 = compute_run_id(sample_audio_file, config_1)
        run_id_2 = compute_run_id(sample_audio_file, config_2)

        assert run_id_1 != run_id_2

    def test_run_paths_structure(self, run_root: Path) -> None:
        """Test run_paths returns correct structure."""
        run_id = "test_run_123"
        paths = run_paths(run_root, run_id)

        assert "root" in paths
        assert "cache" in paths
        assert "artifacts" in paths
        assert "reports" in paths
        assert "checkpoints" in paths

        assert paths["root"] == run_root / run_id
        assert paths["cache"] == run_root / run_id / "cache"

    def test_ensure_run_paths_creates_directories(self, run_root: Path) -> None:
        """Test ensure_run_paths creates all directories."""
        run_id = "test_run_456"
        paths = ensure_run_paths(run_root, run_id)

        for path in paths.values():
            assert path.exists()
            assert path.is_dir()

    def test_checkpoint_write_read_roundtrip(self, run_root: Path) -> None:
        """Test checkpoint write/read roundtrip."""
        from datetime import datetime

        run_id = "test_run_789"
        checkpoint = StageCheckpoint(
            stage="separation",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            artifacts={"vocals": Path("/tmp/vocals.wav")},
            notes="test checkpoint",
        )

        # Write checkpoint
        write_checkpoint(run_root, run_id, checkpoint)

        # Read checkpoint
        loaded = read_checkpoint(run_root, run_id, "separation")

        assert loaded is not None
        assert loaded.stage == "separation"
        assert loaded.notes == "test checkpoint"
        assert "vocals" in loaded.artifacts

    def test_read_checkpoint_returns_none_for_missing(self, run_root: Path) -> None:
        """Test read_checkpoint returns None for missing checkpoint."""
        result = read_checkpoint(run_root, "nonexistent_run", "nonexistent_stage")
        assert result is None

    def test_list_checkpoints_returns_all(self, run_root: Path) -> None:
        """Test list_checkpoints returns all checkpoints for a run."""
        run_id = "test_run_list"

        # Write multiple checkpoints
        stages = ["separation", "transcription", "analysis"]
        for stage in stages:
            checkpoint = StageCheckpoint(stage=stage)
            write_checkpoint(run_root, run_id, checkpoint)

        # List checkpoints
        checkpoints = list_checkpoints(run_root, run_id)

        assert len(checkpoints) == 3
        assert {cp.stage for cp in checkpoints} == set(stages)

    def test_list_checkpoints_empty_run(self, run_root: Path) -> None:
        """Test list_checkpoints returns empty list for run with no checkpoints."""
        checkpoints = list_checkpoints(run_root, "nonexistent_run")
        assert checkpoints == []

    def test_run_artifacts_stem_tracking(self, run_root: Path) -> None:
        """Test RunArtifacts tracks stem paths."""
        artifacts = RunArtifacts(
            run_id="test_run",
            output_dir=run_root / "output",
            stems={
                "vocals": run_root / "vocals.wav",
                "drums": run_root / "drums.wav",
            },
        )

        assert len(artifacts.stems) == 2
        assert "vocals" in artifacts.stems
        assert "drums" in artifacts.stems

    def test_run_artifacts_midi_tracking(self, run_root: Path) -> None:
        """Test RunArtifacts tracks MIDI paths."""
        artifacts = RunArtifacts(
            run_id="test_run",
            output_dir=run_root / "output",
            midi={
                "piano": run_root / "piano.mid",
                "bass": run_root / "bass.mid",
            },
        )

        assert len(artifacts.midi) == 2
        assert "piano" in artifacts.midi
        assert "bass" in artifacts.midi

    def test_pipeline_config_defaults(self) -> None:
        """Test PipelineConfig default values."""
        config = PipelineConfig()

        assert config.run_id is None
        assert config.output_dir is None
        assert config.excerpt_start == 0.0
        assert config.excerpt_duration == 30.0
        assert config.max_candidates == 3
        assert config.resume is True
        assert config.strict is False

    def test_pipeline_config_validation(self) -> None:
        """Test PipelineConfig validation."""
        import pydantic

        # excerpt_duration must be >= 5
        with pytest.raises(pydantic.ValidationError):
            PipelineConfig(excerpt_duration=2.0)

        # excerpt_duration must be <= 120
        with pytest.raises(pydantic.ValidationError):
            PipelineConfig(excerpt_duration=200.0)

        # max_candidates must be >= 1
        with pytest.raises(pydantic.ValidationError):
            PipelineConfig(max_candidates=0)

    def test_qa_config_defaults(self) -> None:
        """Test QAConfig default values."""
        qa = QAConfig()

        assert qa.min_overall_score == 0.7
        assert qa.max_reconstruction_error == 0.15
        assert qa.max_clipping_ratio == 0.01
        assert qa.min_stereo_coherence == 0.2
        assert qa.min_spectral_flatness == 0.1
        assert qa.max_leakage_ratio == 0.2
        assert qa.min_midi_score == 0.6

    def test_stage_checkpoint_model(self) -> None:
        """Test StageCheckpoint model."""
        checkpoint = StageCheckpoint(
            stage="effects",
            notes="Applied compression and reverb",
        )

        assert checkpoint.stage == "effects"
        assert checkpoint.notes == "Applied compression and reverb"
        assert checkpoint.started_at is None
        assert checkpoint.completed_at is None
        assert checkpoint.artifacts == {}
