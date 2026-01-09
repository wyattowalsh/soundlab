"""Pipeline orchestration utilities."""

from __future__ import annotations

from soundlab.pipeline.cache import (
    checkpoint_path,
    compute_run_id,
    ensure_run_paths,
    init_run,
    list_checkpoints,
    read_checkpoint,
    run_paths,
    write_checkpoint,
)
from soundlab.pipeline.candidates import build_candidate_plans, choose_best_candidate
from soundlab.pipeline.interfaces import (
    MidiPostProcessor,
    QAEvaluator,
    SeparatorBackend,
    StemPostProcessor,
    TranscriberBackend,
)
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
from soundlab.pipeline.postprocess import clean_stems, cleanup_midi_notes, mono_amt_exports
from soundlab.pipeline.qa import score_midi, score_separation

__all__ = [
    "CandidatePlan",
    "CandidateScore",
    "MIDIQAResult",
    "MidiPostProcessor",
    "PipelineConfig",
    "QAConfig",
    "QAEvaluator",
    "RunArtifacts",
    "SeparatorBackend",
    "StageCheckpoint",
    "StemPostProcessor",
    "StemQAResult",
    "TranscriberBackend",
    "build_candidate_plans",
    "checkpoint_path",
    "choose_best_candidate",
    "clean_stems",
    "cleanup_midi_notes",
    "compute_run_id",
    "ensure_run_paths",
    "init_run",
    "list_checkpoints",
    "mono_amt_exports",
    "read_checkpoint",
    "run_paths",
    "score_midi",
    "score_separation",
    "write_checkpoint",
]
