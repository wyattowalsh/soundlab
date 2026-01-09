"""Cache helpers for pipeline runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path  # noqa: TC003
from typing import Any

from soundlab.pipeline.models import PipelineConfig, RunArtifacts, StageCheckpoint


def _hash_payload(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_run_id(audio_path: Path, config: PipelineConfig) -> str:
    """Compute a stable run identifier from audio metadata and config."""
    payload = {
        "path": str(audio_path),
        "mtime": audio_path.stat().st_mtime if audio_path.exists() else None,
        "size": audio_path.stat().st_size if audio_path.exists() else None,
        "config": config.model_dump(mode="json"),
    }
    return _hash_payload(json.dumps(payload, sort_keys=True))


def run_paths(root: Path, run_id: str) -> dict[str, Path]:
    """Return standard paths for a run."""
    base = root / run_id
    return {
        "root": base,
        "cache": base / "cache",
        "artifacts": base / "artifacts",
        "reports": base / "reports",
        "checkpoints": base / "checkpoints",
    }


def ensure_run_paths(root: Path, run_id: str) -> dict[str, Path]:
    """Create run directories on disk."""
    paths = run_paths(root, run_id)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def init_run(config: PipelineConfig, audio_path: Path, *, root: Path) -> RunArtifacts:
    """Initialize a run and return its artifacts container."""
    run_id = config.run_id or compute_run_id(audio_path, config)
    paths = ensure_run_paths(root, run_id)
    return RunArtifacts(
        run_id=run_id,
        output_dir=paths["artifacts"],
        cache_dir=paths["cache"],
        reports_dir=paths["reports"],
    )


def checkpoint_path(root: Path, run_id: str, stage: str) -> Path:
    """Path for a checkpoint file."""
    return (root / run_id / "checkpoints") / f"{stage}.json"


def write_checkpoint(root: Path, run_id: str, checkpoint: StageCheckpoint) -> Path:
    """Persist a checkpoint to disk."""
    path = checkpoint_path(root, run_id, checkpoint.stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(checkpoint.model_dump_json(indent=2))
    return path


def read_checkpoint(root: Path, run_id: str, stage: str) -> StageCheckpoint | None:
    """Load a checkpoint if present."""
    path = checkpoint_path(root, run_id, stage)
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return StageCheckpoint.model_validate(data)


def list_checkpoints(root: Path, run_id: str) -> list[StageCheckpoint]:
    """Load all checkpoints for a run."""
    checkpoint_dir = root / run_id / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    checkpoints = []
    for file in sorted(checkpoint_dir.glob("*.json")):
        data: dict[str, Any] = json.loads(file.read_text())
        checkpoints.append(StageCheckpoint.model_validate(data))
    return checkpoints


__all__ = [
    "checkpoint_path",
    "compute_run_id",
    "ensure_run_paths",
    "init_run",
    "list_checkpoints",
    "read_checkpoint",
    "run_paths",
    "write_checkpoint",
]
