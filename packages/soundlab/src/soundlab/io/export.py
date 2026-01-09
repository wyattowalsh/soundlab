"""Export utilities for audio and bundles."""

from __future__ import annotations

import importlib
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import numpy as np

from soundlab.core.audio import AudioFormat, AudioSegment
from soundlab.io.audio_io import save_audio


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_extension(fmt: str | AudioFormat | None) -> str:
    if isinstance(fmt, AudioFormat):
        return fmt.value
    if fmt:
        return fmt.strip().lstrip(".").lower()
    return "wav"


def _normalize_lufs(segment: AudioSegment, target_lufs: float) -> AudioSegment:
    try:
        pyln = importlib.import_module("pyloudnorm")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("pyloudnorm is required for LUFS normalization") from exc

    samples = segment.samples
    data = samples.T if samples.ndim == 2 else samples

    meter = pyln.Meter(segment.sample_rate)
    loudness = meter.integrated_loudness(data)
    normalized = pyln.normalize.loudness(data, loudness, target_lufs)

    if normalized.ndim == 2:
        normalized = normalized.T

    return AudioSegment(
        samples=np.asarray(normalized, dtype=np.float32),
        sample_rate=segment.sample_rate,
        source_path=segment.source_path,
        metadata=segment.metadata,
    )


def export_audio(
    segment: AudioSegment,
    path: str | Path,
    format: str | AudioFormat | None = None,
    normalize_lufs: float | None = None,
) -> Path:
    """Export an AudioSegment to disk."""
    target = Path(path)
    _ensure_parent(target)
    export_segment = segment

    if normalize_lufs is not None:
        export_segment = _normalize_lufs(segment, normalize_lufs)

    save_audio(export_segment, target, format=format)
    return target


def create_zip(files: Iterable[str | Path], output_path: str | Path) -> Path:
    """Create a zip archive containing the provided files."""
    output = Path(output_path)
    _ensure_parent(output)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            path = Path(file)
            archive.write(path, arcname=path.name)

    return output


def batch_export(
    segments: Mapping[str, AudioSegment] | Iterable[tuple[str, AudioSegment]],
    output_dir: str | Path,
    format: str | AudioFormat | None = None,
    normalize_lufs: float | None = None,
) -> dict[str, Path]:
    """Export multiple audio segments to the output directory."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    extension = _normalize_extension(format)

    if isinstance(segments, Mapping):
        items = cast("Iterable[tuple[str, AudioSegment]]", segments.items())
    else:
        items = cast("Iterable[tuple[str, AudioSegment]]", segments)
    results: dict[str, Path] = {}
    for name, segment in items:
        filename = Path(name)
        if not filename.suffix:
            filename = filename.with_suffix(f".{extension}")
        target = output / filename.name
        export_audio(segment, target, format=format, normalize_lufs=normalize_lufs)
        results[name] = target

    return results


__all__ = ["batch_export", "create_zip", "export_audio"]
