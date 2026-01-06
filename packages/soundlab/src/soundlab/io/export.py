"""Export and batch processing utilities."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from soundlab.core.audio import AudioFormat, AudioSegment
from soundlab.core.types import PathLike
from soundlab.io.audio_io import save_audio

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "export_audio",
    "batch_export",
    "create_zip",
    "create_export_package",
]


def export_audio(
    segment: AudioSegment,
    path: PathLike,
    *,
    format: AudioFormat | None = None,
    normalize_lufs: float | None = None,
) -> Path:
    """
    Export an audio segment with optional normalization.

    Parameters
    ----------
    segment
        Audio segment to export.
    path
        Output path.
    format
        Output format. If None, detected from path extension.
    normalize_lufs
        Target loudness in LUFS. If provided, audio will be normalized.

    Returns
    -------
    Path
        Path to exported file.
    """
    path = Path(path)

    # Apply loudness normalization if requested
    if normalize_lufs is not None:
        try:
            import pyloudnorm as pyln
            import numpy as np

            meter = pyln.Meter(segment.sample_rate)
            samples = segment.samples

            # Handle mono/stereo
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)

            # Transpose for pyloudnorm (samples, channels)
            samples_t = samples.T

            current_lufs = meter.integrated_loudness(samples_t)

            if not np.isinf(current_lufs):
                gain_db = normalize_lufs - current_lufs
                gain_linear = 10 ** (gain_db / 20)
                samples = samples * gain_linear

                # Prevent clipping
                max_val = np.max(np.abs(samples))
                if max_val > 1.0:
                    samples = samples / max_val * 0.99

                segment = AudioSegment(
                    samples=samples.squeeze() if samples.shape[0] == 1 else samples,
                    sample_rate=segment.sample_rate,
                    source_path=segment.source_path,
                    metadata=segment.metadata,
                )
                logger.debug(f"Normalized to {normalize_lufs} LUFS (gain: {gain_db:.1f} dB)")
        except ImportError:
            logger.warning("pyloudnorm not installed, skipping normalization")

    return save_audio(segment, path, format=format)


def batch_export(
    segments: Sequence[tuple[AudioSegment, str]],
    output_dir: PathLike,
    *,
    format: AudioFormat = AudioFormat.WAV,
    normalize_lufs: float | None = None,
) -> list[Path]:
    """
    Export multiple audio segments.

    Parameters
    ----------
    segments
        Sequence of (segment, filename) tuples.
    output_dir
        Output directory.
    format
        Output format for all files.
    normalize_lufs
        Target loudness in LUFS.

    Returns
    -------
    list[Path]
        Paths to exported files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for segment, filename in segments:
        # Ensure correct extension
        stem = Path(filename).stem
        path = output_dir / f"{stem}.{format.value}"

        exported_path = export_audio(
            segment,
            path,
            format=format,
            normalize_lufs=normalize_lufs,
        )
        paths.append(exported_path)
        logger.debug(f"Exported: {exported_path}")

    logger.info(f"Batch exported {len(paths)} files to {output_dir}")
    return paths


def create_zip(
    files: Sequence[PathLike],
    output_path: PathLike,
    *,
    base_dir: PathLike | None = None,
) -> Path:
    """
    Create a ZIP archive from files.

    Parameters
    ----------
    files
        Files to include in the archive.
    output_path
        Output ZIP file path.
    base_dir
        Base directory for relative paths in the archive.

    Returns
    -------
    Path
        Path to created ZIP file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_dir = Path(base_dir) if base_dir else None

    logger.debug(f"Creating ZIP: {output_path}")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.warning(f"File not found, skipping: {file_path}")
                continue

            # Determine archive name
            if base_dir and file_path.is_relative_to(base_dir):
                arcname = file_path.relative_to(base_dir)
            else:
                arcname = file_path.name

            zf.write(file_path, arcname)
            logger.debug(f"Added to ZIP: {arcname}")

    logger.info(f"Created ZIP with {len(files)} files: {output_path}")
    return output_path


def create_export_package(
    output_dir: PathLike,
    *,
    include_patterns: list[str] | None = None,
    zip_name: str = "soundlab_export.zip",
) -> Path:
    """
    Create an export package from an output directory.

    Parameters
    ----------
    output_dir
        Directory containing files to package.
    include_patterns
        Glob patterns to include. If None, includes all files.
    zip_name
        Name for the ZIP file.

    Returns
    -------
    Path
        Path to created ZIP file.
    """
    output_dir = Path(output_dir)

    if include_patterns is None:
        include_patterns = ["*"]

    files = []
    for pattern in include_patterns:
        files.extend(output_dir.rglob(pattern))

    # Filter to only files
    files = [f for f in files if f.is_file()]

    zip_path = output_dir / zip_name
    return create_zip(files, zip_path, base_dir=output_dir)
