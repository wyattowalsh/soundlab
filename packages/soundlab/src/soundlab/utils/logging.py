"""Logging utilities for SoundLab."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger


def configure_logging(level: str, log_file: Path | None = None) -> None:
    """Configure loguru logging for console and optional file output."""
    effective_level = (level or os.getenv("SOUNDLAB_LOG_LEVEL") or "INFO").upper()
    format_string = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}"

    logger.remove()
    logger.add(sys.stderr, level=effective_level, format=format_string)

    if log_file is None:
        return

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(path, level=effective_level, format=format_string)
