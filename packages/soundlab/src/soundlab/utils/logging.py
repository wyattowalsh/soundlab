"""Logging configuration using loguru."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass

__all__ = ["configure_logging", "get_logger", "logger"]

# Remove default handler
logger.remove()

# Default format
DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Simple format for console
SIMPLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    *,
    simple_format: bool = True,
    colorize: bool = True,
) -> None:
    """
    Configure loguru logging for SoundLab.

    Parameters
    ----------
    level
        Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file
        Optional path to log file. If provided, logs will also be written to file.
    simple_format
        Use simplified format for console output.
    colorize
        Enable colorized output.
    """
    # Clear existing handlers
    logger.remove()

    # Choose format
    fmt = SIMPLE_FORMAT if simple_format else DEFAULT_FORMAT

    # Add console handler
    logger.add(
        sys.stderr,
        format=fmt,
        level=level.upper(),
        colorize=colorize,
    )

    # Add file handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=DEFAULT_FORMAT,
            level=level.upper(),
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )
        logger.debug(f"Logging to file: {log_file}")

    logger.debug(f"Logging configured at {level} level")


def get_logger(name: str = "soundlab") -> "logger":
    """
    Get a logger instance with the given name.

    Parameters
    ----------
    name
        Logger name (typically module name).

    Returns
    -------
    logger
        Configured logger instance.
    """
    return logger.bind(name=name)


# Configure default logging
configure_logging()
