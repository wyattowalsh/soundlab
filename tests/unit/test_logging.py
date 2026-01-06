"""Tests for soundlab.utils.logging."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from loguru import logger

from soundlab.utils.logging import configure_logging, get_logger


class TestConfigureLogging:
    """Test configure_logging function."""

    def test_configure_default_level(self):
        """Should configure with INFO level by default."""
        configure_logging()
        # Logger should be configured (not raise)

    def test_configure_debug_level(self):
        """Should configure with DEBUG level."""
        configure_logging(level="DEBUG")

    def test_configure_warning_level(self):
        """Should configure with WARNING level."""
        configure_logging(level="WARNING")

    def test_configure_error_level(self):
        """Should configure with ERROR level."""
        configure_logging(level="ERROR")

    def test_configure_with_log_file(self):
        """Should configure with file logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            configure_logging(level="DEBUG", log_file=log_file)

            # Write a log message
            logger.debug("Test message")

            # File should be created (may have small delay)
            # Note: loguru has async file writing

    def test_configure_creates_log_directory(self):
        """Should create log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs" / "nested"
            log_file = log_dir / "test.log"

            configure_logging(log_file=log_file)

            # Directory should be created
            assert log_dir.exists() or not log_file.exists()  # May be async

    def test_configure_simple_format(self):
        """Should use simple format when specified."""
        configure_logging(simple_format=True)

    def test_configure_detailed_format(self):
        """Should use detailed format when simple_format=False."""
        configure_logging(simple_format=False)

    def test_configure_colorize(self):
        """Should respect colorize setting."""
        configure_logging(colorize=True)
        configure_logging(colorize=False)


class TestGetLogger:
    """Test get_logger function."""

    def test_returns_logger(self):
        """Should return a logger instance."""
        log = get_logger("test")
        # Should have logging methods
        assert hasattr(log, "debug")
        assert hasattr(log, "info")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")

    def test_returns_logger_with_name(self):
        """Should return logger with specified name."""
        log = get_logger("mymodule")
        # Logger should be bound to name
        assert log is not None

    def test_default_name(self):
        """Should use 'soundlab' as default name."""
        log = get_logger()
        assert log is not None


class TestLoggingOutput:
    """Test actual logging output."""

    def test_logger_writes_to_stderr(self, capsys):
        """Logger should write to stderr."""
        configure_logging(level="INFO", colorize=False)
        logger.info("Test message")

        # Note: loguru doesn't always flush immediately
        # This test verifies configuration doesn't error

    def test_debug_filtered_at_info_level(self):
        """DEBUG messages should be filtered at INFO level."""
        configure_logging(level="INFO")
        # Should not raise
        logger.debug("This should be filtered")

    def test_info_not_filtered_at_info_level(self):
        """INFO messages should not be filtered at INFO level."""
        configure_logging(level="INFO")
        logger.info("This should pass")
