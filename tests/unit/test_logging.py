"""Tests for logging utilities."""

import pytest

loguru = pytest.importorskip("loguru")

from soundlab.utils.logging import configure_logging


def test_configure_logging_writes_file(tmp_path: object) -> None:
    log_path = tmp_path / "soundlab.log"
    configure_logging("DEBUG", log_file=log_path)

    loguru.logger.debug("hello world")
    contents = log_path.read_text()
    assert "hello world" in contents


def test_configure_logging_respects_level(tmp_path: object) -> None:
    log_path = tmp_path / "soundlab_error.log"
    configure_logging("ERROR", log_file=log_path)

    loguru.logger.debug("skip me")
    loguru.logger.error("boom")
    contents = log_path.read_text()

    assert "boom" in contents
    assert "skip me" not in contents
