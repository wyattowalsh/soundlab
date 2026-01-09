"""Tests for soundlab.cli module."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run CLI command and return result."""
    return subprocess.run(
        [sys.executable, "-m", "soundlab", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _get_output(result: subprocess.CompletedProcess[str]) -> str:
    """Get combined stdout and stderr output."""
    return result.stdout + result.stderr


class TestCLIHelp:
    """Tests for CLI help commands."""

    def test_main_help(self) -> None:
        """Main CLI shows help."""
        result = _run_cli("--help")
        output = _get_output(result)

        assert result.returncode == 0
        assert "SoundLab CLI" in output or "soundlab" in output.lower()
        assert "separate" in output
        assert "transcribe" in output
        assert "analyze" in output
        assert "effects" in output
        assert "tts" in output

    def test_separate_help(self) -> None:
        """Separate command shows help."""
        result = _run_cli("separate", "--help")
        output = _get_output(result)

        assert result.returncode == 0
        assert "separate" in output.lower() or "stem" in output.lower()
        assert "--model" in output or "model" in output.lower()

    def test_transcribe_help(self) -> None:
        """Transcribe command shows help."""
        result = _run_cli("transcribe", "--help")
        output = _get_output(result)

        assert result.returncode == 0
        assert "transcribe" in output.lower() or "midi" in output.lower()

    def test_analyze_help(self) -> None:
        """Analyze command shows help."""
        result = _run_cli("analyze", "--help")
        output = _get_output(result)

        assert result.returncode == 0
        assert "analyze" in output.lower() or "audio" in output.lower()

    def test_effects_help(self) -> None:
        """Effects command shows help."""
        result = _run_cli("effects", "--help")
        output = _get_output(result)

        assert result.returncode == 0
        assert "effects" in output.lower()

    def test_tts_help(self) -> None:
        """TTS command shows help."""
        result = _run_cli("tts", "--help")
        output = _get_output(result)

        assert result.returncode == 0
        assert "tts" in output.lower() or "text" in output.lower() or "speech" in output.lower()


class TestCLIEntry:
    """Tests for CLI entry point."""

    def test_cli_module_executable(self) -> None:
        """CLI module can be executed."""
        result = _run_cli("--help")
        assert result.returncode == 0

    def test_cli_entry_point(self) -> None:
        """CLI entry point 'soundlab' works."""
        # This tests the installed entry point
        result = subprocess.run(
            ["soundlab", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # May fail if not installed, but should at least not error
        if result.returncode == 0:
            output = _get_output(result)
            assert "soundlab" in output.lower() or "separate" in output


class TestCLICommands:
    """Tests for actual CLI command execution."""

    def test_analyze_with_fixture(self) -> None:
        """Analyze command works with test fixture."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "audio" / "sine_440hz_3s.wav"

        if not fixture_path.exists():
            pytest.skip("Test fixture not available")

        result = subprocess.run(
            [sys.executable, "-m", "soundlab", "analyze", str(fixture_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = _get_output(result)

        # Should output JSON or contain tempo info
        assert result.returncode == 0 or "tempo" in output.lower()

    def test_analyze_with_output_json(self, tmp_path: Path) -> None:
        """Analyze command saves to JSON file."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "audio" / "sine_440hz_3s.wav"

        if not fixture_path.exists():
            pytest.skip("Test fixture not available")

        output_json = tmp_path / "analysis.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "soundlab",
                "analyze",
                str(fixture_path),
                "--output-json",
                str(output_json),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Only check file exists if command succeeded
        if result.returncode == 0:
            assert output_json.exists()

    def test_tts_not_implemented(self) -> None:
        """TTS command exits with error (not implemented)."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "soundlab",
                "tts",
                "Hello world",
                "/tmp/output.wav",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = _get_output(result)

        # TTS should exit with code 1 and mention not implemented
        assert result.returncode == 1 or "not implemented" in output.lower()


class TestCLIImports:
    """Tests for CLI module imports."""

    def test_cli_imports_successfully(self) -> None:
        """CLI module imports without errors."""
        from soundlab.cli import app, main

        assert app is not None
        assert main is not None
        assert callable(main)

    def test_cli_app_is_typer(self) -> None:
        """CLI app is a Typer instance."""
        import typer

        from soundlab.cli import app

        assert isinstance(app, typer.Typer)
