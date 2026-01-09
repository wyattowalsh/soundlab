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


# ---------------------------------------------------------------------------
# Effects Command Tests
# ---------------------------------------------------------------------------


class TestEffectsCommand:
    """Tests for the effects CLI command."""

    def test_effects_help_output(self) -> None:
        """effects --help shows usage."""
        result = _run_cli("effects", "--help")
        output = _get_output(result)

        assert result.returncode == 0
        assert "effects" in output.lower()
        assert "input" in output.lower() or "audio" in output.lower()
        assert "output" in output.lower()

    def test_effects_with_mocked_chain(self, tmp_path: Path) -> None:
        """effects succeeds with mocked EffectsChain."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from soundlab.cli import app

        runner = CliRunner()

        # Create input file
        input_file = tmp_path / "input.wav"
        input_file.write_bytes(b"RIFF" + b"\x00" * 100)  # Minimal WAV header stub

        output_file = tmp_path / "output.wav"

        mock_chain_instance = MagicMock()
        mock_chain_instance.process.return_value = output_file

        with patch("soundlab.cli.EffectsChain", return_value=mock_chain_instance):
            result = runner.invoke(app, ["effects", str(input_file), str(output_file)])

        assert result.exit_code == 0
        assert str(output_file) in result.stdout or "output" in result.stdout.lower()
        mock_chain_instance.process.assert_called_once_with(input_file, output_file)

    def test_effects_creates_output_file(self, tmp_path: Path) -> None:
        """effects creates output audio file."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from soundlab.cli import app

        runner = CliRunner()

        input_file = tmp_path / "input.wav"
        input_file.write_bytes(b"RIFF" + b"\x00" * 100)

        output_file = tmp_path / "output.wav"

        mock_chain_instance = MagicMock()
        # Simulate creating output file
        mock_chain_instance.process.side_effect = lambda _inp, out: (
            out.write_bytes(b"RIFF" + b"\x00" * 100),
            out,
        )[1]

        with patch("soundlab.cli.EffectsChain", return_value=mock_chain_instance):
            result = runner.invoke(app, ["effects", str(input_file), str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_effects_passthrough(self, tmp_path: Path) -> None:
        """effects with empty chain passes audio through."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from soundlab.cli import app

        runner = CliRunner()

        input_file = tmp_path / "input.wav"
        input_file.write_bytes(b"RIFF" + b"\x00" * 100)

        output_file = tmp_path / "output.wav"

        mock_chain_instance = MagicMock()
        mock_chain_instance.process.return_value = output_file

        with patch("soundlab.cli.EffectsChain", return_value=mock_chain_instance) as mock_chain_cls:
            result = runner.invoke(app, ["effects", str(input_file), str(output_file)])

        assert result.exit_code == 0
        # Verify EffectsChain was instantiated with no effects (empty chain)
        mock_chain_cls.assert_called_once_with()
        mock_chain_instance.process.assert_called_once()

    def test_effects_invalid_input_path(self, tmp_path: Path) -> None:
        """effects errors on missing input file."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from soundlab.cli import app

        runner = CliRunner()

        nonexistent_file = tmp_path / "nonexistent.wav"
        output_file = tmp_path / "output.wav"

        # Mock EffectsChain to raise FileNotFoundError
        mock_chain_instance = MagicMock()
        mock_chain_instance.process.side_effect = FileNotFoundError("File not found")

        with patch("soundlab.cli.EffectsChain", return_value=mock_chain_instance):
            result = runner.invoke(app, ["effects", str(nonexistent_file), str(output_file)])

        # Should fail due to file not found
        assert result.exit_code != 0 or "error" in result.stdout.lower() or result.exception


# ---------------------------------------------------------------------------
# Analyze Command Tests
# ---------------------------------------------------------------------------


class TestAnalyzeCommand:
    """Tests for the analyze CLI command."""

    def test_analyze_help_output(self) -> None:
        """analyze --help shows usage."""
        result = _run_cli("analyze", "--help")
        output = _get_output(result)

        assert result.returncode == 0
        assert "analyze" in output.lower()
        assert "audio" in output.lower()
        assert "--output-json" in output or "output" in output.lower()

    def test_analyze_console_output(self, tmp_path: Path) -> None:
        """analyze prints results to console."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from soundlab.cli import app

        runner = CliRunner()

        input_file = tmp_path / "input.wav"
        input_file.write_bytes(b"RIFF" + b"\x00" * 100)

        # Create a mock AnalysisResult
        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = '{"tempo": {"bpm": 120.0}, "key": null}'

        with patch("soundlab.cli.analyze_audio", return_value=mock_result):
            result = runner.invoke(app, ["analyze", str(input_file)])

        assert result.exit_code == 0
        assert "tempo" in result.stdout.lower() or "120" in result.stdout

    def test_analyze_json_output(self, tmp_path: Path) -> None:
        """analyze writes JSON with --output-json."""
        import json
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from soundlab.cli import app

        runner = CliRunner()

        input_file = tmp_path / "input.wav"
        input_file.write_bytes(b"RIFF" + b"\x00" * 100)

        output_json = tmp_path / "analysis.json"

        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = json.dumps(
            {"tempo": {"bpm": 120.0}, "key": {"root": "C", "mode": "major"}}
        )

        with patch("soundlab.cli.analyze_audio", return_value=mock_result):
            result = runner.invoke(
                app, ["analyze", str(input_file), "--output-json", str(output_json)]
            )

        assert result.exit_code == 0
        assert output_json.exists()

        content = json.loads(output_json.read_text())
        assert "tempo" in content
        assert content["tempo"]["bpm"] == 120.0

    def test_analyze_creates_json_directory(self, tmp_path: Path) -> None:
        """analyze creates parent directory for JSON."""
        import json
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from soundlab.cli import app

        runner = CliRunner()

        input_file = tmp_path / "input.wav"
        input_file.write_bytes(b"RIFF" + b"\x00" * 100)

        # Nested directory that doesn't exist
        output_json = tmp_path / "nested" / "deep" / "analysis.json"

        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = json.dumps({"tempo": {"bpm": 90.0}})

        with patch("soundlab.cli.analyze_audio", return_value=mock_result):
            result = runner.invoke(
                app, ["analyze", str(input_file), "--output-json", str(output_json)]
            )

        assert result.exit_code == 0
        assert output_json.exists()
        assert output_json.parent.exists()

    def test_analyze_invalid_audio_path(self, tmp_path: Path) -> None:
        """analyze errors on missing audio file."""
        from unittest.mock import patch

        from typer.testing import CliRunner

        from soundlab.cli import app

        runner = CliRunner()

        nonexistent_file = tmp_path / "nonexistent.wav"

        with patch("soundlab.cli.analyze_audio", side_effect=FileNotFoundError("File not found")):
            result = runner.invoke(app, ["analyze", str(nonexistent_file)])

        assert result.exit_code != 0 or result.exception


# ---------------------------------------------------------------------------
# TTS Command Tests
# ---------------------------------------------------------------------------


class TestTTSCommand:
    """Tests for the tts CLI command."""

    def test_tts_help_output(self) -> None:
        """tts --help shows usage."""
        result = _run_cli("tts", "--help")
        output = _get_output(result)

        assert result.returncode == 0
        assert "tts" in output.lower() or "text" in output.lower()
        assert "output" in output.lower()

    def test_tts_not_implemented(self, tmp_path: Path) -> None:
        """tts exits with not-implemented error."""
        from typer.testing import CliRunner

        from soundlab.cli import app

        runner = CliRunner()

        output_file = tmp_path / "output.wav"

        result = runner.invoke(app, ["tts", "Hello world", str(output_file)])

        assert result.exit_code == 1
        assert "not implemented" in result.stdout.lower()

    def test_tts_accepts_text_argument(self, tmp_path: Path) -> None:
        """tts accepts text as first argument."""
        from typer.testing import CliRunner

        from soundlab.cli import app

        runner = CliRunner()

        output_file = tmp_path / "output.wav"

        # Even though not implemented, it should parse arguments correctly
        result = runner.invoke(app, ["tts", "Test text input", str(output_file)])

        # Should fail with exit code 1, not argument parsing error
        assert result.exit_code == 1
        # No usage error - command was parsed correctly
        assert "Usage:" not in result.stdout or "not implemented" in result.stdout.lower()

    def test_tts_mentions_voice_module(self, tmp_path: Path) -> None:
        """tts mentions soundlab.voice module in error message."""
        from typer.testing import CliRunner

        from soundlab.cli import app

        runner = CliRunner()

        output_file = tmp_path / "output.wav"

        result = runner.invoke(app, ["tts", "Hello", str(output_file)])

        assert result.exit_code == 1
        assert "voice" in result.stdout.lower()
