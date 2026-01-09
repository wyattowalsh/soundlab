"""Tests for soundlab.io.export module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from soundlab.core.audio import AudioFormat, AudioSegment
from soundlab.io.export import batch_export, create_zip, export_audio


class TestExportAudio:
    """Tests for export_audio function."""

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        """export_audio creates parent directories if needed."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)

        output_path = tmp_path / "nested" / "dirs" / "output.wav"
        result = export_audio(segment, output_path)

        assert result.parent.exists()

    def test_export_returns_path(self, tmp_path: Path) -> None:
        """export_audio returns the output path."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)

        output_path = tmp_path / "output.wav"
        result = export_audio(segment, output_path)

        assert result == output_path

    def test_export_creates_file(self, tmp_path: Path) -> None:
        """export_audio creates the output file."""
        samples = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)

        output_path = tmp_path / "output.wav"
        export_audio(segment, output_path)

        assert output_path.exists()

    def test_export_with_format(self, tmp_path: Path) -> None:
        """export_audio respects format parameter."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)

        output_path = tmp_path / "output.wav"
        result = export_audio(segment, output_path, format=AudioFormat.WAV)

        assert result.exists()

    def test_export_stereo(self, tmp_path: Path) -> None:
        """export_audio handles stereo audio."""
        # Shape: (channels, samples)
        samples = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)

        output_path = tmp_path / "stereo.wav"
        result = export_audio(segment, output_path)

        assert result.exists()

    def test_export_with_lufs_requires_pyloudnorm(self, tmp_path: Path) -> None:
        """export_audio with LUFS normalization requires pyloudnorm."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)

        with patch("soundlab.io.export.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'pyloudnorm'")

            output_path = tmp_path / "output.wav"
            with pytest.raises(ImportError, match="pyloudnorm"):
                export_audio(segment, output_path, normalize_lufs=-14.0)


class TestCreateZip:
    """Tests for create_zip function."""

    def test_create_zip_with_files(self, tmp_path: Path) -> None:
        """create_zip creates archive with specified files."""
        # Create test files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        output_zip = tmp_path / "archive.zip"
        result = create_zip([file1, file2], output_zip)

        assert result == output_zip
        assert output_zip.exists()

    def test_create_zip_returns_path(self, tmp_path: Path) -> None:
        """create_zip returns the output path."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("content")

        output_zip = tmp_path / "archive.zip"
        result = create_zip([file1], output_zip)

        assert result == output_zip

    def test_create_zip_creates_parent_dirs(self, tmp_path: Path) -> None:
        """create_zip creates parent directories if needed."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("content")

        output_zip = tmp_path / "nested" / "archive.zip"
        result = create_zip([file1], output_zip)

        assert result.parent.exists()
        assert output_zip.exists()

    def test_create_zip_uses_deflate(self, tmp_path: Path) -> None:
        """create_zip uses DEFLATE compression."""
        import zipfile

        file1 = tmp_path / "file1.txt"
        file1.write_text("content" * 100)  # Compressible content

        output_zip = tmp_path / "archive.zip"
        create_zip([file1], output_zip)

        with zipfile.ZipFile(output_zip, "r") as zf:
            info = zf.getinfo("file1.txt")
            assert info.compress_type == zipfile.ZIP_DEFLATED

    def test_create_zip_preserves_filenames(self, tmp_path: Path) -> None:
        """create_zip uses arcname as filename only (no path)."""
        import zipfile

        nested = tmp_path / "subdir"
        nested.mkdir()
        file1 = nested / "file1.txt"
        file1.write_text("content")

        output_zip = tmp_path / "archive.zip"
        create_zip([file1], output_zip)

        with zipfile.ZipFile(output_zip, "r") as zf:
            names = zf.namelist()
            assert "file1.txt" in names


class TestBatchExport:
    """Tests for batch_export function."""

    def test_batch_export_creates_output_dir(self, tmp_path: Path) -> None:
        """batch_export creates output directory."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)
        segments = {"stem1": segment}

        output_dir = tmp_path / "new_dir"
        batch_export(segments, output_dir)

        assert output_dir.exists()

    def test_batch_export_exports_all_segments(self, tmp_path: Path) -> None:
        """batch_export exports all segments."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment1 = AudioSegment(samples=samples, sample_rate=44100)
        segment2 = AudioSegment(samples=samples * 2, sample_rate=44100)

        segments = {"stem1": segment1, "stem2": segment2}
        results = batch_export(segments, tmp_path)

        assert len(results) == 2
        assert "stem1" in results
        assert "stem2" in results

    def test_batch_export_returns_paths(self, tmp_path: Path) -> None:
        """batch_export returns dict of name to path."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)
        segments = {"test": segment}

        results = batch_export(segments, tmp_path)

        assert isinstance(results["test"], Path)
        assert results["test"].exists()

    def test_batch_export_adds_extension(self, tmp_path: Path) -> None:
        """batch_export adds extension if missing."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)
        segments = {"stem1": segment}  # No extension

        results = batch_export(segments, tmp_path, format="wav")

        assert results["stem1"].suffix == ".wav"

    def test_batch_export_respects_format(self, tmp_path: Path) -> None:
        """batch_export respects format parameter."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)
        segments = {"stem1": segment}

        results = batch_export(segments, tmp_path, format=AudioFormat.FLAC)

        assert results["stem1"].suffix == ".flac"

    def test_batch_export_handles_list_input(self, tmp_path: Path) -> None:
        """batch_export handles list of tuples."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)
        segments = [("stem1", segment), ("stem2", segment)]

        results = batch_export(segments, tmp_path)

        assert len(results) == 2

    def test_batch_export_preserves_extension_if_present(self, tmp_path: Path) -> None:
        """batch_export keeps extension if name already has one."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)
        segments = {"stem1.wav": segment}

        results = batch_export(segments, tmp_path, format="wav")

        # Should use the name's extension
        assert results["stem1.wav"].name == "stem1.wav"


class TestNormalizeLufs:
    """Tests for LUFS normalization in export."""

    def test_normalize_lufs_requires_pyloudnorm(self, tmp_path: Path) -> None:
        """LUFS normalization raises error if pyloudnorm unavailable."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(samples=samples, sample_rate=44100)

        with patch("soundlab.io.export.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            with pytest.raises(ImportError):
                export_audio(segment, tmp_path / "out.wav", normalize_lufs=-14.0)
