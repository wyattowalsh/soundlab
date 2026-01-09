"""Integration tests for audio analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")

from soundlab.analysis import (
    AnalysisResult,
    KeyDetectionResult,
    LoudnessResult,
    Mode,
    MusicalKey,
    SpectralResult,
    TempoResult,
    analyze_audio,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_audio_file(tmp_path: Path) -> Path:
    """Create a sample audio file with a 440Hz sine wave."""
    import soundfile as sf

    duration = 3.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    audio_path = tmp_path / "test_sine_440hz.wav"
    sf.write(str(audio_path), audio, sample_rate)
    return audio_path


@pytest.fixture
def mock_tempo_result() -> TempoResult:
    """Create a mock tempo result."""
    return TempoResult(bpm=120.0, confidence=0.9, beats=[0.5, 1.0, 1.5, 2.0])


@pytest.fixture
def mock_key_result() -> KeyDetectionResult:
    """Create a mock key result."""
    return KeyDetectionResult(key=MusicalKey.A, mode=Mode.MINOR, confidence=0.85)


@pytest.fixture
def mock_loudness_result() -> LoudnessResult:
    """Create a mock loudness result."""
    return LoudnessResult(lufs=-14.0, dynamic_range=8.5, peak=-1.0)


@pytest.fixture
def mock_spectral_result() -> SpectralResult:
    """Create a mock spectral result."""
    return SpectralResult(centroid=440.0, bandwidth=200.0, rolloff=880.0)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAnalysisIntegration:
    """End-to-end integration tests for analyze_audio()."""

    def test_analyze_audio_returns_valid_result(
        self,
        sample_audio_file: Path,
        mock_tempo_result: TempoResult,
        mock_key_result: KeyDetectionResult,
        mock_loudness_result: LoudnessResult,
        mock_spectral_result: SpectralResult,
    ) -> None:
        """Test analyze_audio() returns valid AnalysisResult with all fields."""
        with (
            patch("soundlab.analysis.detect_tempo", return_value=mock_tempo_result),
            patch("soundlab.analysis.detect_key", return_value=mock_key_result),
            patch("soundlab.analysis.measure_loudness", return_value=mock_loudness_result),
            patch("soundlab.analysis.analyze_spectral", return_value=mock_spectral_result),
            patch("soundlab.analysis.detect_onsets", return_value=MagicMock()),
        ):
            result = analyze_audio(sample_audio_file)

        # Verify result type
        assert isinstance(result, AnalysisResult)

        # Verify all fields are populated
        assert result.tempo is not None
        assert result.key is not None
        assert result.loudness is not None
        assert result.spectral is not None

        # Verify field types
        assert isinstance(result.tempo, TempoResult)
        assert isinstance(result.key, KeyDetectionResult)
        assert isinstance(result.loudness, LoudnessResult)
        assert isinstance(result.spectral, SpectralResult)

    def test_analyze_audio_tempo_result(
        self,
        sample_audio_file: Path,
        mock_tempo_result: TempoResult,
        mock_key_result: KeyDetectionResult,
        mock_loudness_result: LoudnessResult,
        mock_spectral_result: SpectralResult,
    ) -> None:
        """Test tempo detection returns valid TempoResult."""
        with (
            patch("soundlab.analysis.detect_tempo", return_value=mock_tempo_result),
            patch("soundlab.analysis.detect_key", return_value=mock_key_result),
            patch("soundlab.analysis.measure_loudness", return_value=mock_loudness_result),
            patch("soundlab.analysis.analyze_spectral", return_value=mock_spectral_result),
            patch("soundlab.analysis.detect_onsets", return_value=MagicMock()),
        ):
            result = analyze_audio(sample_audio_file)

        assert result.tempo is not None
        assert result.tempo.bpm == 120.0
        assert result.tempo.confidence == 0.9
        assert len(result.tempo.beats) == 4

    def test_analyze_audio_key_result(
        self,
        sample_audio_file: Path,
        mock_tempo_result: TempoResult,
        mock_key_result: KeyDetectionResult,
        mock_loudness_result: LoudnessResult,
        mock_spectral_result: SpectralResult,
    ) -> None:
        """Test key detection returns valid KeyDetectionResult."""
        with (
            patch("soundlab.analysis.detect_tempo", return_value=mock_tempo_result),
            patch("soundlab.analysis.detect_key", return_value=mock_key_result),
            patch("soundlab.analysis.measure_loudness", return_value=mock_loudness_result),
            patch("soundlab.analysis.analyze_spectral", return_value=mock_spectral_result),
            patch("soundlab.analysis.detect_onsets", return_value=MagicMock()),
        ):
            result = analyze_audio(sample_audio_file)

        assert result.key is not None
        assert result.key.key == MusicalKey.A
        assert result.key.mode == Mode.MINOR
        assert result.key.confidence == 0.85
        assert result.key.name == "A minor"
        assert result.key.camelot == "8A"

    def test_analyze_audio_loudness_result(
        self,
        sample_audio_file: Path,
        mock_tempo_result: TempoResult,
        mock_key_result: KeyDetectionResult,
        mock_loudness_result: LoudnessResult,
        mock_spectral_result: SpectralResult,
    ) -> None:
        """Test loudness measurement returns valid LoudnessResult."""
        with (
            patch("soundlab.analysis.detect_tempo", return_value=mock_tempo_result),
            patch("soundlab.analysis.detect_key", return_value=mock_key_result),
            patch("soundlab.analysis.measure_loudness", return_value=mock_loudness_result),
            patch("soundlab.analysis.analyze_spectral", return_value=mock_spectral_result),
            patch("soundlab.analysis.detect_onsets", return_value=MagicMock()),
        ):
            result = analyze_audio(sample_audio_file)

        assert result.loudness is not None
        assert result.loudness.lufs == -14.0
        assert result.loudness.dynamic_range == 8.5
        assert result.loudness.peak == -1.0

    def test_analyze_audio_spectral_result(
        self,
        sample_audio_file: Path,
        mock_tempo_result: TempoResult,
        mock_key_result: KeyDetectionResult,
        mock_loudness_result: LoudnessResult,
        mock_spectral_result: SpectralResult,
    ) -> None:
        """Test spectral analysis returns valid SpectralResult."""
        with (
            patch("soundlab.analysis.detect_tempo", return_value=mock_tempo_result),
            patch("soundlab.analysis.detect_key", return_value=mock_key_result),
            patch("soundlab.analysis.measure_loudness", return_value=mock_loudness_result),
            patch("soundlab.analysis.analyze_spectral", return_value=mock_spectral_result),
            patch("soundlab.analysis.detect_onsets", return_value=MagicMock()),
        ):
            result = analyze_audio(sample_audio_file)

        assert result.spectral is not None
        assert result.spectral.centroid == 440.0
        assert result.spectral.bandwidth == 200.0
        assert result.spectral.rolloff == 880.0

    def test_analyze_audio_with_path_string(
        self,
        sample_audio_file: Path,
        mock_tempo_result: TempoResult,
        mock_key_result: KeyDetectionResult,
        mock_loudness_result: LoudnessResult,
        mock_spectral_result: SpectralResult,
    ) -> None:
        """Test analyze_audio accepts string path."""
        with (
            patch("soundlab.analysis.detect_tempo", return_value=mock_tempo_result),
            patch("soundlab.analysis.detect_key", return_value=mock_key_result),
            patch("soundlab.analysis.measure_loudness", return_value=mock_loudness_result),
            patch("soundlab.analysis.analyze_spectral", return_value=mock_spectral_result),
            patch("soundlab.analysis.detect_onsets", return_value=MagicMock()),
        ):
            result = analyze_audio(str(sample_audio_file))

        assert isinstance(result, AnalysisResult)
        assert result.tempo is not None

    def test_analysis_result_frozen(
        self,
        sample_audio_file: Path,
        mock_tempo_result: TempoResult,
        mock_key_result: KeyDetectionResult,
        mock_loudness_result: LoudnessResult,
        mock_spectral_result: SpectralResult,
    ) -> None:
        """Test AnalysisResult is immutable."""
        import pydantic

        with (
            patch("soundlab.analysis.detect_tempo", return_value=mock_tempo_result),
            patch("soundlab.analysis.detect_key", return_value=mock_key_result),
            patch("soundlab.analysis.measure_loudness", return_value=mock_loudness_result),
            patch("soundlab.analysis.analyze_spectral", return_value=mock_spectral_result),
            patch("soundlab.analysis.detect_onsets", return_value=MagicMock()),
        ):
            result = analyze_audio(sample_audio_file)

        with pytest.raises(pydantic.ValidationError):
            result.tempo = None  # type: ignore[misc]

    def test_key_detection_camelot_notation(self) -> None:
        """Test Camelot notation conversion for DJ mixing."""
        # Test major keys
        result = KeyDetectionResult(key=MusicalKey.C, mode=Mode.MAJOR, confidence=1.0)
        assert result.camelot == "8B"

        result = KeyDetectionResult(key=MusicalKey.A, mode=Mode.MINOR, confidence=1.0)
        assert result.camelot == "8A"

        result = KeyDetectionResult(key=MusicalKey.G, mode=Mode.MAJOR, confidence=1.0)
        assert result.camelot == "9B"

    def test_analysis_result_serialization(
        self,
        sample_audio_file: Path,
        mock_tempo_result: TempoResult,
        mock_key_result: KeyDetectionResult,
        mock_loudness_result: LoudnessResult,
        mock_spectral_result: SpectralResult,
    ) -> None:
        """Test AnalysisResult can be serialized to dict/JSON."""
        with (
            patch("soundlab.analysis.detect_tempo", return_value=mock_tempo_result),
            patch("soundlab.analysis.detect_key", return_value=mock_key_result),
            patch("soundlab.analysis.measure_loudness", return_value=mock_loudness_result),
            patch("soundlab.analysis.analyze_spectral", return_value=mock_spectral_result),
            patch("soundlab.analysis.detect_onsets", return_value=MagicMock()),
        ):
            result = analyze_audio(sample_audio_file)

        # Test model_dump
        result_dict = result.model_dump()
        assert isinstance(result_dict, dict)
        assert "tempo" in result_dict
        assert "key" in result_dict
        assert "loudness" in result_dict
        assert "spectral" in result_dict

        # Verify nested structures
        assert result_dict["tempo"]["bpm"] == 120.0
        assert result_dict["key"]["key"] == "A"
        assert result_dict["key"]["mode"] == "minor"

    def test_tempo_result_validation(self) -> None:
        """Test TempoResult field validation."""
        # Valid result
        result = TempoResult(bpm=120.0, confidence=0.9)
        assert result.bpm == 120.0

        # BPM must be > 0
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            TempoResult(bpm=0.0, confidence=0.9)

        with pytest.raises(pydantic.ValidationError):
            TempoResult(bpm=-10.0, confidence=0.9)

        # Confidence must be 0-1
        with pytest.raises(pydantic.ValidationError):
            TempoResult(bpm=120.0, confidence=1.5)

    def test_loudness_result_construction(self) -> None:
        """Test LoudnessResult construction."""
        result = LoudnessResult(lufs=-14.0, dynamic_range=8.5, peak=-1.0)

        assert result.lufs == -14.0
        assert result.dynamic_range == 8.5
        assert result.peak == -1.0

    def test_spectral_result_construction(self) -> None:
        """Test SpectralResult construction."""
        result = SpectralResult(centroid=440.0, bandwidth=200.0, rolloff=880.0)

        assert result.centroid == 440.0
        assert result.bandwidth == 200.0
        assert result.rolloff == 880.0

    def test_analysis_result_optional_fields(self) -> None:
        """Test AnalysisResult with optional None fields."""
        result = AnalysisResult()

        assert result.tempo is None
        assert result.key is None
        assert result.loudness is None
        assert result.spectral is None

    def test_analysis_result_partial_fields(
        self,
        mock_tempo_result: TempoResult,
        mock_key_result: KeyDetectionResult,
    ) -> None:
        """Test AnalysisResult with partial fields populated."""
        result = AnalysisResult(tempo=mock_tempo_result, key=mock_key_result)

        assert result.tempo is not None
        assert result.key is not None
        assert result.loudness is None
        assert result.spectral is None

    def test_key_detection_name_property(self) -> None:
        """Test KeyDetectionResult.name property."""
        result = KeyDetectionResult(key=MusicalKey.Cs, mode=Mode.MAJOR, confidence=0.8)
        assert result.name == "C# major"

        result = KeyDetectionResult(key=MusicalKey.Fs, mode=Mode.MINOR, confidence=0.8)
        assert result.name == "F# minor"

    def test_all_camelot_notations(self) -> None:
        """Test Camelot notation for all 24 keys."""
        test_cases = [
            (MusicalKey.C, Mode.MAJOR, "8B"),
            (MusicalKey.A, Mode.MINOR, "8A"),
            (MusicalKey.G, Mode.MAJOR, "9B"),
            (MusicalKey.E, Mode.MINOR, "9A"),
            (MusicalKey.D, Mode.MAJOR, "10B"),
            (MusicalKey.B, Mode.MINOR, "10A"),
            (MusicalKey.A, Mode.MAJOR, "11B"),
            (MusicalKey.Fs, Mode.MINOR, "11A"),
            (MusicalKey.E, Mode.MAJOR, "12B"),
            (MusicalKey.Cs, Mode.MINOR, "12A"),
            (MusicalKey.B, Mode.MAJOR, "1B"),
            (MusicalKey.Gs, Mode.MINOR, "1A"),
        ]

        for key, mode, expected_camelot in test_cases:
            result = KeyDetectionResult(key=key, mode=mode, confidence=1.0)
            assert result.camelot == expected_camelot, f"Failed for {key} {mode}"
