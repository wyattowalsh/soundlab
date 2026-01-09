"""Tests for musical key detection and Camelot conversion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
pydantic = pytest.importorskip("pydantic")

from soundlab.analysis.models import KeyDetectionResult, Mode, MusicalKey

# ---------------------------------------------------------------------------
# Camelot Wheel Conversion Tests
# ---------------------------------------------------------------------------


class TestCamelotConversion:
    """Test Camelot wheel notation conversion for all keys."""

    @pytest.mark.parametrize(
        ("key", "mode", "expected_camelot"),
        [
            # Major keys (B position)
            (MusicalKey.C, Mode.MAJOR, "8B"),
            (MusicalKey.G, Mode.MAJOR, "9B"),
            (MusicalKey.D, Mode.MAJOR, "10B"),
            (MusicalKey.A, Mode.MAJOR, "11B"),
            (MusicalKey.E, Mode.MAJOR, "12B"),
            (MusicalKey.B, Mode.MAJOR, "1B"),
            (MusicalKey.Fs, Mode.MAJOR, "2B"),
            (MusicalKey.Cs, Mode.MAJOR, "3B"),
            (MusicalKey.Gs, Mode.MAJOR, "4B"),
            (MusicalKey.Ds, Mode.MAJOR, "5B"),
            (MusicalKey.As, Mode.MAJOR, "6B"),
            (MusicalKey.F, Mode.MAJOR, "7B"),
            # Minor keys (A position)
            (MusicalKey.A, Mode.MINOR, "8A"),
            (MusicalKey.E, Mode.MINOR, "9A"),
            (MusicalKey.B, Mode.MINOR, "10A"),
            (MusicalKey.Fs, Mode.MINOR, "11A"),
            (MusicalKey.Cs, Mode.MINOR, "12A"),
            (MusicalKey.Gs, Mode.MINOR, "1A"),
            (MusicalKey.Ds, Mode.MINOR, "2A"),
            (MusicalKey.As, Mode.MINOR, "3A"),
            (MusicalKey.F, Mode.MINOR, "4A"),
            (MusicalKey.C, Mode.MINOR, "5A"),
            (MusicalKey.G, Mode.MINOR, "6A"),
            (MusicalKey.D, Mode.MINOR, "7A"),
        ],
    )
    def test_camelot_conversion(
        self,
        key: MusicalKey,
        mode: Mode,
        expected_camelot: str,
    ) -> None:
        result = KeyDetectionResult(key=key, mode=mode, confidence=0.9)
        assert result.camelot == expected_camelot

    def test_camelot_harmonic_mixing_pairs(self) -> None:
        """Test that relative major/minor keys share the same Camelot number."""
        # A minor is the relative minor of C major - both are 8
        a_minor = KeyDetectionResult(key=MusicalKey.A, mode=Mode.MINOR, confidence=0.9)
        c_major = KeyDetectionResult(key=MusicalKey.C, mode=Mode.MAJOR, confidence=0.9)

        assert a_minor.camelot[:-1] == c_major.camelot[:-1]  # Same number
        assert a_minor.camelot[-1] == "A"
        assert c_major.camelot[-1] == "B"


# ---------------------------------------------------------------------------
# Key Detection Result Model Tests
# ---------------------------------------------------------------------------


class TestKeyDetectionResult:
    """Test KeyDetectionResult model properties."""

    def test_name_property(self) -> None:
        result = KeyDetectionResult(key=MusicalKey.A, mode=Mode.MINOR, confidence=0.85)
        assert result.name == "A minor"

        result2 = KeyDetectionResult(key=MusicalKey.Fs, mode=Mode.MAJOR, confidence=0.75)
        assert result2.name == "F# major"

    def test_confidence_bounds(self) -> None:
        # Valid confidence
        result = KeyDetectionResult(key=MusicalKey.C, mode=Mode.MAJOR, confidence=0.5)
        assert result.confidence == 0.5

        # Boundary values
        result_min = KeyDetectionResult(key=MusicalKey.C, mode=Mode.MAJOR, confidence=0.0)
        assert result_min.confidence == 0.0

        result_max = KeyDetectionResult(key=MusicalKey.C, mode=Mode.MAJOR, confidence=1.0)
        assert result_max.confidence == 1.0

        # Out of bounds
        with pytest.raises(pydantic.ValidationError):
            KeyDetectionResult(key=MusicalKey.C, mode=Mode.MAJOR, confidence=1.5)

        with pytest.raises(pydantic.ValidationError):
            KeyDetectionResult(key=MusicalKey.C, mode=Mode.MAJOR, confidence=-0.1)

    def test_frozen_model(self) -> None:
        result = KeyDetectionResult(key=MusicalKey.C, mode=Mode.MAJOR, confidence=0.9)
        with pytest.raises(pydantic.ValidationError):
            result.confidence = 0.5  # type: ignore[misc]

    def test_all_correlations_storage(self) -> None:
        correlations = {
            "C major": 0.85,
            "A minor": 0.75,
            "G major": 0.65,
        }
        result = KeyDetectionResult(
            key=MusicalKey.C,
            mode=Mode.MAJOR,
            confidence=0.85,
            all_correlations=correlations,
        )
        assert result.all_correlations == correlations


# ---------------------------------------------------------------------------
# Musical Key Enum Tests
# ---------------------------------------------------------------------------


class TestMusicalKey:
    """Test MusicalKey enum values."""

    def test_all_keys_present(self) -> None:
        expected_keys = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
        actual_keys = {k.value for k in MusicalKey}
        assert actual_keys == expected_keys

    def test_key_count(self) -> None:
        assert len(MusicalKey) == 12


class TestMode:
    """Test Mode enum values."""

    def test_mode_values(self) -> None:
        assert Mode.MAJOR.value == "major"
        assert Mode.MINOR.value == "minor"


# ---------------------------------------------------------------------------
# Key Detection Algorithm Tests (with mocked librosa)
# ---------------------------------------------------------------------------


class TestDetectKey:
    """Test the detect_key function with mocked librosa."""

    def test_detect_key_mono_audio(self) -> None:
        """Test key detection with mono audio input."""
        mock_librosa = MagicMock()
        # Create a mock chroma that correlates well with C major profile
        # The C major profile emphasizes C (index 0) and G (index 7)
        mock_chroma = np.zeros((12, 100))
        mock_chroma[0] = 1.0  # Strong C
        mock_chroma[4] = 0.6  # E
        mock_chroma[7] = 0.8  # G
        mock_librosa.feature.chroma_cqt.return_value = mock_chroma

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            from soundlab.analysis.key import detect_key

            samples = np.random.randn(22050).astype(np.float32)  # 1 second at 22050 Hz
            result = detect_key(samples, sr=22050)

        assert isinstance(result, KeyDetectionResult)
        assert isinstance(result.key, MusicalKey)
        assert isinstance(result.mode, Mode)
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_key_stereo_to_mono(self) -> None:
        """Test that stereo audio is converted to mono."""
        mock_librosa = MagicMock()
        mock_chroma = np.ones((12, 50)) * 0.5
        mock_librosa.feature.chroma_cqt.return_value = mock_chroma

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            from soundlab.analysis.key import detect_key

            # Stereo: 2 channels x 22050 samples
            samples = np.random.randn(2, 22050).astype(np.float32)
            result = detect_key(samples, sr=22050)

        # Should have processed successfully
        assert isinstance(result, KeyDetectionResult)
        mock_librosa.feature.chroma_cqt.assert_called_once()

    def test_detect_key_returns_all_correlations(self) -> None:
        """Test that all 24 key correlations are computed."""
        mock_librosa = MagicMock()
        mock_chroma = np.random.rand(12, 100)
        mock_librosa.feature.chroma_cqt.return_value = mock_chroma

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            from soundlab.analysis.key import detect_key

            samples = np.random.randn(22050).astype(np.float32)
            result = detect_key(samples, sr=22050)

        # Should have 12 major + 12 minor = 24 correlations
        assert len(result.all_correlations) == 24

        # All correlation keys should follow the pattern "X major" or "X minor"
        for corr_key in result.all_correlations:
            assert "major" in corr_key or "minor" in corr_key

    def test_detect_key_custom_hop_length(self) -> None:
        """Test key detection with custom hop length."""
        mock_librosa = MagicMock()
        mock_chroma = np.random.rand(12, 100)
        mock_librosa.feature.chroma_cqt.return_value = mock_chroma

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            from soundlab.analysis.key import detect_key

            samples = np.random.randn(22050).astype(np.float32)
            _result = detect_key(samples, sr=22050, hop_length=256)

        # Verify custom hop_length was passed
        call_kwargs = mock_librosa.feature.chroma_cqt.call_args[1]
        assert call_kwargs["hop_length"] == 256


# ---------------------------------------------------------------------------
# Internal Helper Tests
# ---------------------------------------------------------------------------


class TestToMono:
    """Test the _to_mono helper function."""

    def test_mono_passthrough(self) -> None:
        """Mono input should pass through unchanged."""
        from soundlab.analysis.key import _to_mono

        mono = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _to_mono(mono)

        assert result.shape == (3,)
        assert np.array_equal(result, mono)

    def test_stereo_to_mono_channels_first(self) -> None:
        """Stereo input with channels as first dimension."""
        from soundlab.analysis.key import _to_mono

        # Shape: (2, 4) - 2 channels, 4 samples
        stereo = np.array(
            [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        result = _to_mono(stereo)

        expected = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_stereo_to_mono_samples_first(self) -> None:
        """Stereo input with samples as first dimension (shape[0] > shape[-1])."""
        from soundlab.analysis.key import _to_mono

        # Shape: (10, 2) - 10 samples, 2 channels
        stereo = np.ones((10, 2), dtype=np.float32)
        stereo[:, 0] = 2.0  # Channel 0 = 2.0
        stereo[:, 1] = 0.0  # Channel 1 = 0.0
        result = _to_mono(stereo)

        expected = np.ones(10, dtype=np.float32)  # Average = 1.0
        assert np.allclose(result, expected)
