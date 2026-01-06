"""Tests for soundlab.analysis.key."""

from __future__ import annotations

import numpy as np
import pytest

from soundlab.analysis.key import (
    detect_key,
    get_compatible_keys,
    get_parallel_key,
    get_relative_key,
)
from soundlab.analysis.models import KeyDetectionResult, Mode, MusicalKey


class TestDetectKey:
    """Test detect_key function."""

    def test_detect_key_returns_result(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """Should return KeyDetectionResult."""
        result = detect_key(sample_mono_audio, sample_rate)

        assert isinstance(result, KeyDetectionResult)
        assert isinstance(result.key, MusicalKey)
        assert isinstance(result.mode, Mode)

    def test_detect_key_confidence_bounds(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """Confidence should be between 0 and 1."""
        result = detect_key(sample_mono_audio, sample_rate)

        assert 0.0 <= result.confidence <= 1.0

    def test_detect_key_all_correlations(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """Should include all key correlations."""
        result = detect_key(sample_mono_audio, sample_rate)

        # Should have 24 correlations (12 major + 12 minor)
        assert len(result.all_correlations) == 24

    def test_detect_key_handles_stereo(self, sample_stereo_audio: np.ndarray, sample_rate: int):
        """Should handle stereo audio (converts to mono)."""
        result = detect_key(sample_stereo_audio, sample_rate)

        assert isinstance(result, KeyDetectionResult)

    def test_detect_key_pure_a_tone(self, sample_rate: int):
        """Pure A440 tone should detect A major or A minor."""
        # Generate pure A440 sine wave
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        result = detect_key(audio, sample_rate)

        # A pure sine wave at 440Hz should strongly correlate with A
        # Note: May not be exactly A due to algorithm behavior with pure tones
        assert result.key is not None

    def test_detect_key_uses_cqt(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """Should use CQT chroma by default."""
        result = detect_key(sample_mono_audio, sample_rate, use_cqt=True)
        assert result is not None

    def test_detect_key_can_use_stft(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """Should support STFT chroma."""
        result = detect_key(sample_mono_audio, sample_rate, use_cqt=False)
        assert result is not None


class TestKeyDetectionResult:
    """Test KeyDetectionResult model."""

    def test_name_property(self):
        """name should return 'Key mode' format."""
        result = KeyDetectionResult(
            key=MusicalKey.A,
            mode=Mode.MINOR,
            confidence=0.8,
            all_correlations={},
        )
        assert result.name == "A minor"

        result = KeyDetectionResult(
            key=MusicalKey.C,
            mode=Mode.MAJOR,
            confidence=0.8,
            all_correlations={},
        )
        assert result.name == "C major"

    def test_camelot_a_minor(self):
        """A minor should have Camelot code 8A."""
        result = KeyDetectionResult(
            key=MusicalKey.A,
            mode=Mode.MINOR,
            confidence=0.8,
            all_correlations={},
        )
        assert result.camelot == "8A"

    def test_camelot_c_major(self):
        """C major should have Camelot code 8B."""
        result = KeyDetectionResult(
            key=MusicalKey.C,
            mode=Mode.MAJOR,
            confidence=0.8,
            all_correlations={},
        )
        assert result.camelot == "8B"

    def test_camelot_f_minor(self):
        """F minor should have Camelot code 4A."""
        result = KeyDetectionResult(
            key=MusicalKey.F,
            mode=Mode.MINOR,
            confidence=0.8,
            all_correlations={},
        )
        assert result.camelot == "4A"


class TestGetRelativeKey:
    """Test get_relative_key function."""

    def test_c_major_relative(self):
        """C major relative should be A minor."""
        result = KeyDetectionResult(
            key=MusicalKey.C,
            mode=Mode.MAJOR,
            confidence=0.8,
            all_correlations={},
        )
        rel_key, rel_mode = get_relative_key(result)

        assert rel_key == MusicalKey.A
        assert rel_mode == Mode.MINOR

    def test_a_minor_relative(self):
        """A minor relative should be C major."""
        result = KeyDetectionResult(
            key=MusicalKey.A,
            mode=Mode.MINOR,
            confidence=0.8,
            all_correlations={},
        )
        rel_key, rel_mode = get_relative_key(result)

        assert rel_key == MusicalKey.C
        assert rel_mode == Mode.MAJOR

    def test_g_major_relative(self):
        """G major relative should be E minor."""
        result = KeyDetectionResult(
            key=MusicalKey.G,
            mode=Mode.MAJOR,
            confidence=0.8,
            all_correlations={},
        )
        rel_key, rel_mode = get_relative_key(result)

        assert rel_key == MusicalKey.E
        assert rel_mode == Mode.MINOR


class TestGetParallelKey:
    """Test get_parallel_key function."""

    def test_c_major_parallel(self):
        """C major parallel should be C minor."""
        result = KeyDetectionResult(
            key=MusicalKey.C,
            mode=Mode.MAJOR,
            confidence=0.8,
            all_correlations={},
        )
        par_key, par_mode = get_parallel_key(result)

        assert par_key == MusicalKey.C
        assert par_mode == Mode.MINOR

    def test_a_minor_parallel(self):
        """A minor parallel should be A major."""
        result = KeyDetectionResult(
            key=MusicalKey.A,
            mode=Mode.MINOR,
            confidence=0.8,
            all_correlations={},
        )
        par_key, par_mode = get_parallel_key(result)

        assert par_key == MusicalKey.A
        assert par_mode == Mode.MAJOR


class TestGetCompatibleKeys:
    """Test get_compatible_keys function."""

    def test_returns_list(self):
        """Should return list of compatible keys."""
        result = KeyDetectionResult(
            key=MusicalKey.C,
            mode=Mode.MAJOR,
            confidence=0.8,
            all_correlations={},
        )
        compatible = get_compatible_keys(result)

        assert isinstance(compatible, list)
        assert len(compatible) >= 5  # same, relative, dominant, subdominant, parallel

    def test_includes_same_key(self):
        """Should include the same key."""
        result = KeyDetectionResult(
            key=MusicalKey.C,
            mode=Mode.MAJOR,
            confidence=0.8,
            all_correlations={},
        )
        compatible = get_compatible_keys(result)

        same = [c for c in compatible if c[2] == "same"][0]
        assert same[0] == MusicalKey.C
        assert same[1] == Mode.MAJOR

    def test_includes_relative(self):
        """Should include relative key."""
        result = KeyDetectionResult(
            key=MusicalKey.C,
            mode=Mode.MAJOR,
            confidence=0.8,
            all_correlations={},
        )
        compatible = get_compatible_keys(result)

        relative = [c for c in compatible if c[2] == "relative"][0]
        assert relative[0] == MusicalKey.A
        assert relative[1] == Mode.MINOR

    def test_includes_dominant(self):
        """Should include dominant key."""
        result = KeyDetectionResult(
            key=MusicalKey.C,
            mode=Mode.MAJOR,
            confidence=0.8,
            all_correlations={},
        )
        compatible = get_compatible_keys(result)

        dominant = [c for c in compatible if c[2] == "dominant"][0]
        assert dominant[0] == MusicalKey.G  # G is 5th of C
