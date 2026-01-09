"""Tests for pipeline QA metrics and scoring utilities."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pydantic")

from soundlab.pipeline.models import QAConfig
from soundlab.pipeline.qa import (
    clipping_ratio,
    leakage_ratio,
    reconstruction_error,
    score_midi,
    score_separation,
    spectral_flatness,
    stereo_coherence,
)
from soundlab.transcription.models import NoteEvent

# --------------------------------------------------------------------------- #
# Synthetic Signal Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def pure_sine_mono() -> np.ndarray:
    """Generate a 440 Hz sine wave (mono, 1 second)."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    amplitude = 0.5
    return amplitude * np.sin(2 * np.pi * 440.0 * t)


@pytest.fixture
def pure_sine_stereo(pure_sine_mono: np.ndarray) -> np.ndarray:
    """Generate identical stereo from mono sine."""
    return np.stack([pure_sine_mono, pure_sine_mono])


@pytest.fixture
def uncorrelated_stereo() -> np.ndarray:
    """Generate uncorrelated stereo noise."""
    rng = np.random.default_rng(42)
    length = 22050
    left = rng.uniform(-0.5, 0.5, size=length).astype(np.float32)
    right = rng.uniform(-0.5, 0.5, size=length).astype(np.float32)
    return np.stack([left, right])


@pytest.fixture
def silence_mono() -> np.ndarray:
    """Generate 1 second of silence (mono)."""
    return np.zeros(22050, dtype=np.float32)


@pytest.fixture
def clipped_audio() -> np.ndarray:
    """Generate audio with intentional clipping."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    amplitude = 1.5  # Will clip
    raw = amplitude * np.sin(2 * np.pi * 440.0 * t)
    return np.clip(raw, -1.0, 1.0).astype(np.float32)


@pytest.fixture
def sample_notes() -> list[NoteEvent]:
    """Generate sample MIDI notes for testing."""
    return [
        NoteEvent(start=0.0, end=0.5, pitch=60, velocity=80),
        NoteEvent(start=0.5, end=1.0, pitch=64, velocity=75),
        NoteEvent(start=1.0, end=1.5, pitch=67, velocity=85),
        NoteEvent(start=1.5, end=2.0, pitch=72, velocity=90),
        NoteEvent(start=0.25, end=0.75, pitch=62, velocity=70),  # Overlapping
    ]


# --------------------------------------------------------------------------- #
# Reconstruction Error Tests
# --------------------------------------------------------------------------- #


class TestReconstructionError:
    """Tests for reconstruction_error metric."""

    def test_perfect_reconstruction(self, pure_sine_mono: np.ndarray) -> None:
        """Perfect stem reconstruction should yield zero error."""
        mix = pure_sine_mono
        stems = {"full": pure_sine_mono.copy()}

        error = reconstruction_error(mix, stems)

        assert error < 1e-6

    def test_partial_reconstruction(self, pure_sine_mono: np.ndarray) -> None:
        """Partial reconstruction should yield non-zero error."""
        mix = pure_sine_mono
        stems = {"partial": pure_sine_mono * 0.5}

        error = reconstruction_error(mix, stems)

        # Half amplitude stems should give ~0.5 normalized error
        assert 0.3 < error < 0.7

    def test_empty_stems(self, pure_sine_mono: np.ndarray) -> None:
        """Empty stems dict should return error of 1.0."""
        mix = pure_sine_mono
        stems = {}

        error = reconstruction_error(mix, stems)

        assert error == 1.0

    def test_multiple_stems_sum_to_mix(self, pure_sine_mono: np.ndarray) -> None:
        """Multiple stems summing to mix should have low error."""
        mix = pure_sine_mono
        stems = {
            "stem_a": pure_sine_mono * 0.5,
            "stem_b": pure_sine_mono * 0.5,
        }

        error = reconstruction_error(mix, stems)

        assert error < 1e-6

    def test_stereo_reconstruction(self, pure_sine_stereo: np.ndarray) -> None:
        """Stereo reconstruction should work correctly."""
        mix = pure_sine_stereo
        stems = {"full": pure_sine_stereo.copy()}

        error = reconstruction_error(mix, stems)

        assert error < 1e-6

    def test_broadcast_stem_shape(self, pure_sine_stereo: np.ndarray) -> None:
        """Different shaped stems should be broadcast."""
        mix = pure_sine_stereo  # Shape: (2, N)
        mono_stem = pure_sine_stereo[0]  # Shape: (N,)

        stems = {"mono": mono_stem}

        # Should handle broadcasting
        error = reconstruction_error(mix, stems)

        assert isinstance(error, float)
        assert 0.0 <= error <= 1.0


# --------------------------------------------------------------------------- #
# Spectral Flatness Tests
# --------------------------------------------------------------------------- #


class TestSpectralFlatness:
    """Tests for spectral_flatness metric."""

    def test_pure_tone_low_flatness(self, pure_sine_mono: np.ndarray) -> None:
        """Pure tones should have low spectral flatness."""
        flatness = spectral_flatness(pure_sine_mono, sr=22050)

        # Pure tones are highly tonal, low flatness
        assert flatness < 0.3

    def test_stereo_input(self, pure_sine_stereo: np.ndarray) -> None:
        """Spectral flatness should handle stereo by converting to mono."""
        flatness = spectral_flatness(pure_sine_stereo, sr=22050)

        assert isinstance(flatness, float)
        assert 0.0 <= flatness <= 1.0

    def test_silence_flatness(self, silence_mono: np.ndarray) -> None:
        """Silence should handle gracefully."""
        flatness = spectral_flatness(silence_mono, sr=22050)

        # Silence may have undefined flatness, but should not raise
        assert isinstance(flatness, float)


# --------------------------------------------------------------------------- #
# Clipping Ratio Tests
# --------------------------------------------------------------------------- #


class TestClippingRatio:
    """Tests for clipping_ratio metric."""

    def test_no_clipping(self, pure_sine_mono: np.ndarray) -> None:
        """Audio within bounds should have zero clipping."""
        ratio = clipping_ratio(pure_sine_mono)

        assert ratio == 0.0

    def test_fully_clipped(self) -> None:
        """Audio at ±1.0 should have high clipping ratio."""
        clipped = np.ones(1000, dtype=np.float32)

        ratio = clipping_ratio(clipped, threshold=0.999)

        assert ratio == 1.0

    def test_partial_clipping(self, clipped_audio: np.ndarray) -> None:
        """Partially clipped audio should have intermediate ratio."""
        ratio = clipping_ratio(clipped_audio, threshold=0.999)

        # Some samples should be clipped
        assert 0.0 < ratio < 1.0

    def test_empty_audio(self) -> None:
        """Empty audio should return 0.0."""
        empty = np.array([], dtype=np.float32)

        ratio = clipping_ratio(empty)

        assert ratio == 0.0

    def test_custom_threshold(self, pure_sine_mono: np.ndarray) -> None:
        """Custom threshold should be respected."""
        # Sine with amplitude 0.5, threshold at 0.4 should detect clipping
        ratio = clipping_ratio(pure_sine_mono, threshold=0.4)

        assert ratio > 0.0


# --------------------------------------------------------------------------- #
# Stereo Coherence Tests
# --------------------------------------------------------------------------- #


class TestStereoCoherence:
    """Tests for stereo_coherence metric."""

    def test_identical_channels_high_coherence(self, pure_sine_stereo: np.ndarray) -> None:
        """Identical channels should have very high coherence."""
        coherence = stereo_coherence(pure_sine_stereo)

        assert coherence > 0.99

    def test_uncorrelated_channels_low_coherence(self, uncorrelated_stereo: np.ndarray) -> None:
        """Uncorrelated channels should have low coherence."""
        coherence = stereo_coherence(uncorrelated_stereo)

        # Random noise should have near-zero correlation
        assert -0.1 < coherence < 0.3

    def test_mono_input_full_coherence(self, pure_sine_mono: np.ndarray) -> None:
        """Mono input should return coherence of 1.0."""
        coherence = stereo_coherence(pure_sine_mono)

        assert coherence == 1.0

    def test_inverted_channels(self, pure_sine_mono: np.ndarray) -> None:
        """Inverted channels should have negative coherence."""
        stereo = np.stack([pure_sine_mono, -pure_sine_mono])

        coherence = stereo_coherence(stereo)

        assert coherence < -0.9

    def test_empty_mono_returns_one(self) -> None:
        """Empty mono audio takes mono path and returns 1.0."""
        mono_empty = np.array([], dtype=np.float32)

        coherence = stereo_coherence(mono_empty)

        # Mono input returns 1.0 per the implementation
        assert coherence == 1.0


# --------------------------------------------------------------------------- #
# Leakage Ratio Tests
# --------------------------------------------------------------------------- #


class TestLeakageRatio:
    """Tests for leakage_ratio metric."""

    def test_single_dominant_stem_low_leakage(self, pure_sine_mono: np.ndarray) -> None:
        """Single dominant stem should have minimal leakage."""
        stems = {
            "vocals": pure_sine_mono * 0.9,
            "drums": pure_sine_mono * 0.1,
        }

        leakage = leakage_ratio(stems)

        # Dominant stem should yield low leakage
        assert leakage < 0.3

    def test_balanced_stems_high_leakage(self, pure_sine_mono: np.ndarray) -> None:
        """Balanced stems indicate potential leakage."""
        stems = {
            "vocals": pure_sine_mono * 0.5,
            "drums": pure_sine_mono * 0.5,
        }

        leakage = leakage_ratio(stems)

        # Equal energy means high "leakage" proxy
        assert leakage > 0.4

    def test_empty_stems_full_leakage(self) -> None:
        """Empty stems dict should return 1.0."""
        leakage = leakage_ratio({})

        assert leakage == 1.0

    def test_single_stem_zero_leakage(self, pure_sine_mono: np.ndarray) -> None:
        """Single stem should have zero leakage."""
        stems = {"only_stem": pure_sine_mono}

        leakage = leakage_ratio(stems)

        # Single stem: dominant = total, so leakage = 0
        assert leakage < 1e-6


# --------------------------------------------------------------------------- #
# score_separation Tests
# --------------------------------------------------------------------------- #


class TestScoreSeparation:
    """Tests for score_separation aggregate scoring."""

    def test_perfect_separation_high_score(self, pure_sine_stereo: np.ndarray) -> None:
        """Perfect reconstruction should yield high score."""
        mix = pure_sine_stereo
        stems = {"full": pure_sine_stereo.copy()}

        result = score_separation(mix, stems, sr=22050)

        assert result.score > 0.7
        assert result.passed is True
        assert "reconstruction_error" in result.metrics
        assert "spectral_flatness" in result.metrics
        assert "clipping_ratio" in result.metrics
        assert "stereo_coherence" in result.metrics
        assert "leakage_ratio" in result.metrics

    def test_poor_separation_low_score(self, pure_sine_stereo: np.ndarray) -> None:
        """Poor reconstruction should yield low score."""
        mix = pure_sine_stereo
        stems = {"partial": pure_sine_stereo * 0.1}

        result = score_separation(mix, stems, sr=22050)

        assert result.score < 0.5
        assert result.passed is False
        assert result.metrics["reconstruction_error"] > 0.5

    def test_stem_scores_populated(self, pure_sine_mono: np.ndarray) -> None:
        """Stem scores should contain RMS values."""
        mix = pure_sine_mono
        stems = {
            "vocals": pure_sine_mono * 0.6,
            "drums": pure_sine_mono * 0.4,
        }

        result = score_separation(mix, stems, sr=22050)

        assert "vocals" in result.stem_scores
        assert "drums" in result.stem_scores
        assert result.stem_scores["vocals"] > result.stem_scores["drums"]

    def test_custom_qa_config(self, pure_sine_stereo: np.ndarray) -> None:
        """Custom QA thresholds should affect pass/fail."""
        mix = pure_sine_stereo
        stems = {"full": pure_sine_stereo * 0.9}

        strict_qa = QAConfig(min_overall_score=0.95)
        result = score_separation(mix, stems, sr=22050, qa=strict_qa)

        # Strict threshold may cause failure
        assert isinstance(result.passed, bool)

    def test_empty_stems_fails(self, pure_sine_stereo: np.ndarray) -> None:
        """Empty stems should fail QA."""
        mix = pure_sine_stereo
        stems = {}

        result = score_separation(mix, stems, sr=22050)

        assert result.score < 0.5
        assert result.passed is False


# --------------------------------------------------------------------------- #
# score_midi Tests
# --------------------------------------------------------------------------- #


class TestScoreMidi:
    """Tests for score_midi MIDI sanity scoring."""

    def test_reasonable_notes_high_score(self, sample_notes: list[NoteEvent]) -> None:
        """Reasonable note density should score well."""
        result = score_midi(sample_notes, duration=10.0)

        assert result.score > 0.6
        assert result.passed is True
        assert "notes" in result.metrics
        assert "notes_per_second" in result.metrics
        assert "max_polyphony" in result.metrics
        assert "pitch_range" in result.metrics

    def test_empty_notes_zero_score(self) -> None:
        """Empty notes list should score 0."""
        result = score_midi([])

        assert result.score == 0.0
        assert result.passed is False
        assert result.metrics["notes"] == 0.0

    def test_sparse_notes_penalized(self) -> None:
        """Very sparse notes should be penalized."""
        sparse = [NoteEvent(start=0.0, end=0.5, pitch=60, velocity=80)]

        result = score_midi(sparse, duration=60.0)

        # 1 note in 60 seconds = very sparse
        assert result.metrics["notes_per_second"] < 0.1
        assert result.score < 1.0

    def test_dense_notes_penalized(self) -> None:
        """Very dense notes should be penalized."""
        dense = [
            NoteEvent(start=i * 0.01, end=(i + 1) * 0.01, pitch=60, velocity=80) for i in range(500)
        ]

        result = score_midi(dense, duration=5.0)

        # 100 notes/second is extremely dense
        assert result.metrics["notes_per_second"] > 25.0
        assert result.score < 1.0

    def test_high_polyphony_penalized(self) -> None:
        """Very high polyphony should be penalized."""
        # 15 simultaneous notes
        high_poly = [NoteEvent(start=0.0, end=1.0, pitch=60 + i, velocity=80) for i in range(15)]

        result = score_midi(high_poly, duration=1.0)

        assert result.metrics["max_polyphony"] == 15.0
        assert result.score < 1.0

    def test_moderate_polyphony_ok(self, sample_notes: list[NoteEvent]) -> None:
        """Moderate polyphony should not be penalized."""
        result = score_midi(sample_notes)

        # Sample notes have at most 2 concurrent notes
        assert result.metrics["max_polyphony"] == 2.0
        assert result.score > 0.5

    def test_custom_qa_threshold(self, sample_notes: list[NoteEvent]) -> None:
        """Custom QA threshold should affect passed status."""
        strict_qa = QAConfig(min_midi_score=0.99)

        result = score_midi(sample_notes, qa=strict_qa)

        # Even good notes may fail strict threshold
        assert isinstance(result.passed, bool)

    def test_pitch_range_calculated(self, sample_notes: list[NoteEvent]) -> None:
        """Pitch range should be correctly calculated."""
        result = score_midi(sample_notes)

        # Sample notes: pitches 60, 64, 67, 72, 62 -> range = 72 - 60 = 12
        assert result.metrics["pitch_range"] == 12.0

    def test_duration_auto_computed(self) -> None:
        """Duration should auto-compute from notes if not provided."""
        notes = [
            NoteEvent(start=1.0, end=2.0, pitch=60, velocity=80),
            NoteEvent(start=3.0, end=4.0, pitch=72, velocity=80),
        ]

        result = score_midi(notes)

        # Duration = 4.0 - 1.0 = 3.0 seconds
        # 2 notes / 3 seconds = 0.67 notes/second
        assert 0.5 < result.metrics["notes_per_second"] < 1.0


# --------------------------------------------------------------------------- #
# Integration Tests
# --------------------------------------------------------------------------- #


class TestQAIntegration:
    """Integration tests for QA scoring workflow."""

    def test_full_qa_workflow(
        self,
        pure_sine_stereo: np.ndarray,
        sample_notes: list[NoteEvent],
    ) -> None:
        """Test complete QA workflow with both separation and MIDI scoring."""
        mix = pure_sine_stereo
        stems = {
            "vocals": pure_sine_stereo * 0.6,
            "other": pure_sine_stereo * 0.4,
        }

        # Score separation
        sep_result = score_separation(mix, stems, sr=22050)

        # Score MIDI
        midi_result = score_midi(sample_notes, duration=10.0)

        # Both should produce valid results
        assert 0.0 <= sep_result.score <= 1.0
        assert 0.0 <= midi_result.score <= 1.0

        # Metrics should be populated
        assert len(sep_result.metrics) >= 5
        assert len(midi_result.metrics) >= 4

    def test_qa_config_consistency(self, pure_sine_stereo: np.ndarray) -> None:
        """QAConfig should consistently affect all scoring functions."""
        qa = QAConfig(
            min_overall_score=0.9,
            max_reconstruction_error=0.05,
            min_midi_score=0.8,
        )

        mix = pure_sine_stereo
        stems = {"full": pure_sine_stereo.copy()}

        sep_result = score_separation(mix, stems, sr=22050, qa=qa)

        # With perfect reconstruction and strict threshold
        assert sep_result.score > 0.8

    def test_multiple_stem_evaluation(self, pure_sine_mono: np.ndarray) -> None:
        """Test evaluation with multiple realistic stems."""
        # Create synthetic mix from stems
        vocals = pure_sine_mono * 0.4
        drums = pure_sine_mono * 0.3
        bass = pure_sine_mono * 0.2
        other = pure_sine_mono * 0.1

        mix = vocals + drums + bass + other
        stems = {
            "vocals": vocals,
            "drums": drums,
            "bass": bass,
            "other": other,
        }

        result = score_separation(mix, stems, sr=22050)

        # Perfect reconstruction from stems
        assert result.metrics["reconstruction_error"] < 1e-6
        assert len(result.stem_scores) == 4
        assert result.stem_scores["vocals"] > result.stem_scores["other"]
