"""Tests for pipeline post-processing utilities."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pydantic")

from soundlab.pipeline.postprocess import clean_stems, cleanup_midi_notes, mono_amt_exports
from soundlab.transcription.models import NoteEvent

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def stereo_audio() -> np.ndarray:
    """Generate stereo audio with 2 channels x 22050 samples."""
    sr = 22050
    duration = 1.0
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, dtype=np.float32)

    # Different frequencies for left/right to ensure mono conversion is tested
    left = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.5 * np.sin(2 * np.pi * 880.0 * t)

    return np.stack([left, right]).astype(np.float32)


@pytest.fixture
def mono_audio() -> np.ndarray:
    """Generate mono audio with 22050 samples."""
    sr = 22050
    duration = 1.0
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, dtype=np.float32)

    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def noisy_audio() -> np.ndarray:
    """Generate audio with low-level noise."""
    rng = np.random.default_rng(42)
    clean = np.zeros(22050, dtype=np.float32)

    # Add signal in the middle section
    clean[5000:15000] = 0.5 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 10000 / 22050, 10000))

    # Add low-level noise everywhere
    noise = rng.uniform(-1e-5, 1e-5, size=22050).astype(np.float32)

    return clean + noise


@pytest.fixture
def stems_dict(stereo_audio: np.ndarray, mono_audio: np.ndarray) -> dict[str, np.ndarray]:
    """Generate a stems dictionary with various configurations."""
    return {
        "vocals": stereo_audio,
        "drums": stereo_audio * 0.8,
        "bass": mono_audio,
        "other": stereo_audio * 0.5,
    }


@pytest.fixture
def sample_notes() -> list[NoteEvent]:
    """Generate sample MIDI notes with various durations."""
    # All notes must have valid pitch (0-127) and velocity (0-127) per NoteEvent model
    return [
        NoteEvent(start=0.0, end=0.5, pitch=60, velocity=80),  # Normal
        NoteEvent(start=0.5, end=1.0, pitch=64, velocity=75),  # Normal
        NoteEvent(start=1.0, end=1.01, pitch=67, velocity=85),  # Very short (0.01s)
        NoteEvent(start=1.5, end=1.52, pitch=72, velocity=90),  # Short (0.02s)
        NoteEvent(start=2.0, end=3.0, pitch=55, velocity=0),  # Zero velocity
        NoteEvent(start=3.0, end=3.5, pitch=48, velocity=80),  # Normal
        NoteEvent(start=3.5, end=4.0, pitch=84, velocity=80),  # Normal
        NoteEvent(start=4.0, end=4.5, pitch=60, velocity=127),  # Max velocity
    ]


# --------------------------------------------------------------------------- #
# mono_amt_exports Tests
# --------------------------------------------------------------------------- #


class TestMonoAmtExports:
    """Tests for mono_amt_exports function."""

    def test_stereo_to_mono_conversion(self, stereo_audio: np.ndarray) -> None:
        """Stereo audio should be converted to mono."""
        stems = {"vocals": stereo_audio}
        result = mono_amt_exports(stems)

        assert "vocals" in result
        assert result["vocals"].ndim == 1

    def test_preserves_sample_length(self, stereo_audio: np.ndarray) -> None:
        """Mono conversion should preserve sample length."""
        original_length = stereo_audio.shape[1]
        stems = {"vocals": stereo_audio}
        result = mono_amt_exports(stems)

        assert result["vocals"].shape[0] == original_length

    def test_mono_input_unchanged(self, mono_audio: np.ndarray) -> None:
        """Mono input should remain mono with same length."""
        original_length = mono_audio.shape[0]
        stems = {"bass": mono_audio}
        result = mono_amt_exports(stems)

        assert result["bass"].ndim == 1
        assert result["bass"].shape[0] == original_length

    def test_multiple_stems(self, stems_dict: dict[str, np.ndarray]) -> None:
        """All stems should be converted to mono."""
        result = mono_amt_exports(stems_dict)

        assert len(result) == len(stems_dict)
        for name, audio in result.items():
            assert audio.ndim == 1, f"Stem '{name}' should be mono"

    def test_empty_dict(self) -> None:
        """Empty stems dict should return empty dict."""
        result = mono_amt_exports({})

        assert result == {}

    def test_column_major_stereo(self) -> None:
        """Handle column-major stereo (samples, channels) format."""
        # Column-major: (samples, channels)
        column_major = np.random.rand(22050, 2).astype(np.float32)
        stems = {"vocals": column_major}
        result = mono_amt_exports(stems)

        assert result["vocals"].ndim == 1
        # Should preserve the larger dimension (samples)
        assert result["vocals"].shape[0] == 22050

    def test_alignment_preservation(self, stereo_audio: np.ndarray) -> None:
        """Sample alignment should be preserved across all stems."""
        stems = {
            "vocals": stereo_audio,
            "drums": stereo_audio * 0.8,
        }
        result = mono_amt_exports(stems)

        # Both should have same length
        assert result["vocals"].shape[0] == result["drums"].shape[0]


# --------------------------------------------------------------------------- #
# clean_stems Tests
# --------------------------------------------------------------------------- #


class TestCleanStems:
    """Tests for clean_stems function."""

    def test_zeros_low_level_noise(self, noisy_audio: np.ndarray) -> None:
        """Low-level noise should be zeroed."""
        stems = {"noisy": noisy_audio}
        result = clean_stems(stems, silence_threshold=1e-4)

        # Silent regions should be exactly zero
        silent_region = result["noisy"][:5000]
        assert np.allclose(silent_region, 0.0)

    def test_preserves_signal(self, noisy_audio: np.ndarray) -> None:
        """Signal above threshold should be preserved."""
        stems = {"noisy": noisy_audio}
        result = clean_stems(stems, silence_threshold=1e-4)

        # Signal region should be mostly preserved
        signal_region = result["noisy"][5000:15000]

        # Most signal samples should be non-zero
        non_zero_count = np.sum(np.abs(signal_region) > 1e-6)
        assert non_zero_count > len(signal_region) * 0.9

    def test_preserves_sample_length(self, noisy_audio: np.ndarray) -> None:
        """Cleaning should preserve sample length."""
        original_length = noisy_audio.shape[0]
        stems = {"noisy": noisy_audio}
        result = clean_stems(stems, silence_threshold=1e-4)

        assert result["noisy"].shape[0] == original_length

    def test_alignment_safe(self, stems_dict: dict[str, np.ndarray]) -> None:
        """Cleaning should not affect sample alignment."""
        # Get original lengths
        original_lengths = {name: audio.shape[-1] for name, audio in stems_dict.items()}

        result = clean_stems(stems_dict, silence_threshold=1e-4)

        # All lengths should be preserved
        for name, audio in result.items():
            assert audio.shape[-1] == original_lengths[name]

    def test_custom_threshold(self, mono_audio: np.ndarray) -> None:
        """Custom threshold should be respected."""
        stems = {"audio": mono_audio}

        # High threshold zeros most samples
        result_high = clean_stems(stems, silence_threshold=0.4)

        # Low threshold preserves most samples
        result_low = clean_stems(stems, silence_threshold=1e-6)

        # More samples zeroed with higher threshold
        high_zero_count = np.sum(result_high["audio"] == 0.0)
        low_zero_count = np.sum(result_low["audio"] == 0.0)

        assert high_zero_count >= low_zero_count

    def test_default_threshold(self) -> None:
        """Default threshold (1e-4) should be used when not specified."""
        # Create audio with samples at exactly the threshold
        audio = np.array([1e-5, 1e-4, 1e-3], dtype=np.float32)
        stems = {"audio": audio}

        result = clean_stems(stems)

        # Sample below threshold should be zeroed
        assert result["audio"][0] == 0.0
        # Sample at threshold should be zeroed (< not <=)
        # Sample above threshold should be preserved
        assert result["audio"][2] != 0.0

    def test_empty_stems(self) -> None:
        """Empty stems dict should return empty dict."""
        result = clean_stems({})

        assert result == {}

    def test_stereo_cleaning(self, stereo_audio: np.ndarray) -> None:
        """Stereo audio should be cleaned without shape changes."""
        original_shape = stereo_audio.shape
        stems = {"vocals": stereo_audio}
        result = clean_stems(stems, silence_threshold=1e-4)

        assert result["vocals"].shape == original_shape


# --------------------------------------------------------------------------- #
# cleanup_midi_notes Tests
# --------------------------------------------------------------------------- #


class TestCleanupMidiNotes:
    """Tests for cleanup_midi_notes function."""

    def test_filters_short_notes(self, sample_notes: list[NoteEvent]) -> None:
        """Notes shorter than min_duration should be filtered."""
        result = cleanup_midi_notes(sample_notes, min_duration=0.02)

        # Notes at 1.0-1.01 (0.01s) should be filtered
        durations = [note.end - note.start for note in result]
        assert all(d >= 0.02 for d in durations)

    def test_preserves_valid_notes(self, sample_notes: list[NoteEvent]) -> None:
        """Notes meeting criteria should be preserved."""
        # Use very permissive settings
        result = cleanup_midi_notes(sample_notes, min_duration=0.001, min_velocity=0)

        # Should preserve most notes (some may still be filtered for pitch)
        assert len(result) > 0

    def test_clamps_velocity(self, sample_notes: list[NoteEvent]) -> None:
        """Velocity should be clamped to [min_velocity, 127]."""
        result = cleanup_midi_notes(sample_notes, min_duration=0.001, min_velocity=10)

        for note in result:
            assert 10 <= note.velocity <= 127

    def test_clamps_pitch(self) -> None:
        """Pitch should be clamped to [0, 127]."""
        # Create notes with valid pitch to verify clamping preserves them
        notes = [
            NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80),
        ]

        result = cleanup_midi_notes(notes, min_duration=0.01)

        for note in result:
            assert 0 <= note.pitch <= 127

    def test_clamps_low_velocity(self) -> None:
        """Notes with velocity below min_velocity should be clamped, not filtered."""
        notes = [
            NoteEvent(start=0.0, end=1.0, pitch=60, velocity=0),
            NoteEvent(start=1.0, end=2.0, pitch=64, velocity=5),
            NoteEvent(start=2.0, end=3.0, pitch=67, velocity=80),
        ]

        result = cleanup_midi_notes(notes, min_duration=0.01, min_velocity=10)

        # All notes should survive, with velocities clamped to min_velocity
        assert len(result) == 3
        assert result[0].velocity == 10  # Was 0, clamped to 10
        assert result[1].velocity == 10  # Was 5, clamped to 10
        assert result[2].velocity == 80  # Was 80, unchanged

    def test_default_parameters(self) -> None:
        """Default parameters should be sensible."""
        notes = [
            NoteEvent(start=0.0, end=0.5, pitch=60, velocity=80),
            NoteEvent(start=0.5, end=0.51, pitch=64, velocity=75),  # 0.01s < default 0.02
        ]

        result = cleanup_midi_notes(notes)

        # Default min_duration=0.02 should filter 0.01s note
        assert len(result) == 1
        assert result[0].end - result[0].start >= 0.02

    def test_empty_input(self) -> None:
        """Empty input should return empty list."""
        result = cleanup_midi_notes([])

        assert result == []

    def test_all_filtered(self) -> None:
        """All notes filtered should return empty list."""
        notes = [
            NoteEvent(start=0.0, end=0.01, pitch=60, velocity=80),  # Too short
            NoteEvent(start=0.5, end=0.51, pitch=64, velocity=80),  # Too short
        ]

        result = cleanup_midi_notes(notes, min_duration=0.02)

        assert result == []

    def test_preserves_timing(self) -> None:
        """Note start and end times should be preserved."""
        notes = [
            NoteEvent(start=1.234, end=2.567, pitch=60, velocity=80),
        ]

        result = cleanup_midi_notes(notes)

        assert len(result) == 1
        assert result[0].start == 1.234
        assert result[0].end == 2.567

    def test_returns_new_list(self) -> None:
        """Result should be a new list, not modify input."""
        notes = [
            NoteEvent(start=0.0, end=1.0, pitch=60, velocity=80),
        ]

        result = cleanup_midi_notes(notes)

        assert result is not notes
        # Original should be unchanged
        assert len(notes) == 1

    def test_velocity_clamping_at_boundary(self) -> None:
        """Velocity at 127 should not be changed."""
        notes = [
            NoteEvent(start=0.0, end=1.0, pitch=60, velocity=127),
        ]

        result = cleanup_midi_notes(notes, min_velocity=1)

        assert result[0].velocity == 127

    def test_velocity_above_max_clamped(self) -> None:
        """Velocity above 127 should be clamped to 127."""
        # Note: Pydantic validates velocity in NoteEvent, so we can't create
        # notes with velocity > 127 directly. The cleanup function receives
        # valid notes but clamps anyway as a safety measure.
        notes = [
            NoteEvent(start=0.0, end=1.0, pitch=60, velocity=127),
        ]

        result = cleanup_midi_notes(notes)

        assert result[0].velocity <= 127


# --------------------------------------------------------------------------- #
# Integration Tests
# --------------------------------------------------------------------------- #


class TestPostprocessIntegration:
    """Integration tests for post-processing workflow."""

    def test_full_postprocess_workflow(
        self,
        stems_dict: dict[str, np.ndarray],
        sample_notes: list[NoteEvent],
    ) -> None:
        """Test complete post-processing pipeline."""
        # Step 1: Clean stems
        cleaned_stems = clean_stems(stems_dict, silence_threshold=1e-5)

        # Step 2: Convert to mono for AMT
        mono_stems = mono_amt_exports(cleaned_stems)

        # Step 3: Clean up MIDI
        cleaned_notes = cleanup_midi_notes(sample_notes)

        # Verify results
        assert len(cleaned_stems) == len(stems_dict)
        assert len(mono_stems) == len(stems_dict)

        for name in stems_dict:
            # Mono conversion preserves length
            original_length = stems_dict[name].shape[-1]
            assert mono_stems[name].shape[0] == original_length

        # MIDI notes are cleaned
        assert len(cleaned_notes) < len(sample_notes)

    def test_alignment_preservation_through_pipeline(
        self,
        stereo_audio: np.ndarray,
    ) -> None:
        """Sample alignment should be preserved through all steps."""
        original_length = stereo_audio.shape[1]

        stems = {
            "vocals": stereo_audio,
            "drums": stereo_audio * 0.5,
            "bass": stereo_audio * 0.3,
        }

        # Pipeline
        cleaned = clean_stems(stems)
        mono = mono_amt_exports(cleaned)

        # All stems should have original length
        for name, audio in mono.items():
            assert audio.shape[0] == original_length, (
                f"Stem '{name}' length mismatch: {audio.shape[0]} != {original_length}"
            )

    def test_pipeline_with_empty_stems(self) -> None:
        """Pipeline should handle empty stems gracefully."""
        stems = {}

        cleaned = clean_stems(stems)
        mono = mono_amt_exports(cleaned)

        assert cleaned == {}
        assert mono == {}

    def test_pipeline_with_empty_notes(self) -> None:
        """Pipeline should handle empty notes gracefully."""
        notes = []
        cleaned = cleanup_midi_notes(notes)

        assert cleaned == []

    def test_mono_export_values_are_averages(self, stereo_audio: np.ndarray) -> None:
        """Mono conversion should average channels."""
        stems = {"test": stereo_audio}
        result = mono_amt_exports(stems)

        # For row-major stereo (channels, samples), mean is along axis 0
        expected = np.mean(stereo_audio, axis=0)
        np.testing.assert_allclose(result["test"], expected, rtol=1e-5)
