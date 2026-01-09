"""Tests for soundlab.separation.utils module."""

from __future__ import annotations

import numpy as np
import pytest

from soundlab.separation.utils import calculate_segments, overlap_add


class TestCalculateSegments:
    """Tests for calculate_segments function."""

    def test_short_duration_single_segment(self) -> None:
        """Short audio produces single segment."""
        segments = calculate_segments(duration=5.0, segment_length=10.0, overlap=0.25)
        assert segments == [(0.0, 5.0)]

    def test_long_duration_multiple_segments(self) -> None:
        """Long audio produces multiple overlapping segments."""
        segments = calculate_segments(duration=30.0, segment_length=10.0, overlap=0.25)

        assert len(segments) >= 3
        assert segments[0] == (0.0, 10.0)
        # Check overlap: step = 10.0 * (1 - 0.25) = 7.5
        assert segments[1][0] == pytest.approx(7.5)

    def test_zero_duration_empty_list(self) -> None:
        """Zero duration returns empty list."""
        segments = calculate_segments(duration=0.0, segment_length=10.0, overlap=0.25)
        assert segments == []

    def test_negative_duration_empty_list(self) -> None:
        """Negative duration returns empty list."""
        segments = calculate_segments(duration=-5.0, segment_length=10.0, overlap=0.25)
        assert segments == []

    def test_zero_segment_length_empty_list(self) -> None:
        """Zero segment length returns empty list."""
        segments = calculate_segments(duration=30.0, segment_length=0.0, overlap=0.25)
        assert segments == []

    def test_negative_segment_length_empty_list(self) -> None:
        """Negative segment length returns empty list."""
        segments = calculate_segments(duration=30.0, segment_length=-10.0, overlap=0.25)
        assert segments == []

    def test_no_overlap(self) -> None:
        """Zero overlap produces non-overlapping segments."""
        segments = calculate_segments(duration=30.0, segment_length=10.0, overlap=0.0)

        assert len(segments) == 3
        assert segments[0] == (0.0, 10.0)
        assert segments[1] == (10.0, 20.0)
        assert segments[2] == (20.0, 30.0)

    def test_high_overlap(self) -> None:
        """High overlap produces many segments."""
        segments = calculate_segments(duration=20.0, segment_length=10.0, overlap=0.9)

        # step = 10.0 * (1 - 0.9) = 1.0, should have ~11 segments
        assert len(segments) > 10

    def test_full_overlap_uses_segment_length_as_step(self) -> None:
        """Full overlap (1.0) falls back to segment_length step."""
        segments = calculate_segments(duration=30.0, segment_length=10.0, overlap=1.0)

        # step = 0, fallback to segment_length
        assert len(segments) == 3

    def test_exact_duration_match(self) -> None:
        """Duration exactly matches segment produces single segment."""
        segments = calculate_segments(duration=10.0, segment_length=10.0, overlap=0.5)
        assert segments == [(0.0, 10.0)]

    def test_segments_cover_full_duration(self) -> None:
        """All segments together cover the full duration."""
        segments = calculate_segments(duration=25.0, segment_length=10.0, overlap=0.25)

        # Last segment should end at or after duration
        assert segments[-1][1] >= 25.0 or segments[-1][1] == 25.0


class TestOverlapAdd:
    """Tests for overlap_add function."""

    def test_single_segment_returns_unchanged(self) -> None:
        """Single segment is returned unchanged."""
        segment = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = overlap_add([segment], overlap=0.25)

        np.testing.assert_array_almost_equal(result, segment)

    def test_empty_segments_returns_empty(self) -> None:
        """Empty segment list returns empty array."""
        result = overlap_add([], overlap=0.25)
        assert len(result) == 0

    def test_non_overlapping_concatenation(self) -> None:
        """Zero overlap concatenates segments."""
        seg1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        seg2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

        result = overlap_add([seg1, seg2], overlap=0.0)

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_overlap_averaging(self) -> None:
        """Overlapping regions are averaged."""
        seg1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        seg2 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)

        # 50% overlap means step = 2, so samples 2-3 overlap
        result = overlap_add([seg1, seg2], overlap=0.5)

        # First 2 samples: 1.0, next 2: average of (1+2)/2 = 1.5, last 2: 2.0
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(1.5)
        assert result[3] == pytest.approx(1.5)
        assert result[4] == pytest.approx(2.0)
        assert result[5] == pytest.approx(2.0)

    def test_stereo_segments(self) -> None:
        """Stereo (2D) segments are handled correctly."""
        # Shape: (channels, samples)
        seg1 = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
        seg2 = np.array([[3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]], dtype=np.float32)

        result = overlap_add([seg1, seg2], overlap=0.0)

        assert result.shape == (2, 8)
        # Channel 0
        np.testing.assert_array_almost_equal(result[0, :4], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result[0, 4:], [3.0, 3.0, 3.0, 3.0])
        # Channel 1
        np.testing.assert_array_almost_equal(result[1, :4], [2.0, 2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(result[1, 4:], [4.0, 4.0, 4.0, 4.0])

    def test_three_segments_overlap(self) -> None:
        """Three segments with overlap are combined correctly."""
        seg1 = np.ones(4, dtype=np.float32)
        seg2 = np.ones(4, dtype=np.float32) * 2
        seg3 = np.ones(4, dtype=np.float32) * 3

        result = overlap_add([seg1, seg2, seg3], overlap=0.5)

        # Should produce smooth transitions
        assert result.shape[0] > 4
        # Values should transition from 1 -> 2 -> 3
        assert result[0] == pytest.approx(1.0)
        assert result[-1] == pytest.approx(3.0)

    def test_generator_input(self) -> None:
        """Generator input works correctly."""

        def segment_gen() -> list[np.ndarray]:
            yield np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            yield np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

        result = overlap_add(segment_gen(), overlap=0.0)
        assert len(result) == 8


class TestIntegration:
    """Integration tests for segmentation + overlap-add round-trip."""

    def test_segmentation_overlap_add_roundtrip(self) -> None:
        """Segmenting and recombining should approximate original."""
        # Create test signal
        original = np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float32)
        duration = len(original) / 100.0  # 1 second at 100 Hz

        # Segment it
        segments_info = calculate_segments(duration, segment_length=0.4, overlap=0.25)

        # Extract segments
        segments = []
        for start, end in segments_info:
            start_idx = int(start * 100)
            end_idx = int(end * 100)
            segments.append(original[start_idx:end_idx])

        # Recombine
        result = overlap_add(segments, overlap=0.25)

        # Result should be close to original length
        assert abs(len(result) - len(original)) <= 10
