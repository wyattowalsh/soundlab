"""Utility helpers for stem separation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from soundlab.core.types import AudioArray


def calculate_segments(
    duration: float,
    segment_length: float,
    overlap: float,
) -> list[tuple[float, float]]:
    """Calculate time segments covering the duration."""
    if duration <= 0 or segment_length <= 0:
        return []

    step = segment_length * (1.0 - overlap)
    if step <= 0:
        step = segment_length

    segments: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + segment_length, duration)
        segments.append((start, end))
        if end >= duration:
            break
        start += step

    return segments


def overlap_add(segments: Iterable[AudioArray], overlap: float) -> AudioArray:
    """Combine audio segments with simple overlap averaging."""
    segment_list = [np.asarray(segment, dtype=np.float32) for segment in segments]
    if not segment_list:
        return np.array([], dtype=np.float32)

    first = segment_list[0]
    segment_length = first.shape[-1]
    step = max(int(segment_length * (1.0 - overlap)), 1)
    total_length = step * (len(segment_list) - 1) + segment_list[-1].shape[-1]

    if first.ndim == 1:
        output = np.zeros(total_length, dtype=np.float32)
        weights = np.zeros(total_length, dtype=np.float32)
    else:
        channels = first.shape[0]
        output = np.zeros((channels, total_length), dtype=np.float32)
        weights = np.zeros((1, total_length), dtype=np.float32)

    for index, segment in enumerate(segment_list):
        start = index * step
        end = start + segment.shape[-1]
        output[..., start:end] += segment
        weights[..., start:end] += 1.0

    return output / np.maximum(weights, 1.0)
