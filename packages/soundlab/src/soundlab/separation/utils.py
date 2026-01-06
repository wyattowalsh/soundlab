"""Utilities for stem separation processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


__all__ = [
    "calculate_segments",
    "overlap_add",
    "estimate_memory_usage",
]


def calculate_segments(
    duration: float,
    segment_length: float,
    overlap: float,
) -> list[tuple[float, float]]:
    """
    Calculate segment boundaries for processing long audio.

    Parameters
    ----------
    duration
        Total duration in seconds.
    segment_length
        Length of each segment in seconds.
    overlap
        Overlap ratio between segments (0.0 to 1.0).

    Returns
    -------
    list[tuple[float, float]]
        List of (start_time, end_time) tuples for each segment.
    """
    if duration <= segment_length:
        return [(0.0, duration)]

    step = segment_length * (1 - overlap)
    segments = []
    start = 0.0

    while start < duration:
        end = min(start + segment_length, duration)
        segments.append((start, end))
        start += step

        # Avoid tiny final segments
        if duration - start < segment_length * 0.5:
            # Extend the last segment to cover the rest
            if segments:
                segments[-1] = (segments[-1][0], duration)
            break

    return segments


def overlap_add(
    segments: list[NDArray[np.float32]],
    overlap: float,
    sample_rate: int,
) -> NDArray[np.float32]:
    """
    Combine segments using overlap-add method.

    Parameters
    ----------
    segments
        List of audio segments as numpy arrays.
    overlap
        Overlap ratio used during segmentation.
    sample_rate
        Sample rate of the audio.

    Returns
    -------
    NDArray[np.float32]
        Combined audio.
    """
    if len(segments) == 0:
        return np.array([], dtype=np.float32)

    if len(segments) == 1:
        return segments[0]

    # Calculate overlap samples
    segment_samples = len(segments[0])
    overlap_samples = int(segment_samples * overlap)
    step_samples = segment_samples - overlap_samples

    # Calculate total output length
    total_samples = step_samples * (len(segments) - 1) + segment_samples

    # Handle multi-channel audio
    if segments[0].ndim == 2:
        channels = segments[0].shape[0]
        output = np.zeros((channels, total_samples), dtype=np.float32)
        weights = np.zeros(total_samples, dtype=np.float32)
    else:
        output = np.zeros(total_samples, dtype=np.float32)
        weights = np.zeros(total_samples, dtype=np.float32)

    # Create fade window for crossfading
    fade_in = np.linspace(0, 1, overlap_samples, dtype=np.float32)
    fade_out = np.linspace(1, 0, overlap_samples, dtype=np.float32)

    for i, segment in enumerate(segments):
        start_idx = i * step_samples
        end_idx = start_idx + len(segment) if segment.ndim == 1 else start_idx + segment.shape[1]

        # Apply fades
        seg_copy = segment.copy()

        if i > 0 and overlap_samples > 0:
            # Fade in at the beginning (except first segment)
            if seg_copy.ndim == 2:
                seg_copy[:, :overlap_samples] *= fade_in
            else:
                seg_copy[:overlap_samples] *= fade_in

        if i < len(segments) - 1 and overlap_samples > 0:
            # Fade out at the end (except last segment)
            if seg_copy.ndim == 2:
                seg_copy[:, -overlap_samples:] *= fade_out
            else:
                seg_copy[-overlap_samples:] *= fade_out

        # Add to output
        seg_len = segment.shape[1] if segment.ndim == 2 else len(segment)
        actual_end = min(end_idx, total_samples)
        actual_len = actual_end - start_idx

        if segment.ndim == 2:
            output[:, start_idx:actual_end] += seg_copy[:, :actual_len]
        else:
            output[start_idx:actual_end] += seg_copy[:actual_len]

        # Track weights for normalization
        weight_segment = np.ones(seg_len, dtype=np.float32)
        if i > 0 and overlap_samples > 0:
            weight_segment[:overlap_samples] = fade_in
        if i < len(segments) - 1 and overlap_samples > 0:
            weight_segment[-overlap_samples:] = fade_out

        weights[start_idx:actual_end] += weight_segment[:actual_len]

    # Normalize by weights
    weights = np.maximum(weights, 1e-8)  # Avoid division by zero
    if output.ndim == 2:
        output /= weights
    else:
        output /= weights

    return output


def estimate_memory_usage(
    duration_seconds: float,
    sample_rate: int = 44100,
    channels: int = 2,
    model_name: str = "htdemucs_ft",
) -> dict[str, float]:
    """
    Estimate memory requirements for separation.

    Parameters
    ----------
    duration_seconds
        Audio duration in seconds.
    sample_rate
        Sample rate.
    channels
        Number of channels.
    model_name
        Model name for estimation.

    Returns
    -------
    dict[str, float]
        Estimated memory usage in GB.
    """
    # Base memory for model
    model_memory = {
        "htdemucs": 1.5,
        "htdemucs_ft": 2.0,
        "htdemucs_6s": 2.5,
        "mdx_extra": 1.8,
        "mdx_extra_q": 1.2,
    }

    base_gb = model_memory.get(model_name, 2.0)

    # Audio memory: samples × channels × 4 bytes × 2 (input + output)
    samples = int(duration_seconds * sample_rate)
    audio_gb = (samples * channels * 4 * 2) / (1024 ** 3)

    # Processing overhead (roughly 0.5GB per minute)
    processing_gb = (duration_seconds / 60) * 0.5

    return {
        "model_gb": base_gb,
        "audio_gb": audio_gb,
        "processing_gb": processing_gb,
        "total_gb": base_gb + audio_gb + processing_gb,
    }
