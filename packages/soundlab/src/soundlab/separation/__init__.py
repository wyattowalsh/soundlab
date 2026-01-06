"""Stem separation module for SoundLab."""

from soundlab.separation.demucs import StemSeparator
from soundlab.separation.models import DemucsModel, SeparationConfig, StemResult
from soundlab.separation.utils import (
    calculate_segments,
    estimate_memory_usage,
    overlap_add,
)

__all__ = [
    # Main class
    "StemSeparator",
    # Models
    "DemucsModel",
    "SeparationConfig",
    "StemResult",
    # Utilities
    "calculate_segments",
    "estimate_memory_usage",
    "overlap_add",
]
