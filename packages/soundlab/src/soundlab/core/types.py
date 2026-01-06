"""Type aliases and protocols for SoundLab."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

__all__ = [
    "AudioArray",
    "SampleRateHz",
    "PathLike",
    "ProgressCallback",
    "AudioProcessor",
]

# Type aliases
AudioArray = NDArray[np.float32]
SampleRateHz = int
PathLike = str | Path


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks."""

    def __call__(self, current: int, total: int, message: str = "") -> None:
        """
        Report progress.

        Parameters
        ----------
        current
            Current progress value.
        total
            Total value for completion.
        message
            Optional status message.
        """
        ...


@runtime_checkable
class AudioProcessor(Protocol):
    """Protocol for audio processing components."""

    def process(self, audio: AudioArray, sample_rate: SampleRateHz) -> AudioArray:
        """
        Process audio samples.

        Parameters
        ----------
        audio
            Input audio samples.
        sample_rate
            Sample rate in Hz.

        Returns
        -------
        AudioArray
            Processed audio samples.
        """
        ...
