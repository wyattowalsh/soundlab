"""Core type aliases and protocols."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

AudioArray = NDArray[np.float32]
SampleRate = int
PathLike = str | Path


class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""

    def __call__(self, progress: float, message: str | None = None) -> None:
        """Report progress as a float from 0.0 to 1.0 with an optional message."""


class AudioProcessor(Protocol):
    """Protocol for audio processor callables."""

    def __call__(self, audio: AudioSegment) -> AudioSegment:
        """Process an AudioSegment and return the processed audio."""


if TYPE_CHECKING:
    from soundlab.core.audio import AudioSegment
