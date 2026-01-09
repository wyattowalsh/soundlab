"""Effects chain orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pedalboard import Pedalboard

from soundlab.io import load_audio, save_audio

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from soundlab.effects.models import EffectConfig


class AudioEffect(Protocol):
    """Protocol for audio effects."""

    def process(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        """Apply the effect to audio samples."""
        ...


class EffectsChain:
    """Chainable audio effects processor."""

    def __init__(self) -> None:
        self._effects: list[EffectConfig] = []
        self._board: Pedalboard | None = None

    def add(self, effect: EffectConfig) -> EffectsChain:
        """Add an effect to the chain (fluent API)."""
        self._effects.append(effect)
        self._board = None
        return self

    def clear(self) -> EffectsChain:
        """Clear all effects from the chain."""
        self._effects.clear()
        self._board = None
        return self

    @property
    def effects(self) -> Sequence[EffectConfig]:
        """Current effects in the chain."""
        return tuple(self._effects)

    def _build_board(self) -> Pedalboard:
        if self._board is not None:
            return self._board

        plugins = [effect.to_plugin() for effect in self._effects]
        self._board = Pedalboard(plugins)
        return self._board

    def process_array(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> NDArray[np.float32]:
        """Process audio samples through the effects chain."""
        board = self._build_board()
        return board(audio, sample_rate)

    def process(self, input_path: str | Path, output_path: str | Path) -> Path:
        """Process an audio file and write the result."""
        segment = load_audio(input_path)
        processed = self.process_array(segment.samples, segment.sample_rate)
        segment.samples = processed
        save_audio(segment, output_path)
        return Path(output_path)


__all__ = ["AudioEffect", "EffectsChain"]
