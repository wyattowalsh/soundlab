"""Effects chain orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from loguru import logger
from numpy.typing import NDArray
from pedalboard import Pedalboard

from soundlab.effects.models import EffectConfig

if TYPE_CHECKING:
    from soundlab.core.types import PathLike


__all__ = ["EffectsChain"]


class EffectsChain:
    """
    Chainable audio effects processor.

    Examples
    --------
    >>> from soundlab.effects import EffectsChain
    >>> from soundlab.effects.models import CompressorConfig, ReverbConfig
    >>>
    >>> chain = EffectsChain()
    >>> chain.add(CompressorConfig(threshold_db=-20, ratio=4.0))
    >>> chain.add(ReverbConfig(room_size=0.5, wet_level=0.3))
    >>>
    >>> result = chain.process("input.wav", "output.wav")
    """

    def __init__(self) -> None:
        """Initialize an empty effects chain."""
        self._effects: list[EffectConfig] = []
        self._board: Pedalboard | None = None

    def add(self, effect: EffectConfig) -> "EffectsChain":
        """
        Add an effect to the chain (fluent API).

        Parameters
        ----------
        effect
            Effect configuration to add.

        Returns
        -------
        EffectsChain
            Self for method chaining.
        """
        self._effects.append(effect)
        self._board = None  # Invalidate cached board
        logger.debug(f"Added effect: {effect.name}")
        return self

    def insert(self, index: int, effect: EffectConfig) -> "EffectsChain":
        """
        Insert an effect at a specific position.

        Parameters
        ----------
        index
            Position to insert at.
        effect
            Effect configuration to insert.

        Returns
        -------
        EffectsChain
            Self for method chaining.
        """
        self._effects.insert(index, effect)
        self._board = None
        return self

    def remove(self, index: int) -> "EffectsChain":
        """
        Remove an effect at a specific position.

        Parameters
        ----------
        index
            Position to remove from.

        Returns
        -------
        EffectsChain
            Self for method chaining.
        """
        del self._effects[index]
        self._board = None
        return self

    def clear(self) -> "EffectsChain":
        """
        Clear all effects from the chain.

        Returns
        -------
        EffectsChain
            Self for method chaining.
        """
        self._effects.clear()
        self._board = None
        logger.debug("Effects chain cleared")
        return self

    @property
    def effects(self) -> Sequence[EffectConfig]:
        """Current effects in the chain."""
        return tuple(self._effects)

    @property
    def effect_names(self) -> list[str]:
        """Names of effects in the chain."""
        return [e.name for e in self._effects]

    def __len__(self) -> int:
        """Number of effects in the chain."""
        return len(self._effects)

    def __repr__(self) -> str:
        """String representation."""
        if not self._effects:
            return "EffectsChain(empty)"
        names = " -> ".join(self.effect_names)
        return f"EffectsChain({names})"

    def _build_board(self) -> Pedalboard:
        """Build the Pedalboard from effect configs."""
        if self._board is not None:
            return self._board

        # Only include enabled effects
        enabled_effects = [e for e in self._effects if e.enabled]
        plugins = [effect.to_plugin() for effect in enabled_effects]
        self._board = Pedalboard(plugins)

        logger.debug(f"Built pedalboard with {len(plugins)} effects")
        return self._board

    def process_array(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> NDArray[np.float32]:
        """
        Process audio samples through the effects chain.

        Parameters
        ----------
        audio
            Audio samples as numpy array. Shape: (samples,) or (channels, samples).
        sample_rate
            Sample rate of the audio.

        Returns
        -------
        NDArray[np.float32]
            Processed audio samples.
        """
        if not self._effects:
            return audio

        board = self._build_board()

        # Ensure correct shape for pedalboard (channels, samples)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Process
        processed = board(audio, sample_rate)

        logger.debug(f"Processed {audio.shape[1] / sample_rate:.2f}s of audio")

        return processed

    def process(
        self,
        input_path: PathLike,
        output_path: PathLike,
        *,
        preserve_format: bool = True,
    ) -> Path:
        """
        Process an audio file through the effects chain.

        Parameters
        ----------
        input_path
            Path to input audio file.
        output_path
            Path to save processed audio.
        preserve_format
            Try to preserve input format characteristics.

        Returns
        -------
        Path
            Path to the processed audio file.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info(f"Processing: {input_path} -> {output_path}")

        # Load audio
        audio, sr = sf.read(input_path, dtype="float32")

        # Ensure correct shape (samples, channels) -> (channels, samples) for pedalboard
        if audio.ndim == 2:
            audio = audio.T

        # Process
        processed = self.process_array(audio, sr)

        # Save - convert back to (samples, channels)
        if processed.ndim == 2:
            processed = processed.T

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine subtype
        if preserve_format:
            try:
                info = sf.info(input_path)
                subtype = info.subtype
            except Exception:
                subtype = "PCM_24"
        else:
            subtype = "PCM_24"

        sf.write(output_path, processed, sr, subtype=subtype)

        logger.info(f"Saved processed audio: {output_path}")

        return output_path

    def preview(
        self,
        input_path: PathLike,
        duration: float = 10.0,
        start: float = 0.0,
    ) -> NDArray[np.float32]:
        """
        Preview effect chain on a portion of audio.

        Parameters
        ----------
        input_path
            Path to input audio file.
        duration
            Duration to preview in seconds.
        start
            Start time in seconds.

        Returns
        -------
        NDArray[np.float32]
            Processed audio preview.
        """
        input_path = Path(input_path)

        # Load portion of audio
        info = sf.info(input_path)
        start_frame = int(start * info.samplerate)
        frames = int(duration * info.samplerate)

        audio, sr = sf.read(
            input_path,
            dtype="float32",
            start=start_frame,
            frames=frames,
        )

        if audio.ndim == 2:
            audio = audio.T

        return self.process_array(audio, sr)

    @classmethod
    def from_configs(cls, configs: Sequence[EffectConfig]) -> "EffectsChain":
        """
        Create an effects chain from a sequence of configs.

        Parameters
        ----------
        configs
            Sequence of effect configurations.

        Returns
        -------
        EffectsChain
            New effects chain.
        """
        chain = cls()
        for config in configs:
            chain.add(config)
        return chain
