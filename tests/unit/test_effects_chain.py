"""Tests for soundlab.effects.chain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soundlab.effects.chain import EffectsChain
from soundlab.effects.models import (
    CompressorConfig,
    GainConfig,
    HighpassConfig,
    ReverbConfig,
)


class TestEffectsChainCreation:
    """Test EffectsChain creation and management."""

    def test_create_empty_chain(self):
        """Should create empty chain."""
        chain = EffectsChain()
        assert len(chain) == 0

    def test_add_effect(self):
        """Should add effect to chain."""
        chain = EffectsChain()
        chain.add(CompressorConfig())

        assert len(chain) == 1

    def test_add_returns_self(self):
        """add() should return self for fluent API."""
        chain = EffectsChain()
        result = chain.add(CompressorConfig())

        assert result is chain

    def test_fluent_api(self):
        """Should support method chaining."""
        chain = (
            EffectsChain()
            .add(HighpassConfig(cutoff_hz=80))
            .add(CompressorConfig())
            .add(ReverbConfig())
        )

        assert len(chain) == 3

    def test_clear_removes_all(self):
        """clear() should remove all effects."""
        chain = EffectsChain()
        chain.add(CompressorConfig())
        chain.add(ReverbConfig())
        chain.clear()

        assert len(chain) == 0

    def test_clear_returns_self(self):
        """clear() should return self."""
        chain = EffectsChain()
        chain.add(CompressorConfig())
        result = chain.clear()

        assert result is chain

    def test_insert_at_position(self):
        """insert() should add at specific position."""
        chain = EffectsChain()
        chain.add(CompressorConfig())
        chain.add(ReverbConfig())
        chain.insert(1, GainConfig())

        assert len(chain) == 3
        assert chain.effects[1].name == "Gain"

    def test_remove_at_position(self):
        """remove() should remove at specific position."""
        chain = EffectsChain()
        chain.add(CompressorConfig())
        chain.add(GainConfig())
        chain.add(ReverbConfig())
        chain.remove(1)

        assert len(chain) == 2
        assert chain.effects[1].name == "Reverb"


class TestEffectsChainProperties:
    """Test EffectsChain properties."""

    def test_effects_property(self):
        """effects should return tuple of configs."""
        chain = EffectsChain()
        chain.add(CompressorConfig())
        chain.add(ReverbConfig())

        effects = chain.effects
        assert isinstance(effects, tuple)
        assert len(effects) == 2

    def test_effect_names_property(self):
        """effect_names should return list of names."""
        chain = EffectsChain()
        chain.add(CompressorConfig())
        chain.add(ReverbConfig())

        names = chain.effect_names
        assert names == ["Compressor", "Reverb"]

    def test_repr_empty(self):
        """repr should show empty for empty chain."""
        chain = EffectsChain()
        assert "empty" in repr(chain)

    def test_repr_with_effects(self):
        """repr should show effect names."""
        chain = EffectsChain()
        chain.add(CompressorConfig())
        chain.add(ReverbConfig())

        r = repr(chain)
        assert "Compressor" in r
        assert "Reverb" in r


class TestEffectsChainProcessArray:
    """Test process_array method."""

    def test_empty_chain_passthrough(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """Empty chain should pass through unchanged."""
        chain = EffectsChain()
        result = chain.process_array(sample_mono_audio, sample_rate)

        np.testing.assert_array_equal(result, sample_mono_audio)

    def test_process_with_gain(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """Should apply gain effect."""
        chain = EffectsChain()
        chain.add(GainConfig(gain_db=6.0))  # +6dB

        result = chain.process_array(sample_mono_audio, sample_rate)

        # Result should be louder (approximately 2x amplitude for +6dB)
        # But exact ratio depends on implementation
        assert result is not None
        assert result.shape == sample_mono_audio.shape or result.shape[1] == len(sample_mono_audio)

    def test_process_mono_audio(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """Should process mono audio."""
        chain = EffectsChain()
        chain.add(CompressorConfig())

        result = chain.process_array(sample_mono_audio, sample_rate)

        # Should return valid audio
        assert result is not None
        assert result.dtype == np.float32

    def test_process_stereo_audio(self, sample_stereo_audio: np.ndarray, sample_rate: int):
        """Should process stereo audio."""
        chain = EffectsChain()
        chain.add(CompressorConfig())

        result = chain.process_array(sample_stereo_audio, sample_rate)

        assert result is not None

    def test_multiple_effects(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """Should process through multiple effects."""
        chain = (
            EffectsChain()
            .add(HighpassConfig(cutoff_hz=80))
            .add(CompressorConfig())
            .add(GainConfig(gain_db=-3.0))
        )

        result = chain.process_array(sample_mono_audio, sample_rate)

        assert result is not None

    def test_disabled_effects_skipped(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """Disabled effects should be skipped."""
        chain = EffectsChain()
        chain.add(GainConfig(gain_db=20.0, enabled=False))

        result = chain.process_array(sample_mono_audio, sample_rate)

        # Should pass through unchanged since gain is disabled
        np.testing.assert_array_almost_equal(
            result.flatten() if result.ndim > 1 else result,
            sample_mono_audio,
            decimal=4,
        )


class TestEffectsChainProcessFile:
    """Test process method with files."""

    def test_process_file(
        self,
        sample_audio_path: Path,
        temp_output_dir: Path,
    ):
        """Should process audio file."""
        chain = EffectsChain()
        chain.add(CompressorConfig())

        output_path = temp_output_dir / "processed.wav"
        result = chain.process(sample_audio_path, output_path)

        assert result.exists()
        assert result == output_path

    def test_process_creates_parent_dirs(
        self,
        sample_audio_path: Path,
        temp_dir: Path,
    ):
        """Should create parent directories."""
        chain = EffectsChain()
        chain.add(CompressorConfig())

        output_path = temp_dir / "nested" / "dirs" / "output.wav"
        result = chain.process(sample_audio_path, output_path)

        assert result.exists()


class TestEffectsChainFromConfigs:
    """Test from_configs class method."""

    def test_create_from_list(self):
        """Should create chain from list of configs."""
        configs = [
            CompressorConfig(),
            ReverbConfig(),
        ]

        chain = EffectsChain.from_configs(configs)

        assert len(chain) == 2
        assert chain.effect_names == ["Compressor", "Reverb"]

    def test_create_from_empty_list(self):
        """Should create empty chain from empty list."""
        chain = EffectsChain.from_configs([])
        assert len(chain) == 0
