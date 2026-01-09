"""Tests for EffectsChain fluent API and audio processing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
pydantic = pytest.importorskip("pydantic")


# ---------------------------------------------------------------------------
# Fluent API Tests
# ---------------------------------------------------------------------------


class TestEffectsChainFluentAPI:
    """Test EffectsChain fluent API for method chaining."""

    def test_add_returns_self(self) -> None:
        """add() should return the chain for fluent chaining."""
        with patch("soundlab.effects.chain.Pedalboard"):
            from soundlab.effects.chain import EffectsChain
            from soundlab.effects.models import GainConfig

            chain = EffectsChain()
            result = chain.add(GainConfig(gain_db=-6.0))

            assert result is chain

    def test_clear_returns_self(self) -> None:
        """clear() should return the chain for fluent chaining."""
        with patch("soundlab.effects.chain.Pedalboard"):
            from soundlab.effects.chain import EffectsChain

            chain = EffectsChain()
            result = chain.clear()

            assert result is chain

    def test_fluent_chain_multiple_effects(self) -> None:
        """Test chaining multiple add() calls."""
        with patch("soundlab.effects.chain.Pedalboard"):
            from soundlab.effects.chain import EffectsChain
            from soundlab.effects.models import CompressorConfig, GainConfig, LimiterConfig

            chain = (
                EffectsChain()
                .add(GainConfig(gain_db=3.0))
                .add(CompressorConfig(threshold_db=-20.0))
                .add(LimiterConfig(threshold_db=-1.0))
            )

            assert len(chain.effects) == 3
            assert isinstance(chain.effects[0], GainConfig)
            assert isinstance(chain.effects[1], CompressorConfig)
            assert isinstance(chain.effects[2], LimiterConfig)

    def test_effects_property_returns_tuple(self) -> None:
        """effects property should return an immutable tuple."""
        with patch("soundlab.effects.chain.Pedalboard"):
            from soundlab.effects.chain import EffectsChain
            from soundlab.effects.models import GainConfig

            chain = EffectsChain()
            chain.add(GainConfig())

            effects = chain.effects

            assert isinstance(effects, tuple)
            assert len(effects) == 1


# ---------------------------------------------------------------------------
# Empty Chain Passthrough Tests
# ---------------------------------------------------------------------------


class TestEmptyChainPassthrough:
    """Test that an empty chain passes audio through unchanged."""

    def test_empty_chain_process_array(self) -> None:
        """Empty chain should return input audio unchanged."""
        mock_pedalboard_class = MagicMock()
        mock_board = MagicMock()
        mock_pedalboard_class.return_value = mock_board

        # Make the board return input unchanged
        mock_board.side_effect = lambda audio, _sr: audio

        with patch("soundlab.effects.chain.Pedalboard", mock_pedalboard_class):
            from soundlab.effects.chain import EffectsChain

            chain = EffectsChain()
            audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

            result = chain.process_array(audio, sample_rate=44100)

            # Board should be called with empty list of plugins
            mock_pedalboard_class.assert_called_once_with([])
            assert np.array_equal(result, audio)

    def test_empty_chain_effects_list(self) -> None:
        """Empty chain should have no effects."""
        with patch("soundlab.effects.chain.Pedalboard"):
            from soundlab.effects.chain import EffectsChain

            chain = EffectsChain()

            assert len(chain.effects) == 0
            assert chain.effects == ()


# ---------------------------------------------------------------------------
# Multi-Effect Processing Tests
# ---------------------------------------------------------------------------


class TestMultiEffectProcessing:
    """Test processing audio through multiple effects."""

    def test_multiple_effects_plugins_created(self) -> None:
        """Each effect config should create a plugin."""
        mock_pedalboard_class = MagicMock()
        mock_board = MagicMock()
        mock_pedalboard_class.return_value = mock_board
        mock_board.side_effect = lambda audio, _sr: audio

        with patch("soundlab.effects.chain.Pedalboard", mock_pedalboard_class):
            from soundlab.effects.chain import EffectsChain
            from soundlab.effects.models import CompressorConfig, GainConfig, ReverbConfig

            # Patch individual effect plugins
            with (
                patch("pedalboard.Gain") as mock_gain,
                patch("pedalboard.Compressor") as mock_comp,
                patch("pedalboard.Reverb") as mock_reverb,
            ):
                chain = (
                    EffectsChain()
                    .add(GainConfig(gain_db=-6.0))
                    .add(CompressorConfig(threshold_db=-18.0))
                    .add(ReverbConfig(room_size=0.7))
                )

                audio = np.random.randn(44100).astype(np.float32)
                _result = chain.process_array(audio, sample_rate=44100)

                # Verify plugins were instantiated
                mock_gain.assert_called_once()
                mock_comp.assert_called_once()
                mock_reverb.assert_called_once()

                # Verify Pedalboard was created with 3 plugins
                call_args = mock_pedalboard_class.call_args[0][0]
                assert len(call_args) == 3

    def test_effect_order_preserved(self) -> None:
        """Effects should be applied in the order they were added."""
        mock_pedalboard_class = MagicMock()
        mock_board = MagicMock()
        mock_pedalboard_class.return_value = mock_board
        mock_board.side_effect = lambda audio, _sr: audio

        plugin_order = []

        def track_gain(**_kwargs: object) -> MagicMock:
            plugin = MagicMock(name="gain_plugin")
            plugin_order.append("gain")
            return plugin

        def track_compressor(**_kwargs: object) -> MagicMock:
            plugin = MagicMock(name="compressor_plugin")
            plugin_order.append("compressor")
            return plugin

        def track_limiter(**_kwargs: object) -> MagicMock:
            plugin = MagicMock(name="limiter_plugin")
            plugin_order.append("limiter")
            return plugin

        with patch("soundlab.effects.chain.Pedalboard", mock_pedalboard_class):
            from soundlab.effects.chain import EffectsChain
            from soundlab.effects.models import CompressorConfig, GainConfig, LimiterConfig

            with (
                patch("pedalboard.Gain", side_effect=track_gain),
                patch("pedalboard.Compressor", side_effect=track_compressor),
                patch("pedalboard.Limiter", side_effect=track_limiter),
            ):
                chain = (
                    EffectsChain().add(GainConfig()).add(CompressorConfig()).add(LimiterConfig())
                )

                audio = np.zeros(1000, dtype=np.float32)
                chain.process_array(audio, sample_rate=44100)

        assert plugin_order == ["gain", "compressor", "limiter"]


# ---------------------------------------------------------------------------
# Board Caching Tests
# ---------------------------------------------------------------------------


class TestBoardCaching:
    """Test that the Pedalboard is cached and rebuilt when needed."""

    def test_board_cached_between_calls(self) -> None:
        """Pedalboard should be cached between process calls."""
        mock_pedalboard_class = MagicMock()
        mock_board = MagicMock()
        mock_pedalboard_class.return_value = mock_board
        mock_board.side_effect = lambda audio, _sr: audio

        with patch("soundlab.effects.chain.Pedalboard", mock_pedalboard_class):
            from soundlab.effects.chain import EffectsChain
            from soundlab.effects.models import GainConfig

            with patch("pedalboard.Gain"):
                chain = EffectsChain().add(GainConfig())

                audio = np.zeros(1000, dtype=np.float32)
                chain.process_array(audio, sample_rate=44100)
                chain.process_array(audio, sample_rate=44100)
                chain.process_array(audio, sample_rate=44100)

        # Pedalboard should only be created once
        assert mock_pedalboard_class.call_count == 1

    def test_board_rebuilt_after_add(self) -> None:
        """Adding an effect should invalidate the cached board."""
        mock_pedalboard_class = MagicMock()
        mock_board = MagicMock()
        mock_pedalboard_class.return_value = mock_board
        mock_board.side_effect = lambda audio, _sr: audio

        with patch("soundlab.effects.chain.Pedalboard", mock_pedalboard_class):
            from soundlab.effects.chain import EffectsChain
            from soundlab.effects.models import GainConfig

            with patch("pedalboard.Gain"):
                chain = EffectsChain().add(GainConfig())

                audio = np.zeros(1000, dtype=np.float32)
                chain.process_array(audio, sample_rate=44100)  # First build

                chain.add(GainConfig())  # Invalidates cache
                chain.process_array(audio, sample_rate=44100)  # Second build

        assert mock_pedalboard_class.call_count == 2

    def test_board_rebuilt_after_clear(self) -> None:
        """Clearing effects should invalidate the cached board."""
        mock_pedalboard_class = MagicMock()
        mock_board = MagicMock()
        mock_pedalboard_class.return_value = mock_board
        mock_board.side_effect = lambda audio, _sr: audio

        with patch("soundlab.effects.chain.Pedalboard", mock_pedalboard_class):
            from soundlab.effects.chain import EffectsChain
            from soundlab.effects.models import GainConfig

            with patch("pedalboard.Gain"):
                chain = EffectsChain().add(GainConfig())

                audio = np.zeros(1000, dtype=np.float32)
                chain.process_array(audio, sample_rate=44100)  # First build

                chain.clear()
                chain.process_array(audio, sample_rate=44100)  # Second build (empty)

        assert mock_pedalboard_class.call_count == 2


# ---------------------------------------------------------------------------
# File Processing Tests
# ---------------------------------------------------------------------------


class TestFileProcessing:
    """Test processing audio files through the chain."""

    def test_process_file(self, tmp_path: Path) -> None:
        """Test process() loads, processes, and saves audio file."""
        mock_pedalboard_class = MagicMock()
        mock_board = MagicMock()
        mock_pedalboard_class.return_value = mock_board
        mock_board.side_effect = lambda audio, _sr: audio

        mock_segment = MagicMock()
        mock_segment.samples = np.zeros(1000, dtype=np.float32)
        mock_segment.sample_rate = 44100

        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        input_path.touch()

        with (
            patch("soundlab.effects.chain.Pedalboard", mock_pedalboard_class),
            patch("soundlab.effects.chain.load_audio", return_value=mock_segment) as mock_load,
            patch("soundlab.effects.chain.save_audio") as mock_save,
        ):
            from soundlab.effects.chain import EffectsChain

            chain = EffectsChain()
            result = chain.process(input_path, output_path)

            mock_load.assert_called_once_with(input_path)
            mock_save.assert_called_once()
            assert result == output_path

    def test_process_returns_path(self, tmp_path: Path) -> None:
        """process() should return the output path."""
        mock_pedalboard_class = MagicMock()
        mock_board = MagicMock()
        mock_pedalboard_class.return_value = mock_board
        mock_board.side_effect = lambda audio, _sr: audio

        mock_segment = MagicMock()
        mock_segment.samples = np.zeros(1000, dtype=np.float32)
        mock_segment.sample_rate = 44100

        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "processed.wav"
        input_path.touch()

        with (
            patch("soundlab.effects.chain.Pedalboard", mock_pedalboard_class),
            patch("soundlab.effects.chain.load_audio", return_value=mock_segment),
            patch("soundlab.effects.chain.save_audio"),
        ):
            from soundlab.effects.chain import EffectsChain

            chain = EffectsChain()
            result = chain.process(str(input_path), str(output_path))

            assert isinstance(result, Path)
            assert result == Path(output_path)


# ---------------------------------------------------------------------------
# Clear Effects Tests
# ---------------------------------------------------------------------------


class TestClearEffects:
    """Test clearing effects from the chain."""

    def test_clear_removes_all_effects(self) -> None:
        """clear() should remove all effects."""
        with patch("soundlab.effects.chain.Pedalboard"):
            from soundlab.effects.chain import EffectsChain
            from soundlab.effects.models import CompressorConfig, GainConfig, LimiterConfig

            chain = EffectsChain().add(GainConfig()).add(CompressorConfig()).add(LimiterConfig())

            assert len(chain.effects) == 3

            chain.clear()

            assert len(chain.effects) == 0
            assert chain.effects == ()

    def test_clear_allows_rebuilding(self) -> None:
        """After clear(), new effects can be added."""
        with patch("soundlab.effects.chain.Pedalboard"):
            from soundlab.effects.chain import EffectsChain
            from soundlab.effects.models import GainConfig, ReverbConfig

            chain = EffectsChain().add(GainConfig())
            chain.clear()
            chain.add(ReverbConfig())

            assert len(chain.effects) == 1
            assert isinstance(chain.effects[0], ReverbConfig)
