"""Tests for effects models and parameter validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pydantic = pytest.importorskip("pydantic")

from soundlab.effects.models import (
    ChorusConfig,
    CompressorConfig,
    DelayConfig,
    DistortionConfig,
    GainConfig,
    GateConfig,
    HighpassConfig,
    LimiterConfig,
    LowpassConfig,
    PhaserConfig,
    ReverbConfig,
)

# ---------------------------------------------------------------------------
# Parameter Validation Tests
# ---------------------------------------------------------------------------


class TestCompressorConfig:
    """Test CompressorConfig validation and plugin creation."""

    def test_default_values(self) -> None:
        cfg = CompressorConfig()
        assert cfg.threshold_db == -24.0
        assert cfg.ratio == 2.0
        assert cfg.attack_ms == 10.0
        assert cfg.release_ms == 100.0

    def test_custom_values(self) -> None:
        cfg = CompressorConfig(threshold_db=-12.0, ratio=4.0, attack_ms=5.0, release_ms=50.0)
        assert cfg.threshold_db == -12.0
        assert cfg.ratio == 4.0
        assert cfg.attack_ms == 5.0
        assert cfg.release_ms == 50.0

    def test_ratio_ge_constraint(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CompressorConfig(ratio=0.5)

    def test_attack_ge_constraint(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CompressorConfig(attack_ms=-1.0)

    def test_release_ge_constraint(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CompressorConfig(release_ms=-1.0)

    def test_frozen_model(self) -> None:
        cfg = CompressorConfig()
        with pytest.raises(pydantic.ValidationError):
            cfg.threshold_db = -10.0  # type: ignore[misc]

    def test_to_plugin_returns_plugin(self) -> None:
        mock_compressor = MagicMock()
        with patch("pedalboard.Compressor", mock_compressor):
            cfg = CompressorConfig()
            plugin = cfg.to_plugin()
            assert plugin is mock_compressor.return_value


class TestLimiterConfig:
    """Test LimiterConfig validation and plugin creation."""

    def test_default_values(self) -> None:
        cfg = LimiterConfig()
        assert cfg.threshold_db == -1.0
        assert cfg.release_ms == 100.0

    def test_custom_values(self) -> None:
        cfg = LimiterConfig(threshold_db=-3.0, release_ms=200.0)
        assert cfg.threshold_db == -3.0
        assert cfg.release_ms == 200.0

    def test_release_ge_constraint(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            LimiterConfig(release_ms=-1.0)

    def test_to_plugin_returns_plugin(self) -> None:
        mock_limiter = MagicMock()
        with patch("pedalboard.Limiter", mock_limiter):
            cfg = LimiterConfig()
            plugin = cfg.to_plugin()
            assert plugin is mock_limiter.return_value


class TestGateConfig:
    """Test GateConfig validation and plugin creation."""

    def test_default_values(self) -> None:
        cfg = GateConfig()
        assert cfg.threshold_db == -40.0
        assert cfg.ratio == 2.0
        assert cfg.attack_ms == 10.0
        assert cfg.release_ms == 100.0

    def test_ratio_ge_constraint(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            GateConfig(ratio=0.5)

    def test_to_plugin_returns_plugin(self) -> None:
        mock_gate = MagicMock()
        with patch("pedalboard.NoiseGate", mock_gate):
            cfg = GateConfig()
            plugin = cfg.to_plugin()
            assert plugin is mock_gate.return_value


class TestReverbConfig:
    """Test ReverbConfig validation and plugin creation."""

    def test_default_values(self) -> None:
        cfg = ReverbConfig()
        assert cfg.room_size == 0.5
        assert cfg.damping == 0.5
        assert cfg.wet == 0.33
        assert cfg.dry == 0.67
        assert cfg.width == 1.0

    def test_bounds_constraints(self) -> None:
        # room_size too high
        with pytest.raises(pydantic.ValidationError):
            ReverbConfig(room_size=1.5)
        # room_size too low
        with pytest.raises(pydantic.ValidationError):
            ReverbConfig(room_size=-0.1)
        # damping out of range
        with pytest.raises(pydantic.ValidationError):
            ReverbConfig(damping=2.0)
        # wet out of range
        with pytest.raises(pydantic.ValidationError):
            ReverbConfig(wet=-0.5)
        # dry out of range
        with pytest.raises(pydantic.ValidationError):
            ReverbConfig(dry=1.5)
        # width out of range
        with pytest.raises(pydantic.ValidationError):
            ReverbConfig(width=-1.0)

    def test_to_plugin_returns_plugin(self) -> None:
        mock_reverb = MagicMock()
        with patch("pedalboard.Reverb", mock_reverb):
            cfg = ReverbConfig()
            plugin = cfg.to_plugin()
            assert plugin is mock_reverb.return_value


class TestDelayConfig:
    """Test DelayConfig validation and plugin creation."""

    def test_default_values(self) -> None:
        cfg = DelayConfig()
        assert cfg.delay_seconds == 0.25
        assert cfg.feedback == 0.5
        assert cfg.mix == 0.5

    def test_feedback_bounds(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            DelayConfig(feedback=1.5)
        with pytest.raises(pydantic.ValidationError):
            DelayConfig(feedback=-0.5)

    def test_mix_bounds(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            DelayConfig(mix=2.0)

    def test_delay_ge_constraint(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            DelayConfig(delay_seconds=-1.0)

    def test_to_plugin_returns_plugin(self) -> None:
        mock_delay = MagicMock()
        with patch("pedalboard.Delay", mock_delay):
            cfg = DelayConfig()
            plugin = cfg.to_plugin()
            assert plugin is mock_delay.return_value


class TestChorusConfig:
    """Test ChorusConfig validation and plugin creation."""

    def test_default_values(self) -> None:
        cfg = ChorusConfig()
        assert cfg.rate_hz == 1.0
        assert cfg.depth == 0.5
        assert cfg.centre_delay_ms == 7.0
        assert cfg.feedback == 0.0
        assert cfg.mix == 0.5

    def test_depth_bounds(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ChorusConfig(depth=1.5)

    def test_to_plugin_returns_plugin(self) -> None:
        mock_chorus = MagicMock()
        with patch("pedalboard.Chorus", mock_chorus):
            cfg = ChorusConfig()
            plugin = cfg.to_plugin()
            assert plugin is mock_chorus.return_value


class TestDistortionConfig:
    """Test DistortionConfig validation and plugin creation."""

    def test_default_values(self) -> None:
        cfg = DistortionConfig()
        assert cfg.drive_db == 20.0

    def test_custom_drive(self) -> None:
        cfg = DistortionConfig(drive_db=30.0)
        assert cfg.drive_db == 30.0

    def test_to_plugin_returns_plugin(self) -> None:
        mock_distortion = MagicMock()
        with patch("pedalboard.Distortion", mock_distortion):
            cfg = DistortionConfig()
            plugin = cfg.to_plugin()
            assert plugin is mock_distortion.return_value


class TestPhaserConfig:
    """Test PhaserConfig validation and plugin creation."""

    def test_default_values(self) -> None:
        cfg = PhaserConfig()
        assert cfg.rate_hz == 1.0
        assert cfg.depth == 0.5
        assert cfg.centre_frequency_hz == 1300.0
        assert cfg.feedback == 0.0
        assert cfg.mix == 0.5

    def test_centre_frequency_constraint(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PhaserConfig(centre_frequency_hz=10.0)  # Below 20Hz minimum

    def test_to_plugin_returns_plugin(self) -> None:
        mock_phaser = MagicMock()
        with patch("pedalboard.Phaser", mock_phaser):
            cfg = PhaserConfig()
            plugin = cfg.to_plugin()
            assert plugin is mock_phaser.return_value


class TestHighpassConfig:
    """Test HighpassConfig validation and plugin creation."""

    def test_default_values(self) -> None:
        cfg = HighpassConfig()
        assert cfg.cutoff_frequency_hz == 80.0
        assert cfg.q == 0.707

    def test_cutoff_gt_constraint(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            HighpassConfig(cutoff_frequency_hz=0.0)
        with pytest.raises(pydantic.ValidationError):
            HighpassConfig(cutoff_frequency_hz=-100.0)

    def test_q_gt_constraint(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            HighpassConfig(q=0.0)

    def test_to_plugin_returns_plugin(self) -> None:
        mock_filter = MagicMock()
        with patch("pedalboard.HighpassFilter", mock_filter):
            cfg = HighpassConfig()
            plugin = cfg.to_plugin()
            assert plugin is mock_filter.return_value


class TestLowpassConfig:
    """Test LowpassConfig validation and plugin creation."""

    def test_default_values(self) -> None:
        cfg = LowpassConfig()
        assert cfg.cutoff_frequency_hz == 18000.0
        assert cfg.q == 0.707

    def test_cutoff_gt_constraint(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            LowpassConfig(cutoff_frequency_hz=0.0)

    def test_to_plugin_returns_plugin(self) -> None:
        mock_filter = MagicMock()
        with patch("pedalboard.LowpassFilter", mock_filter):
            cfg = LowpassConfig()
            plugin = cfg.to_plugin()
            assert plugin is mock_filter.return_value


class TestGainConfig:
    """Test GainConfig validation and plugin creation."""

    def test_default_values(self) -> None:
        cfg = GainConfig()
        assert cfg.gain_db == 0.0

    def test_custom_gain(self) -> None:
        cfg = GainConfig(gain_db=-6.0)
        assert cfg.gain_db == -6.0

    def test_to_plugin_returns_plugin(self) -> None:
        mock_gain = MagicMock()
        with patch("pedalboard.Gain", mock_gain):
            cfg = GainConfig()
            plugin = cfg.to_plugin()
            assert plugin is mock_gain.return_value


# ---------------------------------------------------------------------------
# Build Plugin Fallback Tests
# ---------------------------------------------------------------------------


class TestBuildPluginFallback:
    """Test _build_plugin fallback mechanism for attribute assignment."""

    def test_fallback_to_attribute_assignment(self) -> None:
        """When constructor fails, plugin attributes should be set directly."""
        from soundlab.effects.models import _build_plugin

        class FakePlugin:
            def __init__(self) -> None:
                self.threshold_db = 0.0
                self.ratio = 1.0

        def factory_that_rejects_kwargs(**_kwargs: object) -> FakePlugin:
            raise TypeError("Unexpected kwargs")

        # The fallback path creates FakePlugin() and sets attrs
        plugin = _build_plugin(FakePlugin, threshold_db=-10.0, ratio=3.0)
        assert plugin.threshold_db == -10.0
        assert plugin.ratio == 3.0

    def test_successful_constructor_call(self) -> None:
        """When constructor succeeds, plugin should be created normally."""
        from soundlab.effects.models import _build_plugin

        class FakePlugin:
            def __init__(self, gain_db: float = 0.0) -> None:
                self.gain_db = gain_db

        plugin = _build_plugin(FakePlugin, gain_db=-12.0)
        assert plugin.gain_db == -12.0
