"""Tests for soundlab.effects.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soundlab.effects.models import (
    ChorusConfig,
    ClippingConfig,
    CompressorConfig,
    DelayConfig,
    DistortionConfig,
    EffectConfig,
    GainConfig,
    GateConfig,
    HighpassConfig,
    HighShelfConfig,
    LimiterConfig,
    LowpassConfig,
    LowShelfConfig,
    PeakFilterConfig,
    PhaserConfig,
    ReverbConfig,
)


class TestEffectConfigBase:
    """Test base EffectConfig class."""

    def test_enabled_default(self):
        """Effects should be enabled by default."""
        config = CompressorConfig()
        assert config.enabled is True

    def test_can_disable_effect(self):
        """Should be able to disable effect."""
        config = CompressorConfig(enabled=False)
        assert config.enabled is False


class TestCompressorConfig:
    """Test CompressorConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = CompressorConfig()
        assert config.threshold_db == -20.0
        assert config.ratio == 4.0
        assert config.attack_ms == 10.0
        assert config.release_ms == 100.0

    def test_threshold_bounds(self):
        """Threshold should be -60 to 0 dB."""
        CompressorConfig(threshold_db=-60.0)
        CompressorConfig(threshold_db=0.0)

        with pytest.raises(ValidationError):
            CompressorConfig(threshold_db=1.0)
        with pytest.raises(ValidationError):
            CompressorConfig(threshold_db=-65.0)

    def test_ratio_bounds(self):
        """Ratio should be 1 to 20."""
        CompressorConfig(ratio=1.0)
        CompressorConfig(ratio=20.0)

        with pytest.raises(ValidationError):
            CompressorConfig(ratio=0.5)

    def test_name_property(self):
        """Should have 'Compressor' name."""
        assert CompressorConfig().name == "Compressor"

    def test_to_plugin(self):
        """Should create Pedalboard Compressor."""
        from pedalboard import Compressor

        config = CompressorConfig(threshold_db=-18.0, ratio=3.0)
        plugin = config.to_plugin()

        assert isinstance(plugin, Compressor)


class TestLimiterConfig:
    """Test LimiterConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = LimiterConfig()
        assert config.threshold_db == -1.0

    def test_to_plugin(self):
        """Should create Pedalboard Limiter."""
        from pedalboard import Limiter

        config = LimiterConfig()
        plugin = config.to_plugin()

        assert isinstance(plugin, Limiter)


class TestGateConfig:
    """Test GateConfig (NoiseGate)."""

    def test_to_plugin(self):
        """Should create Pedalboard NoiseGate."""
        from pedalboard import NoiseGate

        config = GateConfig()
        plugin = config.to_plugin()

        assert isinstance(plugin, NoiseGate)


class TestGainConfig:
    """Test GainConfig."""

    def test_default_is_unity(self):
        """Default gain should be 0 dB (unity)."""
        config = GainConfig()
        assert config.gain_db == 0.0

    def test_gain_bounds(self):
        """Gain should be -60 to +60 dB."""
        GainConfig(gain_db=-60.0)
        GainConfig(gain_db=60.0)

        with pytest.raises(ValidationError):
            GainConfig(gain_db=65.0)

    def test_to_plugin(self):
        """Should create Pedalboard Gain."""
        from pedalboard import Gain

        config = GainConfig(gain_db=6.0)
        plugin = config.to_plugin()

        assert isinstance(plugin, Gain)


class TestHighpassConfig:
    """Test HighpassConfig."""

    def test_default_cutoff(self):
        """Default cutoff should be 80 Hz."""
        config = HighpassConfig()
        assert config.cutoff_hz == 80.0

    def test_cutoff_bounds(self):
        """Cutoff should be 20-20000 Hz."""
        HighpassConfig(cutoff_hz=20.0)
        HighpassConfig(cutoff_hz=20000.0)

        with pytest.raises(ValidationError):
            HighpassConfig(cutoff_hz=10.0)

    def test_to_plugin(self):
        """Should create Pedalboard HighpassFilter."""
        from pedalboard import HighpassFilter

        config = HighpassConfig()
        plugin = config.to_plugin()

        assert isinstance(plugin, HighpassFilter)


class TestLowpassConfig:
    """Test LowpassConfig."""

    def test_to_plugin(self):
        """Should create Pedalboard LowpassFilter."""
        from pedalboard import LowpassFilter

        config = LowpassConfig()
        plugin = config.to_plugin()

        assert isinstance(plugin, LowpassFilter)


class TestReverbConfig:
    """Test ReverbConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = ReverbConfig()
        assert config.room_size == 0.5
        assert config.wet_level == 0.33

    def test_room_size_bounds(self):
        """Room size should be 0-1."""
        ReverbConfig(room_size=0.0)
        ReverbConfig(room_size=1.0)

        with pytest.raises(ValidationError):
            ReverbConfig(room_size=1.5)

    def test_to_plugin(self):
        """Should create Pedalboard Reverb."""
        from pedalboard import Reverb

        config = ReverbConfig()
        plugin = config.to_plugin()

        assert isinstance(plugin, Reverb)


class TestDelayConfig:
    """Test DelayConfig."""

    def test_to_plugin(self):
        """Should create Pedalboard Delay."""
        from pedalboard import Delay

        config = DelayConfig()
        plugin = config.to_plugin()

        assert isinstance(plugin, Delay)


class TestChorusConfig:
    """Test ChorusConfig."""

    def test_to_plugin(self):
        """Should create Pedalboard Chorus."""
        from pedalboard import Chorus

        config = ChorusConfig()
        plugin = config.to_plugin()

        assert isinstance(plugin, Chorus)


class TestDistortionConfig:
    """Test DistortionConfig."""

    def test_to_plugin(self):
        """Should create Pedalboard Distortion."""
        from pedalboard import Distortion

        config = DistortionConfig()
        plugin = config.to_plugin()

        assert isinstance(plugin, Distortion)


class TestAllEffectsHaveName:
    """Test that all effects have name property."""

    @pytest.mark.parametrize("config_class", [
        CompressorConfig,
        LimiterConfig,
        GateConfig,
        GainConfig,
        HighpassConfig,
        LowpassConfig,
        HighShelfConfig,
        LowShelfConfig,
        PeakFilterConfig,
        ReverbConfig,
        DelayConfig,
        ChorusConfig,
        PhaserConfig,
        DistortionConfig,
        ClippingConfig,
    ])
    def test_has_name(self, config_class):
        """Each effect should have a name property."""
        config = config_class()
        assert isinstance(config.name, str)
        assert len(config.name) > 0


class TestAllEffectsAreFrozen:
    """Test that all effect configs are immutable."""

    @pytest.mark.parametrize("config_class", [
        CompressorConfig,
        LimiterConfig,
        GateConfig,
        GainConfig,
        ReverbConfig,
        DelayConfig,
    ])
    def test_is_frozen(self, config_class):
        """Effect configs should be immutable."""
        config = config_class()
        with pytest.raises(ValidationError):
            config.enabled = False
