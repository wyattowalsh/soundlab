"""Checks that effect controls change the audio they claim to change."""

from __future__ import annotations

import numpy as np
import pytest

from soundlab.core.exceptions import ConfigurationError
from soundlab.effects import EffectsChain, GainConfig, HighpassConfig, LowpassConfig, ReverbConfig


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples[len(samples) // 2 :]))))


def _tone(frequency: float, sample_rate: int = 48000, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return np.sin(2.0 * np.pi * frequency * time).astype(np.float32)


def test_reverb_plugin_keeps_wet_and_dry() -> None:
    plugin = ReverbConfig(wet=0.2, dry=0.8, room_size=0.4).to_plugin()

    assert plugin.wet_level == pytest.approx(0.2, abs=1e-4)
    assert plugin.dry_level == pytest.approx(0.8, abs=1e-4)
    assert plugin.room_size == pytest.approx(0.4, abs=1e-4)


def test_highpass_attenuates_below_cutoff_and_keeps_the_top() -> None:
    sample_rate = 48000
    chain = EffectsChain().add(HighpassConfig(cutoff_frequency_hz=1000.0, q=0.707))

    low = chain.process_array(_tone(100.0, sample_rate), sample_rate)
    high = chain.process_array(_tone(4000.0, sample_rate), sample_rate)

    assert _rms(low) < 0.05
    assert _rms(high) > 0.5


def test_highpass_q_changes_the_response_at_cutoff() -> None:
    sample_rate = 48000
    tone = _tone(1000.0, sample_rate)
    mild = EffectsChain().add(HighpassConfig(cutoff_frequency_hz=1000.0, q=0.707))
    peaked = EffectsChain().add(HighpassConfig(cutoff_frequency_hz=1000.0, q=8.0))
    mild_level = _rms(mild.process_array(tone, sample_rate))
    peaked_level = _rms(peaked.process_array(tone, sample_rate))

    assert peaked_level > mild_level * 2


def test_lowpass_attenuates_above_cutoff() -> None:
    sample_rate = 48000
    chain = EffectsChain().add(LowpassConfig(cutoff_frequency_hz=500.0, q=0.707))

    low = chain.process_array(_tone(100.0, sample_rate), sample_rate)
    high = chain.process_array(_tone(4000.0, sample_rate), sample_rate)

    assert _rms(low) > 0.5
    assert _rms(high) < 0.05


def test_direct_plugin_refuses_a_q_pedalboard_cannot_store() -> None:
    with pytest.raises(ConfigurationError, match="no Q"):
        HighpassConfig(cutoff_frequency_hz=120.0, q=2.0).to_plugin()


def test_filter_cutoff_above_nyquist_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="cutoff"):
        HighpassConfig(cutoff_frequency_hz=30000.0).process_array(_tone(100.0), 48000)


def test_chain_orders_a_filter_around_a_plugin() -> None:
    sample_rate = 48000
    tone = _tone(100.0, sample_rate)
    chain = (
        EffectsChain().add(GainConfig(gain_db=6.0)).add(HighpassConfig(cutoff_frequency_hz=2000.0))
    )

    processed = chain.process_array(tone, sample_rate)

    assert processed.shape == tone.shape
    assert _rms(processed) < 0.05
