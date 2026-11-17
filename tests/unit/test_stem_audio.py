"""Stem container writes honor the separation format settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf

from soundlab.core.exceptions import ConfigurationError
from soundlab.separation.models import SeparationConfig, write_stem_audio

if TYPE_CHECKING:
    from pathlib import Path


def _tone() -> np.ndarray:
    sample_rate = 44100
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    mono = (0.4 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    return np.column_stack([mono, mono * 0.5])


def test_wav_float_request_stays_float(tmp_path: Path) -> None:
    target = write_stem_audio(
        _tone(),
        tmp_path / "vocals.wav",
        sample_rate=44100,
        output_format="wav",
        float32=True,
        int24=True,
    )

    info = sf.info(target)
    assert target.suffix == ".wav"
    assert info.subtype == "FLOAT"


def test_flac_writes_24_bit_when_requested(tmp_path: Path) -> None:
    target = write_stem_audio(
        _tone(),
        tmp_path / "drums.wav",
        sample_rate=44100,
        output_format="flac",
        float32=False,
        int24=True,
    )

    info = sf.info(target)
    assert target.suffix == ".flac"
    assert info.format == "FLAC"
    assert info.subtype == "PCM_24"


def test_float_flac_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="float32"):
        write_stem_audio(
            _tone(),
            tmp_path / "bass.flac",
            sample_rate=44100,
            output_format="flac",
            float32=True,
        )


def test_mp3_uses_the_requested_bitrate(tmp_path: Path) -> None:
    low = write_stem_audio(
        _tone(),
        tmp_path / "low.mp3",
        sample_rate=44100,
        output_format="mp3",
        float32=False,
        mp3_bitrate=128,
    )
    high = write_stem_audio(
        _tone(),
        tmp_path / "high.mp3",
        sample_rate=44100,
        output_format="mp3",
        float32=False,
        mp3_bitrate=320,
    )

    assert low.suffix == ".mp3"
    assert sf.info(low).format == "MP3"
    assert high.stat().st_size > low.stat().st_size * 1.5


def test_candidate_plans_keep_the_output_format() -> None:
    from soundlab.pipeline import build_candidate_plans
    from soundlab.pipeline.models import PipelineConfig

    PipelineConfig.model_rebuild()
    base = SeparationConfig(output_format="flac", mp3_bitrate=192, float32=False)
    plans = build_candidate_plans(PipelineConfig(max_candidates=4), base=base)

    assert {plan.separation.output_format for plan in plans} == {"flac"}
    assert {plan.separation.mp3_bitrate for plan in plans} == {192}
