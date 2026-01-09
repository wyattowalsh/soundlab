"""Integration tests for voice module (TTS and SVC)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("pydantic")

if TYPE_CHECKING:
    from pathlib import Path


def has_rvc() -> bool:
    """Check if RVC is configured."""
    return os.getenv("SOUNDLAB_RVC_ROOT") is not None


class TestSVCIntegration:
    """Integration tests for SVC module."""

    def test_svc_error_without_setup(self, tmp_path: Path) -> None:
        """VoiceConverter raises error without RVC setup."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice import SVCConfig, VoiceConverter

        converter = VoiceConverter(rvc_root=None)
        config = SVCConfig()

        audio_path = tmp_path / "input.wav"
        audio_path.touch()
        model_path = tmp_path / "model.pth"
        model_path.touch()

        with pytest.raises(VoiceConversionError, match="RVC is not configured"):
            converter.convert(audio_path, model_path, config)

    def test_svc_config_integration(self) -> None:
        """SVCConfig works with VoiceConverter (no execution)."""
        from soundlab.voice import SVCConfig, VoiceConverter

        config = SVCConfig(pitch_shift=2.0, index_rate=0.3)
        converter = VoiceConverter(rvc_root=None)

        assert config.pitch_shift == 2.0
        assert config.index_rate == 0.3
        assert converter._rvc_root is None

    def test_svc_result_model(self, tmp_path: Path) -> None:
        """SVCResult model works correctly."""
        from soundlab.voice import SVCResult

        result = SVCResult(
            audio_path=tmp_path / "converted.wav",
            processing_time=2.5,
        )
        assert result.processing_time == 2.5
        assert result.audio_path == tmp_path / "converted.wav"

    def test_voice_module_exports_svc(self) -> None:
        """SVC components exported from voice module."""
        from soundlab.voice import SVCConfig, SVCResult, VoiceConverter

        assert SVCConfig is not None
        assert SVCResult is not None
        assert VoiceConverter is not None


class TestTTSIntegration:
    """Integration tests for TTS module."""

    def test_tts_config_defaults(self) -> None:
        """TTSConfig has correct defaults."""
        from soundlab.voice import TTSConfig

        config = TTSConfig(text="Hello world")

        assert config.text == "Hello world"
        assert config.language == "en"
        assert config.speaker_wav is None
        assert config.temperature == 0.7
        assert config.speed == 1.0

    def test_tts_result_model(self, tmp_path: Path) -> None:
        """TTSResult model works correctly."""
        from soundlab.voice import TTSResult

        result = TTSResult(
            audio_path=tmp_path / "output.wav",
            processing_time=1.5,
        )
        assert result.processing_time == 1.5
        assert result.audio_path == tmp_path / "output.wav"

    def test_tts_generator_creation(self) -> None:
        """TTSGenerator can be instantiated (lazy loading)."""
        from soundlab.voice import TTSGenerator

        generator = TTSGenerator()
        assert generator._tts is None

    def test_voice_module_exports_tts(self) -> None:
        """TTS components exported from voice module."""
        from soundlab.voice import TTSConfig, TTSGenerator, TTSResult

        assert TTSConfig is not None
        assert TTSGenerator is not None
        assert TTSResult is not None
