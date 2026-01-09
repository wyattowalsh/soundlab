"""Text-to-speech wrapper for XTTS-v2."""

from __future__ import annotations

import os
import time
from pathlib import Path

from loguru import logger

from soundlab.voice.models import TTSConfig, TTSResult


class TTSGenerator:
    """Generate speech audio using Coqui XTTS-v2."""

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2") -> None:
        self._model_name = model_name
        self._tts = None

    def _load_model(self) -> None:
        if self._tts is not None:
            return

        try:
            from TTS.api import TTS
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "coqui-tts is required for TTSGenerator. Install soundlab[voice]."
            ) from exc

        cache_root = os.getenv("SOUNDLAB_CACHE_DIR")
        if cache_root:
            os.environ.setdefault("TTS_HOME", str(Path(cache_root) / "tts"))

        logger.info(f"Loading XTTS model: {self._model_name}")
        try:
            self._tts = TTS(model_name=self._model_name, progress_bar=False)
        except TypeError:
            self._tts = TTS(model_name=self._model_name)

    def generate(self, config: TTSConfig) -> TTSResult:
        """Generate speech audio for the provided configuration."""
        self._load_model()
        if self._tts is None:
            raise RuntimeError("TTS model failed to initialize")

        output_root = Path(os.getenv("SOUNDLAB_OUTPUT_DIR", "outputs")) / "voice"
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"tts_{int(time.time() * 1000)}.wav"

        speaker_wav = str(config.speaker_wav) if config.speaker_wav else None
        kwargs = {
            "text": config.text,
            "file_path": str(output_path),
            "language": config.language,
            "speaker_wav": speaker_wav,
            "temperature": config.temperature,
            "speed": config.speed,
        }

        start = time.perf_counter()
        try:
            self._tts.tts_to_file(**{k: v for k, v in kwargs.items() if v is not None})
        except TypeError:
            fallback = {
                "text": config.text,
                "file_path": str(output_path),
            }
            if config.language:
                fallback["language"] = config.language
            if speaker_wav:
                fallback["speaker_wav"] = speaker_wav
            self._tts.tts_to_file(**fallback)

        processing_time = time.perf_counter() - start
        logger.info(f"TTS completed in {processing_time:.1f}s: {output_path}")
        return TTSResult(audio_path=output_path, processing_time=processing_time)
