"""Text-to-speech generation using XTTS-v2."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from soundlab.core.exceptions import ProcessingError, VoiceConversionError
from soundlab.utils.gpu import clear_gpu_cache, get_device
from soundlab.utils.retry import model_retry
from soundlab.voice.models import TTSConfig, TTSLanguage, TTSResult

if TYPE_CHECKING:
    from soundlab.core.types import PathLike, ProgressCallback


__all__ = ["TTSGenerator"]


class TTSGenerator:
    """
    Text-to-speech generator using Coqui XTTS-v2.

    Examples
    --------
    >>> from soundlab.voice import TTSGenerator, TTSConfig
    >>>
    >>> generator = TTSGenerator()
    >>> config = TTSConfig(
    ...     text="Hello, this is a test.",
    ...     language="en",
    ...     speaker_wav="reference.wav",
    ... )
    >>> result = generator.generate(config, "output.wav")
    >>> print(result.audio_path)
    """

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2") -> None:
        """
        Initialize the TTS generator.

        Parameters
        ----------
        model_name
            TTS model to use. Default is XTTS-v2.
        """
        self.model_name = model_name
        self._tts = None
        self._device = None

    def _load_model(self, device: str = "auto") -> None:
        """Lazy-load the TTS model."""
        if self._tts is not None:
            return

        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError(
                "coqui-tts is required for TTS. "
                "Install with: pip install coqui-tts"
            )

        self._device = get_device(device)
        logger.info(f"Loading TTS model: {self.model_name} on {self._device}")

        try:
            self._tts = TTS(model_name=self.model_name).to(self._device)
            logger.info("TTS model loaded successfully")
        except Exception as e:
            raise VoiceConversionError(f"Failed to load TTS model: {e}") from e

    @model_retry
    def generate(
        self,
        config: TTSConfig,
        output_path: PathLike,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> TTSResult:
        """
        Generate speech from text.

        Parameters
        ----------
        config
            TTS configuration including text, language, and optional speaker reference.
        output_path
            Path to save generated audio.
        progress_callback
            Optional callback for progress updates.

        Returns
        -------
        TTSResult
            Generation result with audio path and metadata.

        Raises
        ------
        VoiceConversionError
            If generation fails.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._load_model(config.device)

        start_time = time.perf_counter()

        if progress_callback:
            progress_callback(0, 100, "Generating speech...")

        logger.info(f"Generating TTS: {len(config.text)} chars, language={config.language}")

        try:
            # Prepare speaker reference if provided
            speaker_wav = None
            if config.speaker_wav is not None:
                speaker_wav = str(config.speaker_wav)
                if not Path(speaker_wav).exists():
                    raise VoiceConversionError(f"Speaker reference file not found: {speaker_wav}")

            # Generate speech
            self._tts.tts_to_file(
                text=config.text,
                file_path=str(output_path),
                speaker_wav=speaker_wav,
                language=config.language.value,
                split_sentences=True,
            )

            if progress_callback:
                progress_callback(90, 100, "Finalizing...")

            # Get duration of generated audio
            import soundfile as sf
            info = sf.info(output_path)
            duration = info.duration

            processing_time = time.perf_counter() - start_time

            if progress_callback:
                progress_callback(100, 100, "Complete")

            logger.info(f"TTS complete: {duration:.1f}s audio in {processing_time:.1f}s")

            return TTSResult(
                audio_path=output_path,
                text=config.text,
                language=config.language,
                duration_seconds=duration,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            raise VoiceConversionError(f"TTS generation failed: {e}") from e

    def generate_stream(
        self,
        config: TTSConfig,
    ):
        """
        Generate speech as a streaming iterator (for real-time playback).

        Parameters
        ----------
        config
            TTS configuration.

        Yields
        ------
        numpy.ndarray
            Audio chunks as they are generated.

        Note
        ----
        This is an experimental feature and may not work with all models.
        """
        self._load_model(config.device)

        try:
            speaker_wav = str(config.speaker_wav) if config.speaker_wav else None

            # Use streaming inference if available
            for chunk in self._tts.tts_stream(
                text=config.text,
                speaker_wav=speaker_wav,
                language=config.language.value,
            ):
                yield chunk
        except AttributeError:
            logger.warning("Streaming not supported for this model, falling back to batch")
            # Fallback: generate full audio and yield as single chunk
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                result = self.generate(config, f.name)
                import soundfile as sf
                audio, sr = sf.read(result.audio_path)
                yield audio

    def list_languages(self) -> list[str]:
        """Get list of supported languages."""
        return [lang.value for lang in TTSLanguage]

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._tts is not None:
            del self._tts
            self._tts = None
            clear_gpu_cache()
            logger.info("TTS model unloaded")
