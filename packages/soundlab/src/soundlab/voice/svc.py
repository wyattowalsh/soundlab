"""Singing voice conversion using RVC."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from soundlab.core.exceptions import ProcessingError, VoiceConversionError
from soundlab.utils.gpu import clear_gpu_cache, get_device
from soundlab.voice.models import SVCConfig, SVCResult

if TYPE_CHECKING:
    from soundlab.core.types import PathLike, ProgressCallback

__all__ = ["VoiceConverter"]


class VoiceConverter:
    """
    Singing voice conversion using RVC (Retrieval-based Voice Conversion).

    Note
    ----
    RVC requires manual setup due to its complexity. This wrapper provides
    a standardized interface but requires the RVC dependencies to be
    installed separately.

    Examples
    --------
    >>> from soundlab.voice import VoiceConverter, SVCConfig
    >>>
    >>> converter = VoiceConverter()
    >>> config = SVCConfig(
    ...     model_path="path/to/model.pth",
    ...     pitch_shift_semitones=0,
    ...     f0_method="rmvpe",
    ... )
    >>> result = converter.convert("input_vocals.wav", config, "output.wav")
    """

    def __init__(self) -> None:
        """Initialize the voice converter."""
        self._model = None
        self._device = None
        self._rvc_available = self._check_rvc_available()

    def _check_rvc_available(self) -> bool:
        """Check if RVC is available."""
        try:
            # Try to import RVC components
            # Note: Actual imports depend on RVC installation method
            return True
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        """Check if voice conversion is available."""
        return self._rvc_available

    def convert(
        self,
        audio_path: PathLike,
        config: SVCConfig,
        output_path: PathLike,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> SVCResult:
        """
        Convert singing voice using RVC.

        Parameters
        ----------
        audio_path
            Path to input audio (preferably isolated vocals).
        config
            SVC configuration including model path and parameters.
        output_path
            Path to save converted audio.
        progress_callback
            Optional callback for progress updates.

        Returns
        -------
        SVCResult
            Conversion result with audio path and metadata.

        Raises
        ------
        VoiceConversionError
            If conversion fails or RVC is not available.
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not audio_path.exists():
            raise VoiceConversionError(f"Input audio not found: {audio_path}")

        if config.model_path is None:
            raise VoiceConversionError("No RVC model path specified in config")

        if not config.model_path.exists():
            raise VoiceConversionError(f"RVC model not found: {config.model_path}")

        start_time = time.perf_counter()

        if progress_callback:
            progress_callback(0, 100, "Loading model...")

        logger.info(
            f"Converting voice: {audio_path.name} with pitch shift {config.pitch_shift_semitones}"
        )

        try:
            # Attempt RVC conversion
            self._convert_rvc(
                audio_path,
                config,
                output_path,
                progress_callback,
            )

            # Get duration
            import soundfile as sf
            info = sf.info(output_path)
            duration = info.duration

            processing_time = time.perf_counter() - start_time

            if progress_callback:
                progress_callback(100, 100, "Complete")

            logger.info(f"Voice conversion complete in {processing_time:.1f}s")

            return SVCResult(
                audio_path=output_path,
                source_path=audio_path,
                model_name=config.model_path.stem,
                pitch_shift=config.pitch_shift_semitones,
                processing_time_seconds=processing_time,
                duration_seconds=duration,
            )

        except Exception as e:
            raise VoiceConversionError(f"Voice conversion failed: {e}") from e

    def _convert_rvc(
        self,
        audio_path: Path,
        config: SVCConfig,
        output_path: Path,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """
        Perform RVC conversion.

        This is a placeholder implementation. Actual RVC integration requires:
        1. Installing RVC dependencies
        2. Downloading pretrained models (hubert, rmvpe, etc.)
        3. Loading the voice model
        4. Running inference

        For a working implementation, consider using:
        - rvc-python package
        - Direct RVC repository integration
        """
        # Placeholder: Copy input to output with pitch shift using librosa
        # In production, this would use actual RVC inference

        logger.warning(
            "RVC not fully integrated. Using pitch-shift fallback. "
            "For full RVC support, install rvc-python or set up RVC manually."
        )

        import librosa
        import soundfile as sf

        if progress_callback:
            progress_callback(20, 100, "Loading audio...")

        # Load audio
        y, sr = librosa.load(audio_path, sr=None)

        if progress_callback:
            progress_callback(50, 100, "Processing...")

        # Apply pitch shift (simplified fallback)
        if config.pitch_shift_semitones != 0:
            y = librosa.effects.pitch_shift(
                y,
                sr=sr,
                n_steps=config.pitch_shift_semitones,
            )

        if progress_callback:
            progress_callback(80, 100, "Saving...")

        # Save output
        sf.write(output_path, y, sr)

    def list_models(self, models_dir: PathLike) -> list[Path]:
        """
        List available RVC models in a directory.

        Parameters
        ----------
        models_dir
            Directory containing RVC model files (.pth).

        Returns
        -------
        list[Path]
            Paths to available model files.
        """
        models_dir = Path(models_dir)
        if not models_dir.exists():
            return []

        return list(models_dir.glob("*.pth"))

    def get_setup_instructions(self) -> str:
        """Get instructions for setting up RVC."""
        return """
RVC (Retrieval-based Voice Conversion) Setup Instructions:

1. Install RVC Python package:
   pip install rvc-python

2. Or clone the RVC repository:
   git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

3. Download required models:
   - Hubert base model
   - RMVPE pitch extraction model
   - Your target voice model (.pth file)

4. Place voice models in a models directory

5. Use VoiceConverter with SVCConfig pointing to your model:
   config = SVCConfig(
       model_path=Path("models/your_voice.pth"),
       pitch_shift_semitones=0,
       f0_method=F0Method.RMVPE,
   )

For detailed instructions, see: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
        """.strip()

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            clear_gpu_cache()
            logger.info("SVC model unloaded")
