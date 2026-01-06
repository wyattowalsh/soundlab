"""Demucs stem separation wrapper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

from soundlab.core.exceptions import GPUMemoryError, ModelNotFoundError, ProcessingError
from soundlab.separation.models import DemucsModel, SeparationConfig, StemResult
from soundlab.separation.utils import estimate_memory_usage
from soundlab.utils.gpu import clear_gpu_cache, get_device, get_free_vram_gb
from soundlab.utils.retry import model_retry

if TYPE_CHECKING:
    from soundlab.core.types import PathLike, ProgressCallback

__all__ = ["StemSeparator"]


class StemSeparator:
    """High-level interface for stem separation using Demucs."""

    def __init__(self, config: SeparationConfig | None = None) -> None:
        self.config = config or SeparationConfig()
        self._model = None
        self._device = None

    def _load_model(self) -> None:
        """Lazy-load the Demucs model."""
        if self._model is not None:
            return

        from demucs.pretrained import get_model

        self._device = get_device(self.config.device)
        logger.info(f"Loading Demucs model: {self.config.model} on {self._device}")

        try:
            self._model = get_model(self.config.model.value)
            self._model.to(self._device)
            self._model.eval()
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load {self.config.model}: {e}") from e

    def _check_memory(self, duration_seconds: float) -> None:
        """Verify sufficient GPU memory for processing."""
        if self._device == "cpu":
            return

        estimates = estimate_memory_usage(
            duration_seconds,
            model_name=self.config.model.value,
        )

        free_gb = get_free_vram_gb()

        if free_gb < estimates["total_gb"]:
            logger.warning(
                f"Low VRAM: {free_gb:.1f}GB free, ~{estimates['total_gb']:.1f}GB needed. "
                "Consider enabling segment processing or using CPU."
            )
            if free_gb < 2.0:
                raise GPUMemoryError(
                    f"Insufficient VRAM ({free_gb:.1f}GB). "
                    "Use config.split=True or device='cpu'."
                )

    @model_retry
    def separate(
        self,
        audio_path: PathLike,
        output_dir: PathLike,
        progress_callback: ProgressCallback | None = None,
    ) -> StemResult:
        """
        Separate audio into stems.

        Parameters
        ----------
        audio_path
            Path to input audio file.
        output_dir
            Directory to save separated stems.
        progress_callback
            Optional callback for progress updates.

        Returns
        -------
        StemResult
            Paths to separated stem files and metadata.
        """
        from demucs.apply import apply_model
        from demucs.audio import AudioFile
        import soundfile as sf

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._load_model()

        start_time = time.perf_counter()

        # Load audio
        logger.info(f"Loading audio: {audio_path}")
        wav = AudioFile(audio_path).read(
            streams=0,
            samplerate=self._model.samplerate,
            channels=self._model.audio_channels,
        )
        wav = wav.to(self._device)

        duration = wav.shape[-1] / self._model.samplerate
        self._check_memory(duration)

        if progress_callback:
            progress_callback(0, 100, "Separating stems...")

        # Apply model
        logger.info(f"Separating with {self.config.model}...")

        with torch.no_grad():
            sources = apply_model(
                self._model,
                wav[None],
                segment=self.config.segment_length if self.config.split else None,
                overlap=self.config.overlap,
                shifts=self.config.shifts,
                progress=True,
            )[0]

        if progress_callback:
            progress_callback(80, 100, "Saving stems...")

        # Save stems
        stems = {}
        stem_names = self.config.model.stems

        if self.config.two_stems:
            # Find index of target stem
            target_idx = stem_names.index(self.config.two_stems)
            other_sources = torch.stack([s for i, s in enumerate(sources) if i != target_idx])
            other_combined = other_sources.sum(dim=0)

            stem_names = [self.config.two_stems, f"no_{self.config.two_stems}"]
            sources = torch.stack([sources[target_idx], other_combined])

        for i, stem_name in enumerate(stem_names):
            stem_path = output_dir / f"{stem_name}.wav"
            self._save_stem(sources[i], stem_path)
            stems[stem_name] = stem_path
            logger.debug(f"Saved: {stem_path}")

        processing_time = time.perf_counter() - start_time

        if progress_callback:
            progress_callback(100, 100, "Complete")

        logger.info(f"Separation complete in {processing_time:.1f}s")

        # Clear GPU memory
        clear_gpu_cache()

        return StemResult(
            stems=stems,
            source_path=audio_path,
            config=self.config,
            processing_time_seconds=processing_time,
        )

    def _save_stem(self, tensor: torch.Tensor, path: Path) -> None:
        """Save a stem tensor to file."""
        import soundfile as sf

        audio = tensor.cpu().numpy()
        if audio.ndim == 2:
            audio = audio.T  # (channels, samples) -> (samples, channels)

        subtype = "PCM_24" if self.config.int24 else "FLOAT" if self.config.float32 else "PCM_16"
        sf.write(path, audio, self._model.samplerate, subtype=subtype)

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            clear_gpu_cache()
            logger.info("Model unloaded")
