"""Demucs stem separation wrapper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from soundlab.core.exceptions import GPUMemoryError, ModelNotFoundError, ProcessingError
from soundlab.separation.models import SeparationConfig, StemResult
from soundlab.utils.gpu import get_device, get_free_vram_gb

if TYPE_CHECKING:
    from soundlab.utils.progress import ProgressCallback


class StemSeparator:
    """High-level interface for stem separation using Demucs."""

    def __init__(self, config: SeparationConfig | None = None) -> None:
        """Initialize the stem separator."""
        self.config = config or SeparationConfig()
        self._model: object | None = None
        self._device: str | None = None

    def _load_model(self) -> None:
        """Lazy-load the Demucs model."""
        if self._model is not None:
            return

        from demucs.pretrained import get_model

        self._device = get_device(self.config.device)
        logger.info(f"Loading Demucs model: {self.config.model} on {self._device}")

        try:
            model = get_model(self.config.model.value)
            model.to(self._device)
            model.eval()
            self._model = model
        except Exception as exc:
            raise ModelNotFoundError(f"Failed to load {self.config.model}: {exc}") from exc

    def _check_memory(self, duration_seconds: float) -> None:
        """Verify sufficient GPU memory for processing."""
        if self._device == "cpu":
            return

        estimated_gb = 2.0 + (duration_seconds / 60.0) * 0.5
        free_gb = get_free_vram_gb()

        if free_gb < estimated_gb:
            logger.warning(
                "Low VRAM: "
                f"{free_gb:.1f}GB free, ~{estimated_gb:.1f}GB needed. "
                "Consider enabling segment processing or using CPU."
            )
            if free_gb < 2.0:
                raise GPUMemoryError(
                    f"Insufficient VRAM ({free_gb:.1f}GB). Use config.split=True or device='cpu'."
                )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def separate(
        self,
        audio_path: Path | str,
        output_dir: Path | str,
        progress_callback: ProgressCallback | None = None,
    ) -> StemResult:
        """Separate audio into stems."""
        from demucs.apply import apply_model
        from demucs.audio import AudioFile

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._load_model()
        if self._model is None or self._device is None:
            raise ProcessingError("Demucs model failed to initialize")

        start_time = time.perf_counter()

        logger.info(f"Loading audio: {audio_path}")
        wav = AudioFile(audio_path).read(
            streams=0,
            samplerate=self._model.samplerate,
            channels=self._model.audio_channels,
        )
        wav = wav.to(self._device)

        self._check_memory(wav.shape[-1] / self._model.samplerate)

        logger.info(f"Separating with {self.config.model}...")
        with torch.no_grad():
            sources = apply_model(
                self._model,
                wav[None],
                segment=self.config.segment_length if self.config.split else None,
                overlap=self.config.overlap,
                shifts=self.config.shifts,
                progress=progress_callback is not None,
            )[0]

        stem_names = self.config.model.stems
        if self.config.two_stems:
            if self.config.two_stems not in stem_names:
                raise ProcessingError(f"Invalid two_stems target: {self.config.two_stems}")
            target_index = stem_names.index(self.config.two_stems)
            target = sources[target_index]
            remaining = torch.cat(
                [
                    sources[:target_index],
                    sources[target_index + 1 :],
                ],
                dim=0,
            )
            complement = remaining.sum(dim=0)
            sources = torch.stack([target, complement], dim=0)
            stem_names = [self.config.two_stems, f"no_{self.config.two_stems}"]

        stems: dict[str, Path] = {}
        for i, stem_name in enumerate(stem_names):
            stem_path = output_dir / f"{stem_name}.wav"
            self._save_stem(sources[i], stem_path)
            stems[stem_name] = stem_path
            logger.debug(f"Saved: {stem_path}")

        processing_time = time.perf_counter() - start_time
        logger.info(f"Separation complete in {processing_time:.1f}s")

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
            audio = audio.T

        subtype = "PCM_24" if self.config.int24 else "FLOAT" if self.config.float32 else "PCM_16"
        sf.write(path, audio, self._model.samplerate, subtype=subtype)
