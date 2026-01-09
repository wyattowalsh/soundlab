"""Singing voice conversion wrapper for RVC."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from soundlab.core.exceptions import VoiceConversionError

if TYPE_CHECKING:
    from soundlab.voice.models import SVCConfig, SVCResult


class VoiceConverter:
    """Convert vocals using an external RVC setup.

    RVC is not bundled with SoundLab. Install it manually and set
    `SOUNDLAB_RVC_ROOT` (or pass `rvc_root`) to the RVC repository path.
    The repo should include the inference scripts and model weights.
    """

    def __init__(self, rvc_root: str | Path | None = None) -> None:
        env_root = os.getenv("SOUNDLAB_RVC_ROOT")
        root_value = rvc_root or env_root
        self._rvc_root = Path(root_value) if root_value else None

    def convert(
        self,
        audio_path: str | Path,
        model_path: str | Path,
        config: SVCConfig,
    ) -> SVCResult:
        """Convert audio to a target voice using RVC."""
        _ = config
        source = Path(audio_path)
        model = Path(model_path)
        output_root = Path(os.getenv("SOUNDLAB_OUTPUT_DIR", "outputs")) / "voice"
        output_root.mkdir(parents=True, exist_ok=True)

        if self._rvc_root is None:
            raise VoiceConversionError(
                "RVC is not configured. Install RVC and set SOUNDLAB_RVC_ROOT "
                "to the repository path, then provide a valid model file."
            )

        if not source.exists():
            raise VoiceConversionError(f"Input audio not found: {source}")
        if not model.exists():
            raise VoiceConversionError(f"RVC model not found: {model}")

        raise VoiceConversionError(
            "RVC conversion requires manual setup of the inference pipeline. "
            "See the RVC repository docs for configuration steps."
        )
