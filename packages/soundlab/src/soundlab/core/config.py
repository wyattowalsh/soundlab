"""Global configuration management for SoundLab."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class SoundLabConfig:
    """Singleton configuration loaded from environment variables."""

    log_level: str
    gpu_mode: str
    cache_dir: Path
    output_dir: Path

    _instance: ClassVar[SoundLabConfig | None] = None

    @classmethod
    def from_env(cls) -> SoundLabConfig:
        """Create a configuration instance from environment variables."""
        log_level = os.getenv("SOUNDLAB_LOG_LEVEL", "INFO")
        gpu_mode = os.getenv("SOUNDLAB_GPU_MODE", "auto")
        cache_dir = Path(os.getenv("SOUNDLAB_CACHE_DIR", "~/.cache/soundlab")).expanduser()
        output_dir = Path(os.getenv("SOUNDLAB_OUTPUT_DIR", "./outputs")).expanduser()
        return cls(
            log_level=log_level,
            gpu_mode=gpu_mode,
            cache_dir=cache_dir,
            output_dir=output_dir,
        )

    @classmethod
    def load(cls) -> SoundLabConfig:
        """Return the cached configuration, loading it if needed."""
        if cls._instance is None:
            cls._instance = cls.from_env()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the cached configuration instance."""
        cls._instance = None
