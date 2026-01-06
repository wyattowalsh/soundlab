"""Global configuration management for SoundLab."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


__all__ = [
    "SoundLabConfig",
    "get_config",
    "reset_config",
]


class SoundLabConfig(BaseModel):
    """Global configuration for SoundLab."""

    # Logging
    log_level: str = Field(default="INFO")

    # GPU/Processing
    gpu_mode: str = Field(default="auto")  # "auto", "cuda", "cpu"

    # Directories
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "soundlab")
    output_dir: Path = Field(default_factory=lambda: Path.cwd() / "outputs")

    @classmethod
    def from_env(cls) -> "SoundLabConfig":
        """Create config from environment variables."""
        return cls(
            log_level=os.getenv("SOUNDLAB_LOG_LEVEL", "INFO"),
            gpu_mode=os.getenv("SOUNDLAB_GPU_MODE", "auto"),
            cache_dir=Path(os.getenv("SOUNDLAB_CACHE_DIR", str(Path.home() / ".cache" / "soundlab"))),
            output_dir=Path(os.getenv("SOUNDLAB_OUTPUT_DIR", str(Path.cwd() / "outputs"))),
        )


@lru_cache(maxsize=1)
def get_config() -> SoundLabConfig:
    """Get the global configuration (singleton)."""
    return SoundLabConfig.from_env()


def reset_config() -> None:
    """Reset the cached configuration (for testing)."""
    get_config.cache_clear()
