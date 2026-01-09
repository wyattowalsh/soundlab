"""GPU utilities for SoundLab."""

from __future__ import annotations

import os

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in minimal envs
    torch = None  # type: ignore[assignment]


def is_cuda_available() -> bool:
    """Return True if CUDA is available via torch."""
    return bool(torch and torch.cuda.is_available())


def get_device(mode: str) -> str:
    """Resolve the device string based on mode and environment overrides."""
    env_mode = os.getenv("SOUNDLAB_GPU_MODE")
    raw_mode = env_mode if env_mode is not None else mode
    normalized = (raw_mode or "auto").strip().lower()

    if normalized in {"cpu", "force_cpu"}:
        return "cpu"

    if normalized in {"cuda", "gpu", "force_gpu"}:
        return "cuda" if is_cuda_available() else "cpu"

    if normalized in {"auto", ""}:
        return "cuda" if is_cuda_available() else "cpu"

    return "cpu"


def get_free_vram_gb() -> float:
    """Return estimated free VRAM in gigabytes."""
    if not is_cuda_available() or torch is None:
        return 0.0

    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return 0.0

    return float(free_bytes) / (1024**3)


def clear_gpu_cache() -> None:
    """Clear the CUDA cache if available."""
    if not is_cuda_available() or torch is None:
        return

    torch.cuda.empty_cache()
