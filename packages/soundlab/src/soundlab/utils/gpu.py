"""GPU detection and memory management utilities."""

from __future__ import annotations

from loguru import logger

__all__ = [
    "is_cuda_available",
    "get_device",
    "get_free_vram_gb",
    "clear_gpu_cache",
    "get_gpu_info",
]


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device(mode: str = "auto") -> str:
    """
    Get the appropriate device for computation.

    Parameters
    ----------
    mode
        Device selection mode: "auto", "cuda", "cpu"

    Returns
    -------
    str
        Device string ("cuda" or "cpu")
    """
    if mode == "cpu":
        return "cpu"

    if mode == "cuda" or mode == "auto":
        if is_cuda_available():
            logger.debug("Using CUDA device")
            return "cuda"
        elif mode == "cuda":
            logger.warning("CUDA requested but not available, falling back to CPU")

    logger.debug("Using CPU device")
    return "cpu"


def get_free_vram_gb() -> float:
    """
    Get free GPU VRAM in gigabytes.

    Returns
    -------
    float
        Free VRAM in GB, or 0.0 if no GPU available.
    """
    if not is_cuda_available():
        return 0.0

    try:
        import torch
        device = torch.cuda.current_device()
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        return free_memory / (1024 ** 3)
    except Exception as e:
        logger.warning(f"Failed to get VRAM info: {e}")
        return 0.0


def clear_gpu_cache() -> None:
    """Clear GPU memory cache."""
    if not is_cuda_available():
        return

    try:
        import torch
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear GPU cache: {e}")


def get_gpu_info() -> dict[str, str | float | None]:
    """
    Get GPU information.

    Returns
    -------
    dict
        Dictionary with GPU name, total memory, free memory.
    """
    if not is_cuda_available():
        return {"available": False, "name": None, "total_gb": None, "free_gb": None}

    try:
        import torch
        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        return {
            "available": True,
            "name": name,
            "total_gb": total_memory / (1024 ** 3),
            "free_gb": free_memory / (1024 ** 3),
        }
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
        return {"available": False, "name": None, "total_gb": None, "free_gb": None}
