"""Retry helpers built on tenacity."""

from __future__ import annotations

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in minimal envs
    torch = None  # type: ignore[assignment]


class _FallbackOOM(RuntimeError):
    """Fallback OOM error when torch is unavailable."""


def _clear_cuda_cache(_: RetryCallState) -> None:
    if torch is None or not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()


_oom_error = torch.cuda.OutOfMemoryError if torch is not None else _FallbackOOM

io_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((IOError, ConnectionError)),
)

gpu_retry = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type(_oom_error),
    before_sleep=_clear_cuda_cache,
)

network_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.RequestError, ConnectionError, TimeoutError)),
)

__all__ = ["gpu_retry", "io_retry", "network_retry"]
