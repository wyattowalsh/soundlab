"""Retry decorators using tenacity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    pass


__all__ = [
    "io_retry",
    "gpu_retry",
    "network_retry",
    "model_retry",
]


def _log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempts."""
    logger.warning(
        f"Retrying {retry_state.fn.__name__} "
        f"(attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}"
    )


def _clear_gpu_before_retry(retry_state: RetryCallState) -> None:
    """Clear GPU cache before retry."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared GPU cache before retry")
    except ImportError:
        pass
    _log_retry(retry_state)


# Standard retry for I/O operations
io_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((IOError, OSError, ConnectionError)),
    before_sleep=_log_retry,
    reraise=True,
)


# GPU retry with memory clearing
gpu_retry = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    before_sleep=_clear_gpu_before_retry,
    reraise=True,
)


# Network retry for HTTP operations
network_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=_log_retry,
    reraise=True,
)


# Model loading retry
model_retry = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    before_sleep=_clear_gpu_before_retry,
    reraise=True,
)
