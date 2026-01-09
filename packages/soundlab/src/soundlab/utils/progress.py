"""Progress callback adapters."""

from __future__ import annotations

import importlib
from typing import Protocol


class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""

    def __call__(self, progress: float, message: str | None = None) -> None:
        """Report progress as a float from 0.0 to 1.0 with an optional message."""


def _clamp_progress(progress: float) -> float:
    if progress < 0:
        return 0.0
    if progress > 1:
        return 1.0
    return float(progress)


class TqdmProgressCallback:
    """Adapter for tqdm progress bars."""

    def __init__(self, total: int = 100, description: str | None = None) -> None:
        try:
            tqdm_module = importlib.import_module("tqdm")
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("tqdm is required for TqdmProgressCallback") from exc

        self._total = max(total, 1)
        self._current = 0
        self._closed = False
        self._bar = tqdm_module.tqdm(total=self._total)
        if description:
            self._bar.set_description_str(description)

    def __call__(self, progress: float, message: str | None = None) -> None:
        if self._closed:
            return

        clamped = _clamp_progress(progress)
        target = round(clamped * self._total)
        delta = target - self._current
        if delta > 0:
            self._bar.update(delta)
            self._current = target

        if message:
            self._bar.set_description_str(message)

        if clamped >= 1.0:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._bar.close()
        self._closed = True


class GradioProgressCallback:
    """Adapter for gradio.Progress objects."""

    def __init__(self, progress: object | None = None) -> None:
        if progress is None:
            try:
                gr = importlib.import_module("gradio")
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError("gradio is required for GradioProgressCallback") from exc

            progress = gr.Progress()

        self._progress = progress

    def __call__(self, progress: float, message: str | None = None) -> None:
        clamped = _clamp_progress(progress)
        if message:
            try:
                self._progress(clamped, desc=message)
                return
            except TypeError:
                pass
        self._progress(clamped)


__all__ = ["GradioProgressCallback", "ProgressCallback", "TqdmProgressCallback"]
