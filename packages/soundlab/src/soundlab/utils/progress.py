"""Progress reporting utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tqdm import tqdm

if TYPE_CHECKING:
    pass

__all__ = [
    "TqdmProgressCallback",
    "GradioProgressCallback",
    "NullProgressCallback",
    "create_progress_callback",
]


class TqdmProgressCallback:
    """Progress callback using tqdm progress bars."""

    def __init__(
        self,
        desc: str = "Processing",
        unit: str = "it",
        leave: bool = True,
    ) -> None:
        """
        Initialize the progress callback.

        Parameters
        ----------
        desc
            Description shown on the progress bar.
        unit
            Unit name for the progress.
        leave
            Whether to leave the progress bar after completion.
        """
        self.desc = desc
        self.unit = unit
        self.leave = leave
        self._pbar: tqdm | None = None

    def __call__(self, current: int, total: int, message: str = "") -> None:
        """Update progress."""
        if self._pbar is None:
            self._pbar = tqdm(
                total=total,
                desc=self.desc,
                unit=self.unit,
                leave=self.leave,
            )

        # Update to current position
        self._pbar.n = current
        if message:
            self._pbar.set_postfix_str(message)
        self._pbar.refresh()

        # Close on completion
        if current >= total:
            self._pbar.close()
            self._pbar = None

    def close(self) -> None:
        """Manually close the progress bar."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


class GradioProgressCallback:
    """Progress callback for Gradio interfaces."""

    def __init__(self, progress: Any = None) -> None:
        """
        Initialize with a Gradio progress object.

        Parameters
        ----------
        progress
            Gradio progress object (gr.Progress).
        """
        self._progress = progress

    def __call__(self, current: int, total: int, message: str = "") -> None:
        """Update progress."""
        if self._progress is not None:
            fraction = current / total if total > 0 else 0
            self._progress(fraction, desc=message or "Processing...")


class NullProgressCallback:
    """No-op progress callback."""

    def __call__(self, current: int, total: int, message: str = "") -> None:
        """Do nothing."""
        pass


def create_progress_callback(
    backend: str = "tqdm",
    **kwargs: Any,
) -> TqdmProgressCallback | GradioProgressCallback | NullProgressCallback:
    """
    Create a progress callback.

    Parameters
    ----------
    backend
        Backend to use: "tqdm", "gradio", or "none".
    **kwargs
        Additional arguments passed to the callback constructor.

    Returns
    -------
    Progress callback instance.
    """
    if backend == "tqdm":
        return TqdmProgressCallback(**kwargs)
    elif backend == "gradio":
        return GradioProgressCallback(**kwargs)
    else:
        return NullProgressCallback()
