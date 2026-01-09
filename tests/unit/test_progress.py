"""Progress callback tests - Gradio adapter and helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest  # noqa: I001


# ---------------------------------------------------------------------------
# TestClampProgress
# ---------------------------------------------------------------------------


class TestClampProgress:
    """Tests for _clamp_progress helper."""

    def test_clamp_below_zero_returns_zero(self) -> None:
        """Values < 0 clamped to 0."""
        from soundlab.utils.progress import _clamp_progress

        assert _clamp_progress(-0.5) == 0.0
        assert _clamp_progress(-100) == 0.0

    def test_clamp_above_one_returns_one(self) -> None:
        """Values > 1 clamped to 1."""
        from soundlab.utils.progress import _clamp_progress

        assert _clamp_progress(1.5) == 1.0
        assert _clamp_progress(100) == 1.0

    def test_clamp_in_range_returns_value(self) -> None:
        """Values in [0, 1] returned unchanged."""
        from soundlab.utils.progress import _clamp_progress

        assert _clamp_progress(0.5) == 0.5
        assert _clamp_progress(0.75) == 0.75

    def test_clamp_boundary_zero(self) -> None:
        """0.0 returned as-is."""
        from soundlab.utils.progress import _clamp_progress

        assert _clamp_progress(0.0) == 0.0

    def test_clamp_boundary_one(self) -> None:
        """1.0 returned as-is."""
        from soundlab.utils.progress import _clamp_progress

        assert _clamp_progress(1.0) == 1.0


# ---------------------------------------------------------------------------
# TestGradioProgressCallback
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_gradio():
    """Mock gradio module."""
    mock_progress_instance = MagicMock()
    mock_gradio_module = MagicMock()
    mock_gradio_module.Progress.return_value = mock_progress_instance

    with patch(
        "soundlab.utils.progress.importlib.import_module",
        return_value=mock_gradio_module,
    ):
        yield mock_progress_instance, mock_gradio_module


class TestGradioProgressCallback:
    """Tests for GradioProgressCallback."""

    def test_init_creates_progress_when_none(self, mock_gradio) -> None:
        """Creates Progress when None passed."""
        _mock_progress_instance, mock_module = mock_gradio
        from soundlab.utils.progress import GradioProgressCallback

        GradioProgressCallback()
        mock_module.Progress.assert_called_once()

    def test_init_wraps_existing_progress(self, mock_gradio) -> None:
        """Uses provided progress object."""
        _mock_progress_instance, mock_module = mock_gradio
        from soundlab.utils.progress import GradioProgressCallback

        existing = MagicMock()
        callback = GradioProgressCallback(progress=existing)

        # Should not create new Progress
        mock_module.Progress.assert_not_called()
        assert callback._progress is existing

    def test_init_import_error_when_missing(self) -> None:
        """Raises ImportError when gradio not installed."""
        with patch(
            "soundlab.utils.progress.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'gradio'"),
        ):
            from soundlab.utils.progress import GradioProgressCallback

            with pytest.raises(ImportError, match="gradio is required"):
                GradioProgressCallback()

    def test_call_without_message(self, mock_gradio) -> None:
        """Calls progress(value) without message."""
        mock_progress_instance, _mock_module = mock_gradio
        from soundlab.utils.progress import GradioProgressCallback

        callback = GradioProgressCallback()
        callback(0.5)

        mock_progress_instance.assert_called_once_with(0.5)

    def test_call_with_message(self, mock_gradio) -> None:
        """Calls progress(value, desc=message)."""
        mock_progress_instance, _mock_module = mock_gradio
        from soundlab.utils.progress import GradioProgressCallback

        callback = GradioProgressCallback()
        callback(0.75, message="Processing...")

        mock_progress_instance.assert_called_once_with(0.75, desc="Processing...")

    def test_call_fallback_on_typeerror(self, mock_gradio) -> None:
        """Falls back to no-desc on TypeError."""
        mock_progress_instance, _mock_module = mock_gradio
        from soundlab.utils.progress import GradioProgressCallback

        # First call with desc raises TypeError, fallback should call without desc
        def side_effect_fn(value, desc=None):
            if desc is not None:
                raise TypeError("unexpected keyword argument 'desc'")

        mock_progress_instance.side_effect = side_effect_fn

        callback = GradioProgressCallback()
        callback(0.6, message="Test")

        # Should have been called twice: once with desc (failed), once without
        assert mock_progress_instance.call_count == 2
        mock_progress_instance.assert_any_call(0.6, desc="Test")
        mock_progress_instance.assert_any_call(0.6)

    def test_call_clamps_progress(self, mock_gradio) -> None:
        """Progress clamped before calling."""
        mock_progress_instance, _mock_module = mock_gradio
        from soundlab.utils.progress import GradioProgressCallback

        callback = GradioProgressCallback()

        callback(1.5)  # Above 1
        mock_progress_instance.assert_called_with(1.0)

        mock_progress_instance.reset_mock()

        callback(-0.5)  # Below 0
        mock_progress_instance.assert_called_with(0.0)
