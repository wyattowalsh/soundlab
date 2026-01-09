"""Visualization module tests with mocked matplotlib."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pydantic")


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib for visualization tests."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_fig.savefig = MagicMock()
    mock_fig.tight_layout = MagicMock()

    mock_cmap = MagicMock(return_value=(1, 0, 0, 1))
    mock_colormaps = MagicMock()
    mock_colormaps.__getitem__ = MagicMock(return_value=mock_cmap)

    with (
        patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)) as mock_subplots,
        patch("matplotlib.pyplot.close") as mock_close,
        patch("soundlab.transcription.visualization.mpl.colormaps", mock_colormaps),
    ):
        yield {
            "fig": mock_fig,
            "ax": mock_ax,
            "subplots": mock_subplots,
            "close": mock_close,
            "cmap": mock_cmap,
            "colormaps": mock_colormaps,
        }


@pytest.fixture
def sample_notes():
    """Sample NoteEvent list for testing."""
    from soundlab.transcription.models import NoteEvent

    return [
        NoteEvent(pitch=60, start=0.0, end=1.0, velocity=100),
        NoteEvent(pitch=64, start=0.5, end=1.5, velocity=80),
        NoteEvent(pitch=67, start=1.0, end=2.0, velocity=90),
    ]


class TestRenderPianoRoll:
    """Tests for render_piano_roll function."""

    def test_empty_notes_list(self, mock_matplotlib, tmp_path):
        """Handles empty note list (ylim 0-127)."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "empty.png"
        render_piano_roll([], output)

        mock_matplotlib["ax"].set_ylim.assert_called_once_with(0, 127)

    def test_single_note(self, mock_matplotlib, tmp_path):
        """Renders single note correctly."""
        from soundlab.transcription.models import NoteEvent
        from soundlab.transcription.visualization import render_piano_roll

        note = NoteEvent(pitch=60, start=0.0, end=1.0, velocity=100)
        output = tmp_path / "single.png"
        render_piano_roll([note], output)

        mock_matplotlib["ax"].broken_barh.assert_called_once()
        call_args = mock_matplotlib["ax"].broken_barh.call_args
        xranges = call_args[0][0]
        yrange = call_args[0][1]

        assert xranges == [(0.0, 1.0)]
        assert yrange == (59.6, 0.8)

    def test_multiple_notes(self, mock_matplotlib, sample_notes, tmp_path):
        """Renders multiple notes."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "multiple.png"
        render_piano_roll(sample_notes, output)

        assert mock_matplotlib["ax"].broken_barh.call_count == 3

    def test_creates_output_directory(self, mock_matplotlib, sample_notes, tmp_path):  # noqa: ARG002
        """Creates parent directories for output."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "nested" / "dir" / "output.png"
        render_piano_roll(sample_notes, output)

        assert output.parent.exists()

    def test_returns_path_object(self, mock_matplotlib, sample_notes, tmp_path):  # noqa: ARG002
        """Returns Path object."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "result.png"
        result = render_piano_roll(sample_notes, output)

        assert isinstance(result, Path)
        assert result == output

    def test_custom_figsize(self, mock_matplotlib, sample_notes, tmp_path):
        """Passes custom figsize to subplots."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "figsize.png"
        figsize = (16, 8)
        render_piano_roll(sample_notes, output, figsize=figsize)

        mock_matplotlib["subplots"].assert_called_once_with(figsize=figsize)

    def test_custom_colormap(self, mock_matplotlib, sample_notes, tmp_path):
        """Uses custom colormap name."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "colormap.png"
        colormap = "plasma"
        render_piano_roll(sample_notes, output, colormap=colormap)

        mock_matplotlib["colormaps"].__getitem__.assert_called_with(colormap)

    def test_closes_figure(self, mock_matplotlib, sample_notes, tmp_path):
        """Calls plt.close(fig) after saving."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "close.png"
        render_piano_roll(sample_notes, output)

        mock_matplotlib["close"].assert_called_once_with(mock_matplotlib["fig"])

    def test_velocity_maps_to_color(self, mock_matplotlib, sample_notes, tmp_path):
        """Velocity values map to colormap."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "velocity.png"
        render_piano_roll(sample_notes, output)

        cmap_calls = mock_matplotlib["cmap"].call_args_list
        assert len(cmap_calls) == 3

        velocities = [100, 80, 90]
        for i, call in enumerate(cmap_calls):
            norm_velocity = velocities[i] / 127.0
            assert abs(call[0][0] - norm_velocity) < 0.01

    def test_zero_duration_note_handled(self, mock_matplotlib, tmp_path):
        """Handles note.end == note.start."""
        from soundlab.transcription.models import NoteEvent
        from soundlab.transcription.visualization import render_piano_roll

        note = NoteEvent(pitch=60, start=1.0, end=1.0, velocity=80)
        output = tmp_path / "zero_duration.png"
        render_piano_roll([note], output)

        call_args = mock_matplotlib["ax"].broken_barh.call_args
        xranges = call_args[0][0]
        assert xranges == [(1.0, 0.0)]

    def test_saves_file(self, mock_matplotlib, sample_notes, tmp_path):
        """Calls fig.savefig() with correct path."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "saved.png"
        render_piano_roll(sample_notes, output)

        mock_matplotlib["fig"].savefig.assert_called_once_with(output, dpi=150)

    def test_ylim_from_note_pitches(self, mock_matplotlib, sample_notes, tmp_path):
        """Sets ylim based on min/max pitch with padding."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "ylim.png"
        render_piano_roll(sample_notes, output)

        mock_matplotlib["ax"].set_ylim.assert_called_once_with(59, 68)


class TestRenderPianoRollEdgeCases:
    """Edge case tests for render_piano_roll function."""

    def test_accepts_string_path(self, mock_matplotlib, sample_notes, tmp_path):  # noqa: ARG002
        """Accepts string path instead of Path object."""
        from soundlab.transcription.visualization import render_piano_roll

        output = str(tmp_path / "string_path.png")
        result = render_piano_roll(sample_notes, output)

        assert isinstance(result, Path)
        assert str(result) == output

    def test_accepts_generator(self, mock_matplotlib, tmp_path):
        """Accepts generator of notes."""
        from soundlab.transcription.models import NoteEvent
        from soundlab.transcription.visualization import render_piano_roll

        def note_generator():
            yield NoteEvent(pitch=60, start=0.0, end=1.0, velocity=100)
            yield NoteEvent(pitch=64, start=0.5, end=1.5, velocity=80)

        output = tmp_path / "generator.png"
        render_piano_roll(note_generator(), output)

        assert mock_matplotlib["ax"].broken_barh.call_count == 2

    def test_single_pitch_ylim(self, mock_matplotlib, tmp_path):
        """Sets ylim correctly for single pitch value."""
        from soundlab.transcription.models import NoteEvent
        from soundlab.transcription.visualization import render_piano_roll

        notes = [
            NoteEvent(pitch=60, start=0.0, end=1.0, velocity=100),
            NoteEvent(pitch=60, start=1.0, end=2.0, velocity=80),
        ]
        output = tmp_path / "single_pitch.png"
        render_piano_roll(notes, output)

        mock_matplotlib["ax"].set_ylim.assert_called_once_with(59, 61)

    def test_default_figsize(self, mock_matplotlib, sample_notes, tmp_path):
        """Uses default figsize (12, 4) when not specified."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "default_figsize.png"
        render_piano_roll(sample_notes, output)

        mock_matplotlib["subplots"].assert_called_once_with(figsize=(12, 4))

    def test_default_colormap(self, mock_matplotlib, sample_notes, tmp_path):
        """Uses default colormap 'viridis' when not specified."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "default_colormap.png"
        render_piano_roll(sample_notes, output)

        mock_matplotlib["colormaps"].__getitem__.assert_called_with("viridis")

    def test_sets_axis_labels(self, mock_matplotlib, sample_notes, tmp_path):
        """Sets correct axis labels."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "labels.png"
        render_piano_roll(sample_notes, output)

        mock_matplotlib["ax"].set_xlabel.assert_called_once_with("Time (s)")
        mock_matplotlib["ax"].set_ylabel.assert_called_once_with("MIDI pitch")

    def test_sets_title(self, mock_matplotlib, sample_notes, tmp_path):
        """Sets correct plot title."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "title.png"
        render_piano_roll(sample_notes, output)

        mock_matplotlib["ax"].set_title.assert_called_once_with("Piano Roll")

    def test_enables_grid(self, mock_matplotlib, sample_notes, tmp_path):
        """Enables grid on x-axis."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "grid.png"
        render_piano_roll(sample_notes, output)

        mock_matplotlib["ax"].grid.assert_called_once_with(True, axis="x", alpha=0.3)

    def test_tight_layout_called(self, mock_matplotlib, sample_notes, tmp_path):
        """Calls tight_layout before saving."""
        from soundlab.transcription.visualization import render_piano_roll

        output = tmp_path / "tight_layout.png"
        render_piano_roll(sample_notes, output)

        mock_matplotlib["fig"].tight_layout.assert_called_once()
