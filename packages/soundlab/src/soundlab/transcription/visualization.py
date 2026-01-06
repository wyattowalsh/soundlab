"""Visualization utilities for MIDI transcription."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from soundlab.transcription.models import NoteEvent


__all__ = ["render_piano_roll", "render_note_density"]


def render_piano_roll(
    notes: Sequence[NoteEvent],
    output_path: Path | str | None = None,
    *,
    figsize: tuple[int, int] = (16, 8),
    colormap: str = "viridis",
    title: str = "Piano Roll",
    show_velocity: bool = True,
    time_range: tuple[float, float] | None = None,
    pitch_range: tuple[int, int] | None = None,
) -> Path | None:
    """
    Render a piano roll visualization of MIDI notes.

    Parameters
    ----------
    notes
        Sequence of NoteEvent objects.
    output_path
        Path to save the image. If None, displays interactively.
    figsize
        Figure size as (width, height) in inches.
    colormap
        Matplotlib colormap name for velocity coloring.
    title
        Plot title.
    show_velocity
        Whether to color notes by velocity.
    time_range
        Optional (start, end) time range to display.
    pitch_range
        Optional (min_pitch, max_pitch) range to display.

    Returns
    -------
    Path | None
        Path to saved image, or None if displayed interactively.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib is required for piano roll visualization")
        return None

    if not notes:
        logger.warning("No notes to visualize")
        return None

    # Filter by time range if specified
    if time_range:
        notes = [n for n in notes if n.start_time >= time_range[0] and n.end_time <= time_range[1]]

    if not notes:
        logger.warning("No notes in specified time range")
        return None

    # Determine pitch range
    all_pitches = [n.pitch for n in notes]
    if pitch_range:
        min_pitch, max_pitch = pitch_range
    else:
        min_pitch = min(all_pitches) - 2
        max_pitch = max(all_pitches) + 2

    # Determine time range
    if time_range:
        min_time, max_time = time_range
    else:
        min_time = 0
        max_time = max(n.end_time for n in notes)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Setup colormap for velocity
    if show_velocity:
        norm = Normalize(vmin=0, vmax=127)
        cmap = cm.get_cmap(colormap)

    # Draw notes
    for note in notes:
        if note.pitch < min_pitch or note.pitch > max_pitch:
            continue

        # Note rectangle
        rect_x = note.start_time
        rect_y = note.pitch - 0.4
        rect_width = note.duration
        rect_height = 0.8

        if show_velocity:
            color = cmap(norm(note.velocity))
        else:
            color = cmap(0.6)

        rect = patches.Rectangle(
            (rect_x, rect_y),
            rect_width,
            rect_height,
            linewidth=0.5,
            edgecolor="black",
            facecolor=color,
            alpha=0.8,
        )
        ax.add_patch(rect)

    # Configure axes
    ax.set_xlim(min_time, max_time)
    ax.set_ylim(min_pitch - 1, max_pitch + 1)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title(title)

    # Add pitch labels on y-axis
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    yticks = list(range(min_pitch, max_pitch + 1, 12))  # Every octave
    ylabels = [f"{note_names[p % 12]}{p // 12 - 1}" for p in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add colorbar for velocity
    if show_velocity:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Velocity")

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"Saved piano roll: {output_path}")
        return output_path
    else:
        plt.show()
        return None


def render_note_density(
    notes: Sequence[NoteEvent],
    output_path: Path | str | None = None,
    *,
    bin_size_seconds: float = 1.0,
    figsize: tuple[int, int] = (12, 4),
    title: str = "Note Density Over Time",
) -> Path | None:
    """
    Render a histogram of note density over time.

    Parameters
    ----------
    notes
        Sequence of NoteEvent objects.
    output_path
        Path to save the image.
    bin_size_seconds
        Size of time bins in seconds.
    figsize
        Figure size.
    title
        Plot title.

    Returns
    -------
    Path | None
        Path to saved image.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib is required for visualization")
        return None

    if not notes:
        logger.warning("No notes to visualize")
        return None

    # Get note start times
    start_times = [n.start_time for n in notes]
    max_time = max(n.end_time for n in notes)

    # Create bins
    bins = np.arange(0, max_time + bin_size_seconds, bin_size_seconds)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    ax.hist(start_times, bins=bins, edgecolor="black", alpha=0.7)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Note Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path
    else:
        plt.show()
        return None
