"""Visualization helpers for transcription output."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Iterable

    from soundlab.transcription.models import NoteEvent


def render_piano_roll(
    notes: Iterable[NoteEvent],
    output_path: str | Path,
    figsize: tuple[float, float] = (12, 4),
    colormap: str = "viridis",
) -> Path:
    """Render a simple piano roll and save it to disk."""
    output = Path(output_path)
    note_list = list(notes)

    fig, ax = plt.subplots(figsize=figsize)
    if note_list:
        pitches = [note.pitch for note in note_list]
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        norm = mpl.colors.Normalize(vmin=0, vmax=127)
        cmap = mpl.colormaps[colormap]

        for note in note_list:
            duration = max(note.end - note.start, 0.0)
            ax.broken_barh(
                [(note.start, duration)],
                (note.pitch - 0.4, 0.8),
                facecolors=cmap(norm(note.velocity)),
            )

        ax.set_ylim(min_pitch - 1, max_pitch + 1)
    else:
        ax.set_ylim(0, 127)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI pitch")
    ax.set_title("Piano Roll")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
