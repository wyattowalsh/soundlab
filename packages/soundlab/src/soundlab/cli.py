"""SoundLab command-line interface."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated

import typer

from soundlab.analysis import analyze_audio
from soundlab.effects import EffectsChain
from soundlab.separation import DemucsModel, SeparationConfig, StemSeparator
from soundlab.transcription import MIDITranscriber, TranscriptionConfig

app = typer.Typer(help="SoundLab CLI", add_completion=False)


@app.command()
def separate(
    audio_path: Annotated[Path, typer.Argument(help="Path to input audio file.")],
    output_dir: Annotated[Path, typer.Argument(help="Directory to write stem files.")],
    model: Annotated[
        DemucsModel, typer.Option(help="Demucs model name.")
    ] = DemucsModel.HTDEMUCS_FT,
    device: Annotated[str, typer.Option(help="Device selection (auto/cpu/cuda).")] = "auto",
    split: Annotated[bool, typer.Option(help="Enable chunked separation.")] = True,
    vocals_only: Annotated[
        bool, typer.Option("--vocals-only", help="Isolate vocals only (outputs vocals + instrumental stems).")
    ] = False,
) -> None:
    """Separate a mix into stems."""
    two_stems = "vocals" if vocals_only else None
    config = SeparationConfig(model=model, device=device, split=split, two_stems=two_stems)
    separator = StemSeparator(config)
    result = separator.separate(audio_path, output_dir)
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def transcribe(
    audio_path: Annotated[Path, typer.Argument(help="Path to input audio file.")],
    output_dir: Annotated[Path, typer.Argument(help="Directory to write MIDI output.")],
    onset_thresh: Annotated[float, typer.Option(help="Onset threshold.")] = 0.5,
    frame_thresh: Annotated[float, typer.Option(help="Frame threshold.")] = 0.3,
    min_note_length: Annotated[float, typer.Option(help="Minimum note length (seconds).")] = 0.058,
    min_freq: Annotated[float, typer.Option(help="Minimum frequency (Hz).")] = 32.7,
    max_freq: Annotated[float, typer.Option(help="Maximum frequency (Hz).")] = 2093.0,
) -> None:
    """Transcribe audio to MIDI using Basic Pitch."""
    config = TranscriptionConfig(
        onset_thresh=onset_thresh,
        frame_thresh=frame_thresh,
        min_note_length=min_note_length,
        min_freq=min_freq,
        max_freq=max_freq,
    )
    transcriber = MIDITranscriber(config)
    result = transcriber.transcribe(audio_path, output_dir)
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def analyze(
    audio_path: Annotated[Path, typer.Argument(help="Path to input audio file.")],
    output_json: Annotated[
        Path | None, typer.Option(help="Optional JSON output path for analysis results.")
    ] = None,
) -> None:
    """Analyze audio tempo, key, loudness, and spectral features."""
    result = analyze_audio(audio_path)
    payload = result.model_dump_json(indent=2)
    if output_json is None:
        typer.echo(payload)
        return
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(payload)
    typer.echo(f"Saved analysis to {output_json}")


@app.command()
def effects(
    input_path: Annotated[Path, typer.Argument(help="Path to input audio file.")],
    output_path: Annotated[Path, typer.Argument(help="Path to output audio file.")],
) -> None:
    """Apply an effects chain (currently pass-through unless configured in code)."""
    chain = EffectsChain()
    result = chain.process(input_path, output_path)
    typer.echo(f"Wrote processed audio to {result}")


@app.command()
def tts(
    text: Annotated[str, typer.Argument(help="Text to synthesize.")],
    output_path: Annotated[Path, typer.Argument(help="Path to output audio file.")],
) -> None:
    """Text-to-speech placeholder (requires voice implementation)."""
    _ = text
    _ = output_path
    typer.echo("TTS is not implemented yet. Install voice extras and use soundlab.voice.")
    raise typer.Exit(code=1)


def main() -> None:
    """CLI entry point."""
    app()


__all__ = ["app", "main"]
