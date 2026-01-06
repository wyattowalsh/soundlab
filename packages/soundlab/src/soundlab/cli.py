"""Command-line interface for SoundLab."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

__all__ = ["app", "main"]

app = typer.Typer(
    name="soundlab",
    help="SoundLab - Production-ready music processing CLI",
    add_completion=False,
)
console = Console()


@app.command()
def separate(
    input_file: Annotated[Path, typer.Argument(help="Input audio file")],
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("./stems"),
    model: Annotated[str, typer.Option("--model", "-m", help="Demucs model")] = "htdemucs_ft",
    two_stems: Annotated[Optional[str], typer.Option("--two-stems", help="Extract only one stem")] = None,
    device: Annotated[str, typer.Option("--device", "-d", help="Device (auto/cuda/cpu)")] = "auto",
) -> None:
    """Separate audio into stems using Demucs."""
    from soundlab.separation import StemSeparator, SeparationConfig, DemucsModel

    console.print(f"[bold blue]Separating:[/bold blue] {input_file}")

    try:
        config = SeparationConfig(
            model=DemucsModel(model),
            two_stems=two_stems,
            device=device,
        )
        separator = StemSeparator(config)
        result = separator.separate(input_file, output_dir)

        console.print(f"[bold green]Complete![/bold green] ({result.processing_time_seconds:.1f}s)")
        for stem_name, stem_path in result.stems.items():
            console.print(f"  • {stem_name}: {stem_path}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def transcribe(
    input_file: Annotated[Path, typer.Argument(help="Input audio file")],
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("./midi"),
    onset_thresh: Annotated[float, typer.Option("--onset", help="Onset threshold")] = 0.5,
    frame_thresh: Annotated[float, typer.Option("--frame", help="Frame threshold")] = 0.3,
) -> None:
    """Transcribe audio to MIDI using Basic Pitch."""
    from soundlab.transcription import MIDITranscriber, TranscriptionConfig

    console.print(f"[bold blue]Transcribing:[/bold blue] {input_file}")

    try:
        config = TranscriptionConfig(
            onset_thresh=onset_thresh,
            frame_thresh=frame_thresh,
        )
        transcriber = MIDITranscriber(config)
        result = transcriber.transcribe(input_file, output_dir)

        console.print(f"[bold green]Complete![/bold green] ({result.processing_time_seconds:.1f}s)")
        console.print(f"  • Notes: {result.note_count}")
        console.print(f"  • MIDI: {result.midi_path}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def analyze(
    input_file: Annotated[Path, typer.Argument(help="Input audio file")],
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
) -> None:
    """Analyze audio file (tempo, key, loudness, etc.)."""
    from soundlab.analysis import analyze_audio
    import json

    console.print(f"[bold blue]Analyzing:[/bold blue] {input_file}")

    try:
        result = analyze_audio(input_file)

        if json_output:
            print(json.dumps(result.summary, indent=2))
        else:
            table = Table(title="Audio Analysis Results")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            for key, value in result.summary.items():
                table.add_row(key, str(value))

            console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def effects(
    input_file: Annotated[Path, typer.Argument(help="Input audio file")],
    output_file: Annotated[Path, typer.Argument(help="Output audio file")],
    preset: Annotated[str, typer.Option("--preset", "-p", help="Effect preset")] = "default",
) -> None:
    """Apply audio effects using presets."""
    from soundlab.effects import EffectsChain, CompressorConfig, ReverbConfig, LimiterConfig

    console.print(f"[bold blue]Processing:[/bold blue] {input_file}")

    presets = {
        "default": [CompressorConfig(threshold_db=-20, ratio=4.0)],
        "master": [
            CompressorConfig(threshold_db=-18, ratio=3.0),
            LimiterConfig(threshold_db=-1.0),
        ],
        "vocal": [
            CompressorConfig(threshold_db=-24, ratio=4.0),
            ReverbConfig(room_size=0.3, wet_level=0.15),
        ],
        "ambient": [
            ReverbConfig(room_size=0.8, wet_level=0.5, damping=0.7),
        ],
    }

    if preset not in presets:
        console.print(f"[bold red]Unknown preset:[/bold red] {preset}")
        console.print(f"Available: {', '.join(presets.keys())}")
        raise typer.Exit(1)

    try:
        chain = EffectsChain.from_configs(presets[preset])
        chain.process(input_file, output_file)

        console.print(f"[bold green]Complete![/bold green] {output_file}")
        console.print(f"  • Preset: {preset}")
        console.print(f"  • Effects: {' -> '.join(chain.effect_names)}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def tts(
    text: Annotated[str, typer.Argument(help="Text to convert to speech")],
    output_file: Annotated[Path, typer.Argument(help="Output audio file")],
    language: Annotated[str, typer.Option("--lang", "-l", help="Language code")] = "en",
    speaker_wav: Annotated[Optional[Path], typer.Option("--speaker", "-s", help="Speaker reference audio")] = None,
) -> None:
    """Generate speech from text using XTTS-v2."""
    from soundlab.voice import TTSGenerator, TTSConfig, TTSLanguage

    console.print(f"[bold blue]Generating TTS:[/bold blue] {len(text)} characters")

    try:
        config = TTSConfig(
            text=text,
            language=TTSLanguage(language),
            speaker_wav=speaker_wav,
        )
        generator = TTSGenerator()
        result = generator.generate(config, output_file)

        console.print(f"[bold green]Complete![/bold green] ({result.processing_time_seconds:.1f}s)")
        console.print(f"  • Duration: {result.duration_seconds:.1f}s")
        console.print(f"  • Output: {result.audio_path}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show SoundLab version."""
    from soundlab import __version__
    console.print(f"SoundLab v{__version__}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
