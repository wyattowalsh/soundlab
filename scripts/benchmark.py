#!/usr/bin/env python3
"""Benchmark SoundLab modules.

This script benchmarks all SoundLab modules on reference audio files
and outputs results as a markdown table.

Usage:
    python scripts/benchmark.py                           # Run all benchmarks
    python scripts/benchmark.py --audio path/to/file.wav  # Use custom audio
    python scripts/benchmark.py --output results.md       # Save to file
    python scripts/benchmark.py --modules separation      # Benchmark specific modules
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    module: str
    operation: str
    duration_seconds: float
    memory_mb: float | None = None
    notes: str = ""


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """Context manager for timing operations."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["duration"] = time.perf_counter() - start


def get_gpu_memory() -> float | None:
    """Get current GPU memory usage in MB."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
    except ImportError:
        pass
    return None


def generate_test_audio(duration: float = 30.0, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Generate synthetic test audio.

    Parameters
    ----------
    duration
        Duration in seconds.
    sr
        Sample rate.

    Returns
    -------
    tuple
        Audio samples and sample rate.
    """
    t = np.linspace(0, duration, int(duration * sr), dtype=np.float32)

    # Mix of frequencies to simulate music
    audio = (
        0.3 * np.sin(2 * np.pi * 110 * t)  # Bass
        + 0.2 * np.sin(2 * np.pi * 220 * t)  # Low
        + 0.2 * np.sin(2 * np.pi * 440 * t)  # Mid
        + 0.1 * np.sin(2 * np.pi * 880 * t)  # High
        + 0.1 * np.sin(2 * np.pi * 1760 * t)  # Very high
    ).astype(np.float32)

    # Add some noise
    audio += 0.05 * np.random.randn(len(audio)).astype(np.float32)

    # Stereo
    audio = np.stack([audio, audio * 0.9])

    return audio, sr


def benchmark_io(audio_path: Path) -> list[BenchmarkResult]:
    """Benchmark I/O operations."""
    results = []

    try:
        from soundlab.io import get_audio_metadata, load_audio, save_audio
    except ImportError:
        return [BenchmarkResult("io", "import", 0, notes="Import failed")]

    # Load
    with timer() as t:
        audio = load_audio(audio_path)
    results.append(BenchmarkResult("io", "load_audio", t["duration"]))

    # Metadata
    with timer() as t:
        _ = get_audio_metadata(audio_path)
    results.append(BenchmarkResult("io", "get_metadata", t["duration"]))

    # Save
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        with timer() as t:
            save_audio(audio.samples, audio.sample_rate, Path(f.name))
        results.append(BenchmarkResult("io", "save_audio", t["duration"]))
        Path(f.name).unlink()

    return results


def benchmark_separation(audio_path: Path) -> list[BenchmarkResult]:
    """Benchmark stem separation."""
    results = []

    try:
        from soundlab.separation import DemucsModel, SeparationConfig, StemSeparator
    except ImportError:
        return [BenchmarkResult("separation", "import", 0, notes="Import failed")]

    # Test with fastest model
    config = SeparationConfig(
        model=DemucsModel.HTDEMUCS,
        segment=7.8,  # Shorter segment for speed
        shifts=1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        separator = StemSeparator(config)

        gc.collect()
        mem_before = get_gpu_memory()

        with timer() as t:
            try:
                result = separator.separate(audio_path, Path(tmpdir))
                notes = f"{len(result.stems)} stems"
            except Exception as e:
                notes = f"Failed: {e}"

        mem_after = get_gpu_memory()
        mem_used = (mem_after - mem_before) if mem_before and mem_after else None

        results.append(
            BenchmarkResult(
                "separation",
                "htdemucs",
                t["duration"],
                memory_mb=mem_used,
                notes=notes,
            )
        )

    return results


def benchmark_transcription(audio_path: Path) -> list[BenchmarkResult]:
    """Benchmark audio-to-MIDI transcription."""
    results = []

    try:
        from soundlab.transcription import BasicPitchTranscriber, TranscriptionConfig
    except ImportError:
        return [BenchmarkResult("transcription", "import", 0, notes="Import failed")]

    config = TranscriptionConfig()
    transcriber = BasicPitchTranscriber(config)

    with timer() as t:
        try:
            result = transcriber.transcribe(audio_path)
            notes = f"{len(result.notes)} notes"
        except Exception as e:
            notes = f"Failed: {e}"

    results.append(BenchmarkResult("transcription", "basic_pitch", t["duration"], notes=notes))

    return results


def benchmark_analysis(audio_path: Path) -> list[BenchmarkResult]:
    """Benchmark audio analysis."""
    results = []

    try:
        from soundlab.analysis import (
            analyze_loudness,
            analyze_spectral,
            detect_key,
            detect_onsets,
            detect_tempo,
        )
        from soundlab.io import load_audio
    except ImportError:
        return [BenchmarkResult("analysis", "import", 0, notes="Import failed")]

    audio = load_audio(audio_path)
    samples = audio.samples
    sr = audio.sample_rate

    # Mono for analysis
    mono = np.mean(samples, axis=0) if samples.ndim > 1 else samples

    # Tempo
    with timer() as t:
        try:
            tempo = detect_tempo(mono, sr)
            notes = f"{tempo.bpm:.1f} BPM"
        except Exception as e:
            notes = f"Failed: {e}"
    results.append(BenchmarkResult("analysis", "detect_tempo", t["duration"], notes=notes))

    # Key
    with timer() as t:
        try:
            key = detect_key(mono, sr)
            notes = f"{key.key} {key.mode}"
        except Exception as e:
            notes = f"Failed: {e}"
    results.append(BenchmarkResult("analysis", "detect_key", t["duration"], notes=notes))

    # Loudness
    with timer() as t:
        try:
            loudness = analyze_loudness(mono, sr)
            notes = f"{loudness.integrated_lufs:.1f} LUFS"
        except Exception as e:
            notes = f"Failed: {e}"
    results.append(BenchmarkResult("analysis", "analyze_loudness", t["duration"], notes=notes))

    # Spectral
    with timer() as t:
        try:
            spectral = analyze_spectral(mono, sr)
            notes = f"centroid={spectral.centroid_mean:.0f}Hz"
        except Exception as e:
            notes = f"Failed: {e}"
    results.append(BenchmarkResult("analysis", "analyze_spectral", t["duration"], notes=notes))

    # Onsets
    with timer() as t:
        try:
            onsets = detect_onsets(mono, sr)
            notes = f"{len(onsets.times)} onsets"
        except Exception as e:
            notes = f"Failed: {e}"
    results.append(BenchmarkResult("analysis", "detect_onsets", t["duration"], notes=notes))

    return results


def benchmark_effects(audio_path: Path) -> list[BenchmarkResult]:
    """Benchmark effects processing."""
    results = []

    try:
        from soundlab.effects import EffectsChain
        from soundlab.io import load_audio
    except ImportError:
        return [BenchmarkResult("effects", "import", 0, notes="Import failed")]

    audio = load_audio(audio_path)

    # Build chain with multiple effects
    chain = EffectsChain()

    try:
        chain.add_compressor(threshold_db=-20, ratio=4.0)
        chain.add_eq(low_gain_db=2.0, mid_gain_db=-1.0, high_gain_db=1.0)
        chain.add_reverb(room_size=0.5, damping=0.5, wet_level=0.3)
        chain.add_limiter(threshold_db=-1.0)
    except Exception as e:
        return [BenchmarkResult("effects", "setup", 0, notes=f"Failed: {e}")]

    with timer() as t:
        try:
            _ = chain.process(audio.samples, audio.sample_rate)
            notes = f"{len(chain)} effects"
        except Exception as e:
            notes = f"Failed: {e}"

    results.append(BenchmarkResult("effects", "chain_4fx", t["duration"], notes=notes))

    return results


def format_results_markdown(results: list[BenchmarkResult], audio_info: str) -> str:
    """Format benchmark results as markdown table.

    Parameters
    ----------
    results
        List of benchmark results.
    audio_info
        Description of audio used for benchmarking.

    Returns
    -------
    str
        Markdown-formatted results.
    """
    lines = [
        "# SoundLab Benchmark Results",
        "",
        f"**Audio:** {audio_info}",
        "",
        "| Module | Operation | Time (s) | Memory (MB) | Notes |",
        "|--------|-----------|----------|-------------|-------|",
    ]

    for r in results:
        time_str = f"{r.duration_seconds:.3f}"
        memory_str = f"{r.memory_mb:.1f}" if r.memory_mb else "-"
        lines.append(f"| {r.module} | {r.operation} | {time_str} | {memory_str} | {r.notes} |")

    # Summary
    total_time = sum(r.duration_seconds for r in results)
    lines.extend(
        [
            "",
            f"**Total time:** {total_time:.2f}s",
            "",
            "---",
            "",
            "*Generated by `scripts/benchmark.py`*",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark SoundLab modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--audio",
        type=Path,
        help="Path to audio file for benchmarking (default: generate synthetic)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration of synthetic audio in seconds (default: 30)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for results (default: stdout)",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=["io", "separation", "transcription", "analysis", "effects"],
        help="Modules to benchmark (default: all)",
    )

    args = parser.parse_args()

    print("=" * 60, file=sys.stderr)
    print("SoundLab Benchmark", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(file=sys.stderr)

    # Prepare audio
    if args.audio:
        audio_path = args.audio
        audio_info = f"{audio_path.name}"
        print(f"Using audio: {audio_path}", file=sys.stderr)
    else:
        print(f"Generating {args.duration}s synthetic audio...", file=sys.stderr)
        audio, sr = generate_test_audio(args.duration)
        audio_info = f"Synthetic {args.duration}s stereo @ {sr}Hz"

        # Save to temp file
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            sf.write(tmp_audio.name, audio.T, sr)
            audio_path = Path(tmp_audio.name)

    print(file=sys.stderr)

    # Run benchmarks
    results: list[BenchmarkResult] = []
    modules = args.modules or ["io", "separation", "transcription", "analysis", "effects"]

    if "io" in modules:
        print("Benchmarking I/O...", file=sys.stderr)
        results.extend(benchmark_io(audio_path))

    if "analysis" in modules:
        print("Benchmarking Analysis...", file=sys.stderr)
        results.extend(benchmark_analysis(audio_path))

    if "effects" in modules:
        print("Benchmarking Effects...", file=sys.stderr)
        results.extend(benchmark_effects(audio_path))

    if "transcription" in modules:
        print("Benchmarking Transcription...", file=sys.stderr)
        results.extend(benchmark_transcription(audio_path))

    if "separation" in modules:
        print("Benchmarking Separation...", file=sys.stderr)
        results.extend(benchmark_separation(audio_path))

    # Cleanup temp audio
    if not args.audio:
        audio_path.unlink()

    print(file=sys.stderr)

    # Format and output results
    markdown = format_results_markdown(results, audio_info)

    if args.output:
        args.output.write_text(markdown)
        print(f"Results saved to: {args.output}", file=sys.stderr)
    else:
        print(markdown)

    print(file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("✅ Benchmark complete!", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
