#!/usr/bin/env python3
"""
Audio Analysis Example

This script demonstrates how to use SoundLab to perform comprehensive
audio analysis including tempo, key, loudness, spectral features, and onsets.

Usage:
    python examples/analyze_audio.py input.mp3
    python examples/analyze_audio.py song.wav --no-spectral
    python examples/analyze_audio.py track.flac --only tempo key
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from soundlab.analysis import analyze_audio


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive audio analysis with SoundLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis
  %(prog)s input.mp3

  # Disable specific analyses
  %(prog)s input.mp3 --no-spectral --no-onsets

  # Only specific analyses
  %(prog)s input.mp3 --only tempo key loudness

Analysis Components:
  tempo     - BPM detection with confidence scores
  key       - Musical key detection (e.g., "A minor")
  loudness  - LUFS measurements (integrated, short-term, momentary)
  spectral  - Frequency content (centroid, bandwidth, rolloff, etc.)
  onsets    - Beat and onset detection
        """,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input audio file (WAV, MP3, FLAC, etc.)",
    )

    parser.add_argument(
        "--no-tempo",
        action="store_true",
        help="Skip tempo/BPM detection",
    )

    parser.add_argument(
        "--no-key",
        action="store_true",
        help="Skip key detection",
    )

    parser.add_argument(
        "--no-loudness",
        action="store_true",
        help="Skip loudness analysis",
    )

    parser.add_argument(
        "--no-spectral",
        action="store_true",
        help="Skip spectral analysis",
    )

    parser.add_argument(
        "--no-onsets",
        action="store_true",
        help="Skip onset detection",
    )

    parser.add_argument(
        "--only",
        nargs="+",
        choices=["tempo", "key", "loudness", "spectral", "onsets"],
        help="Run only specific analyses",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def format_key(key_result) -> str:
    """Format key detection result."""
    if not key_result:
        return "N/A"
    return f"{key_result.tonic} {key_result.mode.value}"


def format_tempo(tempo_result) -> str:
    """Format tempo result with confidence."""
    if not tempo_result:
        return "N/A"
    conf_pct = tempo_result.confidence * 100
    return f"{tempo_result.bpm:.1f} BPM (confidence: {conf_pct:.0f}%)"


def format_loudness(loudness_result) -> dict[str, str]:
    """Format loudness results."""
    if not loudness_result:
        return {
            "Integrated": "N/A",
            "Short-term Max": "N/A",
            "Momentary Max": "N/A",
            "LRA": "N/A",
            "True Peak": "N/A",
        }

    return {
        "Integrated": f"{loudness_result.integrated_lufs:.2f} LUFS",
        "Short-term Max": f"{loudness_result.short_term_max_lufs:.2f} LUFS",
        "Momentary Max": f"{loudness_result.momentary_max_lufs:.2f} LUFS",
        "LRA": f"{loudness_result.loudness_range_lu:.2f} LU",
        "True Peak": f"{loudness_result.true_peak_dbfs:.2f} dBFS",
    }


def format_spectral(spectral_result) -> dict[str, str]:
    """Format spectral analysis results."""
    if not spectral_result:
        return {
            "Centroid": "N/A",
            "Bandwidth": "N/A",
            "Rolloff": "N/A",
            "Flatness": "N/A",
            "RMS Energy": "N/A",
            "Zero Crossing Rate": "N/A",
        }

    return {
        "Centroid": f"{spectral_result.spectral_centroid_hz:.1f} Hz",
        "Bandwidth": f"{spectral_result.spectral_bandwidth_hz:.1f} Hz",
        "Rolloff": f"{spectral_result.spectral_rolloff_hz:.1f} Hz",
        "Flatness": f"{spectral_result.spectral_flatness:.3f}",
        "RMS Energy": f"{spectral_result.rms_energy:.4f}",
        "Zero Crossing Rate": f"{spectral_result.zero_crossing_rate:.1f} Hz",
    }


def format_onsets(onset_result) -> dict[str, str]:
    """Format onset detection results."""
    if not onset_result:
        return {
            "Total Onsets": "N/A",
            "Onset Rate": "N/A",
            "Beat Times": "N/A",
        }

    onset_rate = onset_result.onset_count / onset_result.duration_seconds
    beats_str = f"{onset_result.beat_count} beats" if onset_result.beat_count > 0 else "N/A"

    return {
        "Total Onsets": str(onset_result.onset_count),
        "Onset Rate": f"{onset_rate:.2f} onsets/sec",
        "Beat Times": beats_str,
    }


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Validate input file
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    if not args.input.is_file():
        logger.error(f"Input path is not a file: {args.input}")
        return 1

    # Determine which analyses to run
    if args.only:
        # Only run specified analyses
        include_tempo = "tempo" in args.only
        include_key = "key" in args.only
        include_loudness = "loudness" in args.only
        include_spectral = "spectral" in args.only
        include_onsets = "onsets" in args.only
    else:
        # Run all except those disabled
        include_tempo = not args.no_tempo
        include_key = not args.no_key
        include_loudness = not args.no_loudness
        include_spectral = not args.no_spectral
        include_onsets = not args.no_onsets

    print(f"\n{'=' * 70}")
    print("SoundLab Audio Analysis")
    print(f"{'=' * 70}")
    print(f"File: {args.input}\n")

    try:
        # Run analysis
        logger.info("Starting audio analysis...")

        result = analyze_audio(
            args.input,
            include_tempo=include_tempo,
            include_key=include_key,
            include_loudness=include_loudness,
            include_spectral=include_spectral,
            include_onsets=include_onsets,
        )

        # Print basic information
        print("Basic Information:")
        print(f"  Duration:      {result.duration_seconds:.2f} seconds")
        print(f"  Sample Rate:   {result.sample_rate} Hz")
        print(f"  Channels:      {result.channels}")
        print()

        # Print tempo results
        if include_tempo and result.tempo:
            print("Tempo Analysis:")
            print(f"  Primary BPM:   {format_tempo(result.tempo)}")
            print()

        # Print key results
        if include_key and result.key:
            print("Key Detection:")
            print(f"  Key:           {format_key(result.key)}")
            print(f"  Confidence:    {result.key.confidence * 100:.0f}%")
            print()

        # Print loudness results
        if include_loudness and result.loudness:
            print("Loudness Analysis:")
            loudness_data = format_loudness(result.loudness)
            for label, value in loudness_data.items():
                print(f"  {label:18s} {value}")
            print()

        # Print spectral results
        if include_spectral and result.spectral:
            print("Spectral Analysis:")
            spectral_data = format_spectral(result.spectral)
            for label, value in spectral_data.items():
                print(f"  {label:22s} {value}")
            print()

        # Print onset results
        if include_onsets and result.onsets:
            print("Onset Detection:")
            onset_data = format_onsets(result.onsets)
            for label, value in onset_data.items():
                print(f"  {label:18s} {value}")
            print()

        # Print summary
        print(f"{'=' * 70}")
        print("Summary:")
        summary = result.summary
        if summary.get("bpm"):
            print(f"  BPM:           {summary['bpm']}")
        if summary.get("key"):
            print(f"  Key:           {summary['key']}")
        if summary.get("lufs"):
            print(f"  Loudness:      {summary['lufs']}")
        if summary.get("spectral_centroid"):
            print(f"  Brightness:    {summary['spectral_centroid']}")
        if summary.get("onsets"):
            print(f"  Onsets:        {summary['onsets']}")
        print(f"{'=' * 70}\n")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
