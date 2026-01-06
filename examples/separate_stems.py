#!/usr/bin/env python3
"""
Stem Separation Example

This script demonstrates how to use SoundLab to separate audio into stems
(vocals, drums, bass, other) using the Demucs model.

Usage:
    python examples/separate_stems.py input.mp3 -o output/ -m htdemucs_ft
    python examples/separate_stems.py song.wav --two-stems vocals
    python examples/separate_stems.py track.flac -o stems/ --shifts 2
"""

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

from soundlab.separation import DemucsModel, SeparationConfig, StemSeparator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Separate audio into stems using Demucs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Separate with default settings
  %(prog)s input.mp3

  # Use specific model
  %(prog)s input.mp3 -m htdemucs_ft

  # Extract only vocals
  %(prog)s input.mp3 --two-stems vocals

  # Advanced settings
  %(prog)s input.mp3 -o stems/ --shifts 2 --segment 10.0

Available models:
  htdemucs      - Fast, good quality (default)
  htdemucs_ft   - Best quality, slower
  htdemucs_6s   - 6 stems (piano/guitar - experimental)
  mdx_extra     - Alternative MDX architecture
  mdx_extra_q   - Quantized MDX (faster)

Two-stems options: vocals, drums, bass, other
        """,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input audio file (WAV, MP3, FLAC, etc.)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("separated"),
        help="Output directory for stems (default: separated/)",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="htdemucs",
        choices=[m.value for m in DemucsModel],
        help="Demucs model to use (default: htdemucs)",
    )

    parser.add_argument(
        "--two-stems",
        type=str,
        choices=["vocals", "drums", "bass", "other"],
        help="Extract only one stem instead of all stems",
    )

    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        choices=[0, 1, 2, 3, 4, 5],
        help="Random shifts for demucsing (more = better quality, slower) (default: 1)",
    )

    parser.add_argument(
        "--segment",
        type=float,
        default=7.8,
        help="Segment length in seconds (default: 7.8)",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap between segments (0.0-1.0) (default: 0.25)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Processing device (default: auto)",
    )

    parser.add_argument(
        "--mp3",
        action="store_true",
        help="Save stems as MP3 instead of WAV",
    )

    parser.add_argument(
        "--mp3-bitrate",
        type=int,
        default=320,
        choices=[128, 192, 256, 320],
        help="MP3 bitrate in kbps (default: 320)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


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

    # Print configuration
    print(f"\n{'=' * 60}")
    print("SoundLab Stem Separation")
    print(f"{'=' * 60}")
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Model:       {args.model}")
    if args.two_stems:
        print(f"Two-stems:   {args.two_stems}")
    print(f"Shifts:      {args.shifts}")
    print(f"Segment:     {args.segment}s")
    print(f"Device:      {args.device}")
    print(f"{'=' * 60}\n")

    try:
        # Create configuration
        config = SeparationConfig(
            model=DemucsModel(args.model),
            segment_length=args.segment,
            overlap=args.overlap,
            shifts=args.shifts,
            two_stems=args.two_stems,
            device=args.device,
            mp3_bitrate=args.mp3_bitrate,
        )

        # Create separator
        separator = StemSeparator(config=config)

        # Separate stems
        logger.info("Starting separation...")
        start_time = time.time()

        result = separator.separate(
            audio_path=args.input,
            output_dir=args.output,
        )

        elapsed = time.time() - start_time

        # Print results
        print(f"\n{'=' * 60}")
        print("Separation Complete!")
        print(f"{'=' * 60}")
        print(f"Processing time: {elapsed:.2f}s")
        print(f"Output directory: {args.output}")
        print(f"\nExtracted stems:")
        for stem_name, stem_path in result.stems.items():
            file_size = stem_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  {stem_name:10s} -> {stem_path.name} ({file_size:.2f} MB)")
        print(f"{'=' * 60}\n")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Separation failed: {e}")
        if args.verbose:
            logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
