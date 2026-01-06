#!/usr/bin/env python3
"""
Audio Effects Example

This script demonstrates how to use SoundLab to apply audio effects chains
to audio files. Includes preset chains and custom effect configurations.

Usage:
    python examples/apply_effects.py input.wav output.wav --preset mastering
    python examples/apply_effects.py song.mp3 processed.wav --preset vocal
    python examples/apply_effects.py track.flac out.wav --custom
"""

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

from soundlab.effects import (
    ChorusConfig,
    CompressorConfig,
    DelayConfig,
    DistortionConfig,
    EffectsChain,
    GainConfig,
    HighpassConfig,
    LimiterConfig,
    LowpassConfig,
    PeakFilterConfig,
    ReverbConfig,
)


def create_mastering_chain() -> EffectsChain:
    """
    Create a mastering effects chain.

    Typical mastering chain: EQ -> Compression -> Limiting
    """
    chain = EffectsChain()

    # Subtle EQ adjustments
    chain.add(
        HighpassConfig(
            cutoff_frequency_hz=30.0,  # Remove sub-bass rumble
        )
    )
    chain.add(
        PeakFilterConfig(
            cutoff_frequency_hz=3000.0,  # Presence boost
            gain_db=1.5,
            q=1.0,
        )
    )

    # Gentle compression for glue
    chain.add(
        CompressorConfig(
            threshold_db=-18.0,
            ratio=2.5,
            attack_ms=10.0,
            release_ms=100.0,
        )
    )

    # Final limiting for loudness
    chain.add(
        LimiterConfig(
            threshold_db=-1.0,
            release_ms=50.0,
        )
    )

    return chain


def create_vocal_chain() -> EffectsChain:
    """
    Create a vocal processing chain.

    Typical vocal chain: HP Filter -> Compression -> EQ -> Reverb
    """
    chain = EffectsChain()

    # Remove low-end rumble
    chain.add(
        HighpassConfig(
            cutoff_frequency_hz=80.0,
        )
    )

    # Compression for consistency
    chain.add(
        CompressorConfig(
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=5.0,
            release_ms=50.0,
        )
    )

    # Presence boost
    chain.add(
        PeakFilterConfig(
            cutoff_frequency_hz=5000.0,
            gain_db=3.0,
            q=1.5,
        )
    )

    # Subtle reverb for space
    chain.add(
        ReverbConfig(
            room_size=0.3,
            damping=0.5,
            wet_level=0.15,
            dry_level=0.85,
        )
    )

    return chain


def create_lofi_chain() -> EffectsChain:
    """
    Create a lo-fi aesthetic chain.

    Lo-fi elements: Low-pass -> Distortion -> Chorus
    """
    chain = EffectsChain()

    # Reduce high frequencies
    chain.add(
        LowpassConfig(
            cutoff_frequency_hz=4000.0,
        )
    )

    # Add warmth/grit
    chain.add(
        DistortionConfig(
            drive_db=8.0,
        )
    )

    # Subtle chorus for vintage feel
    chain.add(
        ChorusConfig(
            rate_hz=0.5,
            depth=0.3,
            centre_delay_ms=7.0,
            feedback=0.2,
            mix=0.3,
        )
    )

    # Slight gain reduction
    chain.add(
        GainConfig(
            gain_db=-2.0,
        )
    )

    return chain


def create_radio_chain() -> EffectsChain:
    """
    Create a radio/telephone effect chain.

    Radio effect: Bandpass (HP + LP) -> Compression -> Distortion
    """
    chain = EffectsChain()

    # Bandpass filter (300-3000 Hz)
    chain.add(
        HighpassConfig(
            cutoff_frequency_hz=300.0,
        )
    )
    chain.add(
        LowpassConfig(
            cutoff_frequency_hz=3000.0,
        )
    )

    # Heavy compression
    chain.add(
        CompressorConfig(
            threshold_db=-25.0,
            ratio=10.0,
            attack_ms=1.0,
            release_ms=100.0,
        )
    )

    # Distortion for "broken speaker" effect
    chain.add(
        DistortionConfig(
            drive_db=15.0,
        )
    )

    return chain


def create_spacey_chain() -> EffectsChain:
    """
    Create an ambient/spacey effect chain.

    Spacey effects: Reverb -> Delay -> Chorus
    """
    chain = EffectsChain()

    # Large reverb
    chain.add(
        ReverbConfig(
            room_size=0.8,
            damping=0.3,
            wet_level=0.4,
            dry_level=0.6,
            width=1.0,
        )
    )

    # Delay for depth
    chain.add(
        DelayConfig(
            delay_seconds=0.375,  # Dotted eighth at 120 BPM
            feedback=0.4,
            mix=0.3,
        )
    )

    # Chorus for width
    chain.add(
        ChorusConfig(
            rate_hz=0.8,
            depth=0.5,
            centre_delay_ms=10.0,
            feedback=0.3,
            mix=0.4,
        )
    )

    return chain


PRESETS = {
    "mastering": ("Mastering chain (EQ, compression, limiting)", create_mastering_chain),
    "vocal": ("Vocal processing (HP filter, compression, EQ, reverb)", create_vocal_chain),
    "lofi": ("Lo-fi aesthetic (low-pass, distortion, chorus)", create_lofi_chain),
    "radio": ("Radio/telephone effect (bandpass, compression, distortion)", create_radio_chain),
    "spacey": ("Ambient/spacey (reverb, delay, chorus)", create_spacey_chain),
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply audio effects chains with SoundLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply mastering preset
  %(prog)s input.wav output.wav --preset mastering

  # Apply vocal processing
  %(prog)s vocals.mp3 processed.wav --preset vocal

  # Apply lo-fi effect
  %(prog)s song.wav lofi.wav --preset lofi

  # List available presets
  %(prog)s --list-presets

Available Presets:
  mastering - Professional mastering chain
  vocal     - Vocal processing and enhancement
  lofi      - Lo-fi/vintage aesthetic
  radio     - Radio/telephone effect
  spacey    - Ambient/spacey atmosphere
        """,
    )

    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Input audio file (WAV, MP3, FLAC, etc.)",
    )

    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output audio file",
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        help="Effects preset to apply",
    )

    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def list_presets() -> None:
    """Print available presets and their descriptions."""
    print("\nAvailable Effects Presets:")
    print("=" * 70)
    for name, (description, _) in PRESETS.items():
        print(f"\n{name.upper()}")
        print(f"  {description}")
    print("\n" + "=" * 70 + "\n")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle --list-presets
    if args.list_presets:
        list_presets()
        return 0

    # Validate required arguments
    if not args.input or not args.output:
        logger.error("Both input and output files are required")
        return 1

    if not args.preset:
        logger.error("--preset is required. Use --list-presets to see available options")
        return 1

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
    preset_desc = PRESETS[args.preset][0]
    print(f"\n{'=' * 60}")
    print("SoundLab Effects Processing")
    print(f"{'=' * 60}")
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Preset:      {args.preset}")
    print(f"Description: {preset_desc}")
    print(f"{'=' * 60}\n")

    try:
        # Create effects chain
        logger.info(f"Creating '{args.preset}' effects chain...")
        chain_factory = PRESETS[args.preset][1]
        chain = chain_factory()

        # Print chain details
        print("Effects Chain:")
        for i, effect in enumerate(chain.effects, 1):
            print(f"  {i}. {effect.name}")
        print()

        # Process audio
        logger.info("Processing audio...")
        start_time = time.time()

        output_path = chain.process(
            input_path=args.input,
            output_path=args.output,
            preserve_format=True,
        )

        elapsed = time.time() - start_time

        # Print results
        output_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"\n{'=' * 60}")
        print("Processing Complete!")
        print(f"{'=' * 60}")
        print(f"Processing time: {elapsed:.2f}s")
        print(f"Output file:     {output_path}")
        print(f"File size:       {output_size:.2f} MB")
        print(f"{'=' * 60}\n")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Effects processing failed: {e}")
        if args.verbose:
            logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
