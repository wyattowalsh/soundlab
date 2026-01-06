#!/usr/bin/env python3
"""
Batch Processing Example

This script demonstrates how to batch process multiple audio files with SoundLab.
Supports stem separation, audio analysis, and effects processing for entire directories.

Usage:
    python examples/batch_process.py audio_dir/ --mode analyze
    python examples/batch_process.py songs/ --mode separate --output stems/
    python examples/batch_process.py tracks/ --mode effects --preset mastering
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from soundlab.analysis import analyze_audio
from soundlab.effects import (
    CompressorConfig,
    EffectsChain,
    GainConfig,
    HighpassConfig,
    LimiterConfig,
    ReverbConfig,
)
from soundlab.separation import DemucsModel, SeparationConfig, StemSeparator

# Supported audio extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def find_audio_files(directory: Path) -> list[Path]:
    """
    Find all audio files in a directory.

    Parameters
    ----------
    directory
        Directory to search.

    Returns
    -------
    list[Path]
        List of audio file paths.
    """
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(directory.glob(f"*{ext}"))
        audio_files.extend(directory.glob(f"*{ext.upper()}"))

    # Sort for consistent ordering
    return sorted(audio_files)


def batch_analyze(
    audio_files: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """
    Batch analyze audio files.

    Parameters
    ----------
    audio_files
        List of audio files to analyze.
    output_dir
        Directory to save analysis reports.

    Returns
    -------
    dict
        Summary statistics.
    """
    results = {}
    errors = []

    print(f"\nAnalyzing {len(audio_files)} files...\n")

    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Analyzing: {audio_file.name}")

        try:
            result = analyze_audio(audio_file)

            # Store result
            results[audio_file.name] = {
                "duration": result.duration_seconds,
                "sample_rate": result.sample_rate,
                "channels": result.channels,
                "summary": result.summary,
            }

            # Print quick summary
            summary = result.summary
            print(f"  BPM: {summary.get('bpm', 'N/A'):>8s}  "
                  f"Key: {summary.get('key', 'N/A'):>10s}  "
                  f"LUFS: {summary.get('lufs', 'N/A'):>8s}")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            errors.append((audio_file.name, str(e)))

    # Save detailed report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "analysis_report.json"

    with open(report_path, "w") as f:
        json.dump(
            {
                "results": results,
                "errors": errors,
                "total_files": len(audio_files),
                "successful": len(results),
                "failed": len(errors),
            },
            f,
            indent=2,
        )

    return {
        "total": len(audio_files),
        "successful": len(results),
        "failed": len(errors),
        "report": report_path,
    }


def batch_separate(
    audio_files: list[Path],
    output_dir: Path,
    model: str,
) -> dict[str, Any]:
    """
    Batch separate audio files into stems.

    Parameters
    ----------
    audio_files
        List of audio files to separate.
    output_dir
        Directory to save stems.
    model
        Demucs model to use.

    Returns
    -------
    dict
        Summary statistics.
    """
    config = SeparationConfig(
        model=DemucsModel(model),
        device="auto",
    )

    separator = StemSeparator(config=config)

    successful = 0
    failed = 0
    errors = []

    print(f"\nSeparating {len(audio_files)} files using {model}...\n")

    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Separating: {audio_file.name}")

        # Create output directory for this file
        file_output_dir = output_dir / audio_file.stem

        try:
            result = separator.separate(
                audio_path=audio_file,
                output_dir=file_output_dir,
            )

            print(f"  Extracted {len(result.stems)} stems in {result.processing_time_seconds:.1f}s")
            successful += 1

        except Exception as e:
            logger.error(f"  Failed: {e}")
            errors.append((audio_file.name, str(e)))
            failed += 1

    return {
        "total": len(audio_files),
        "successful": successful,
        "failed": failed,
        "output_dir": output_dir,
        "errors": errors,
    }


def batch_effects(
    audio_files: list[Path],
    output_dir: Path,
    preset: str,
) -> dict[str, Any]:
    """
    Batch apply effects to audio files.

    Parameters
    ----------
    audio_files
        List of audio files to process.
    output_dir
        Directory to save processed files.
    preset
        Effects preset to apply.

    Returns
    -------
    dict
        Summary statistics.
    """
    # Create effects chain based on preset
    chain = EffectsChain()

    if preset == "mastering":
        chain.add(HighpassConfig(cutoff_frequency_hz=30.0))
        chain.add(CompressorConfig(threshold_db=-18.0, ratio=2.5))
        chain.add(LimiterConfig(threshold_db=-1.0))
    elif preset == "normalize":
        chain.add(GainConfig(gain_db=0.0))
    elif preset == "vocal":
        chain.add(HighpassConfig(cutoff_frequency_hz=80.0))
        chain.add(CompressorConfig(threshold_db=-20.0, ratio=4.0))
        chain.add(ReverbConfig(room_size=0.3, wet_level=0.15))
    else:
        logger.error(f"Unknown preset: {preset}")
        return {"error": f"Unknown preset: {preset}"}

    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0
    errors = []

    print(f"\nApplying '{preset}' effect to {len(audio_files)} files...\n")

    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")

        output_path = output_dir / f"{audio_file.stem}_processed{audio_file.suffix}"

        try:
            chain.process(
                input_path=audio_file,
                output_path=output_path,
            )

            print(f"  Saved to: {output_path.name}")
            successful += 1

        except Exception as e:
            logger.error(f"  Failed: {e}")
            errors.append((audio_file.name, str(e)))
            failed += 1

    return {
        "total": len(audio_files),
        "successful": successful,
        "failed": failed,
        "output_dir": output_dir,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch process multiple audio files with SoundLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all files in a directory
  %(prog)s audio_dir/ --mode analyze

  # Separate all files
  %(prog)s songs/ --mode separate -o stems/ --model htdemucs_ft

  # Apply effects to all files
  %(prog)s tracks/ --mode effects -o processed/ --preset mastering

Processing Modes:
  analyze   - Comprehensive audio analysis (BPM, key, loudness, etc.)
  separate  - Stem separation (vocals, drums, bass, other)
  effects   - Apply effects chain

Effect Presets:
  mastering - Professional mastering chain
  normalize - Normalize audio levels
  vocal     - Vocal processing
        """,
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing audio files",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["analyze", "separate", "effects"],
        help="Processing mode",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (default: input_dir/output)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="htdemucs",
        choices=[m.value for m in DemucsModel],
        help="Demucs model for separation (default: htdemucs)",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="mastering",
        choices=["mastering", "normalize", "vocal"],
        help="Effects preset (default: mastering)",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for audio files recursively",
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

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1

    if not args.input_dir.is_dir():
        logger.error(f"Input path is not a directory: {args.input_dir}")
        return 1

    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = args.input_dir / "output"

    # Find audio files
    logger.info(f"Scanning for audio files in: {args.input_dir}")
    audio_files = find_audio_files(args.input_dir)

    if not audio_files:
        logger.error(f"No audio files found in: {args.input_dir}")
        logger.info(f"Supported formats: {', '.join(AUDIO_EXTENSIONS)}")
        return 1

    # Print header
    print(f"\n{'=' * 70}")
    print("SoundLab Batch Processing")
    print(f"{'=' * 70}")
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mode:             {args.mode}")
    print(f"Files found:      {len(audio_files)}")
    if args.mode == "separate":
        print(f"Model:            {args.model}")
    elif args.mode == "effects":
        print(f"Preset:           {args.preset}")
    print(f"{'=' * 70}")

    try:
        start_time = time.time()

        # Process based on mode
        if args.mode == "analyze":
            results = batch_analyze(audio_files, output_dir)
        elif args.mode == "separate":
            results = batch_separate(audio_files, output_dir, args.model)
        elif args.mode == "effects":
            results = batch_effects(audio_files, output_dir, args.preset)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1

        elapsed = time.time() - start_time

        # Print summary
        print(f"\n{'=' * 70}")
        print("Batch Processing Complete!")
        print(f"{'=' * 70}")
        print(f"Total files:      {results['total']}")
        print(f"Successful:       {results['successful']}")
        print(f"Failed:           {results['failed']}")
        print(f"Processing time:  {elapsed:.2f}s")
        print(f"Avg time/file:    {elapsed / results['total']:.2f}s")

        if args.mode == "analyze" and "report" in results:
            print(f"Analysis report:  {results['report']}")
        elif "output_dir" in results:
            print(f"Output directory: {results['output_dir']}")

        # Print errors if any
        if results.get("errors"):
            print(f"\nErrors ({len(results['errors'])}):")
            for filename, error in results["errors"][:5]:  # Show first 5 errors
                print(f"  {filename}: {error}")
            if len(results["errors"]) > 5:
                print(f"  ... and {len(results['errors']) - 5} more")

        print(f"{'=' * 70}\n")

        return 0 if results["failed"] == 0 else 1

    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        if args.verbose:
            logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
