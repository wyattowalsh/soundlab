#!/usr/bin/env python3
"""
MIDI Transcription Example

This script demonstrates how to use SoundLab to transcribe audio to MIDI
using the Basic Pitch model.

Usage:
    python examples/transcribe_to_midi.py input.mp3
    python examples/transcribe_to_midi.py song.wav -o output.mid --onset 0.6
    python examples/transcribe_to_midi.py track.flac --min-note 100 --save-pianoroll
"""

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

from soundlab.transcription import (
    MIDITranscriber,
    TranscriptionConfig,
    render_piano_roll,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio to MIDI using Basic Pitch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  %(prog)s input.mp3

  # Save to specific file
  %(prog)s input.mp3 -o output.mid

  # Adjust detection thresholds
  %(prog)s input.mp3 --onset 0.6 --frame 0.4

  # Filter short notes
  %(prog)s input.mp3 --min-note 100

  # Save piano roll visualization
  %(prog)s input.mp3 --save-pianoroll

Threshold Guidelines:
  - Higher onset threshold = fewer false positive notes
  - Lower frame threshold = more sustained notes detected
  - Minimum note length filters out very short artifacts
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
        help="Output MIDI file (default: input_name.mid)",
    )

    parser.add_argument(
        "--onset",
        "--onset-thresh",
        type=float,
        default=0.5,
        help="Onset threshold (0.1-0.9, higher = fewer notes) (default: 0.5)",
    )

    parser.add_argument(
        "--frame",
        "--frame-thresh",
        type=float,
        default=0.3,
        help="Frame threshold (0.1-0.9, lower = longer notes) (default: 0.3)",
    )

    parser.add_argument(
        "--min-note",
        "--minimum-note-length",
        type=float,
        default=58.0,
        help="Minimum note length in milliseconds (default: 58.0)",
    )

    parser.add_argument(
        "--min-freq",
        "--minimum-frequency",
        type=float,
        default=32.7,
        help="Minimum frequency in Hz (default: 32.7 = C1)",
    )

    parser.add_argument(
        "--max-freq",
        "--maximum-frequency",
        type=float,
        default=2093.0,
        help="Maximum frequency in Hz (default: 2093.0 = C7)",
    )

    parser.add_argument(
        "--no-melodia-trick",
        action="store_true",
        help="Disable melodia trick (viterbi decoding)",
    )

    parser.add_argument(
        "--pitch-bends",
        action="store_true",
        help="Include pitch bends in MIDI output",
    )

    parser.add_argument(
        "--save-pianoroll",
        action="store_true",
        help="Save piano roll visualization as PNG",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Processing device (default: auto)",
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

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.input.with_suffix(".mid")

    # Print configuration
    print(f"\n{'=' * 60}")
    print("SoundLab MIDI Transcription")
    print(f"{'=' * 60}")
    print(f"Input:          {args.input}")
    print(f"Output:         {output_path}")
    print(f"Onset thresh:   {args.onset}")
    print(f"Frame thresh:   {args.frame}")
    print(f"Min note len:   {args.min_note}ms")
    print(f"Freq range:     {args.min_freq:.1f} - {args.max_freq:.1f} Hz")
    print(f"Device:         {args.device}")
    print(f"{'=' * 60}\n")

    try:
        # Create configuration
        config = TranscriptionConfig(
            onset_thresh=args.onset,
            frame_thresh=args.frame,
            minimum_note_length=args.min_note,
            minimum_frequency=args.min_freq,
            maximum_frequency=args.max_freq,
            melodia_trick=not args.no_melodia_trick,
            include_pitch_bends=args.pitch_bends,
            device=args.device,
        )

        # Create transcriber
        transcriber = MIDITranscriber(config=config)

        # Transcribe
        logger.info("Starting transcription...")
        start_time = time.time()

        result = transcriber.transcribe(
            audio_path=args.input,
            output_path=output_path,
        )

        elapsed = time.time() - start_time

        # Print results
        print(f"\n{'=' * 60}")
        print("Transcription Complete!")
        print(f"{'=' * 60}")
        print(f"Processing time:  {elapsed:.2f}s")
        print(f"MIDI file:        {result.midi_path}")
        print(f"\nStatistics:")
        print(f"  Total notes:    {result.note_count}")
        print(f"  Duration:       {result.duration:.2f}s")

        if result.notes:
            pitch_min, pitch_max = result.pitch_range
            # Convert MIDI pitch to note name
            def pitch_to_name(pitch: int) -> str:
                notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                octave = (pitch // 12) - 1
                note = notes[pitch % 12]
                return f"{note}{octave}"

            print(f"  Pitch range:    {pitch_to_name(pitch_min)} - {pitch_to_name(pitch_max)}")
            print(f"  Avg velocity:   {result.average_velocity:.1f}")
            print(f"  Notes/second:   {result.note_count / result.duration:.2f}")

            # Show a few example notes
            print(f"\nFirst 5 notes:")
            for i, note in enumerate(result.notes[:5]):
                print(
                    f"  {i+1}. {note.pitch_name:4s} @ {note.start_time:.2f}s "
                    f"(duration: {note.duration_ms:.0f}ms, vel: {note.velocity})"
                )

        # Save piano roll if requested
        if args.save_pianoroll and result.notes:
            pianoroll_path = output_path.with_suffix(".png")
            logger.info(f"Rendering piano roll to {pianoroll_path}...")

            render_piano_roll(
                result.notes,
                output_path=pianoroll_path,
                title=f"Piano Roll: {args.input.name}",
            )

            print(f"\nPiano roll:       {pianoroll_path}")

        print(f"{'=' * 60}\n")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        if args.verbose:
            logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
