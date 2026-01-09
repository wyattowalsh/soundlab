#!/usr/bin/env python3
"""Download and cache SoundLab models.

This script pre-downloads models used by SoundLab to avoid runtime delays.
Supports Demucs stem separation models and optionally XTTS voice models.

Usage:
    python scripts/download_models.py                    # Download Demucs models
    python scripts/download_models.py --all              # Download all models
    python scripts/download_models.py --demucs htdemucs  # Specific Demucs model
    python scripts/download_models.py --xtts             # Download XTTS model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_demucs_models(models: list[str] | None = None) -> None:
    """Download Demucs models to cache.

    Parameters
    ----------
    models
        List of model names to download. If None, downloads default models.
    """
    try:
        import torch
        from demucs.pretrained import get_model
    except ImportError:
        print("❌ Demucs not installed. Install with: pip install demucs")
        sys.exit(1)

    default_models = [
        "htdemucs",
        "htdemucs_ft",
        "htdemucs_6s",
        "mdx_extra",
        "mdx_extra_q",
    ]

    models_to_download = models or default_models

    print("🎚️ Downloading Demucs models...")
    print(f"   Cache directory: {torch.hub.get_dir()}")
    print()

    for model_name in models_to_download:
        try:
            print(f"   Downloading {model_name}...", end=" ", flush=True)
            model = get_model(model_name)
            print(f"✅ ({model.sources})")
        except Exception as e:
            print(f"❌ Failed: {e}")

    print()
    print("✅ Demucs models downloaded")


def download_xtts_model(cache_dir: Path | None = None) -> None:  # noqa: ARG001
    """Download XTTS-v2 model to cache.

    Parameters
    ----------
    cache_dir
        Optional custom cache directory. Defaults to HuggingFace cache.
        (Reserved for future use)
    """
    try:
        from TTS.api import TTS
    except ImportError:
        print("❌ TTS not installed. Install with: pip install TTS")
        print("   Or install soundlab with voice extras: pip install soundlab[voice]")
        sys.exit(1)

    print("🗣️ Downloading XTTS-v2 model...")
    print("   This may take several minutes on first download (~1.8GB)")
    print()

    try:
        # Initialize TTS which triggers model download
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        print(f"   Model path: {tts.model_path}")
        print()
        print("✅ XTTS-v2 model downloaded")
    except Exception as e:
        print(f"❌ Failed to download XTTS model: {e}")
        sys.exit(1)


def download_basic_pitch_model() -> None:
    """Download Basic Pitch model to cache."""
    try:
        from basic_pitch.inference import predict
    except ImportError:
        print("❌ Basic Pitch not installed. Install with: pip install basic-pitch")
        sys.exit(1)

    print("🎹 Downloading Basic Pitch model...")

    try:
        # Basic Pitch downloads on first import/use
        # Create a small test to trigger download
        import tempfile

        import numpy as np
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Create 1 second of silence
            sr = 22050
            silence = np.zeros(sr, dtype=np.float32)
            sf.write(f.name, silence, sr)

            # Run prediction to trigger model download
            _ = predict(f.name)

            # Cleanup
            Path(f.name).unlink()

        print("✅ Basic Pitch model downloaded")
    except Exception as e:
        print(f"❌ Failed to download Basic Pitch model: {e}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and cache SoundLab models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                         # Download default Demucs models
    %(prog)s --all                   # Download all models (Demucs + XTTS + Basic Pitch)
    %(prog)s --demucs htdemucs_ft    # Download specific Demucs model
    %(prog)s --xtts                  # Download XTTS voice model
    %(prog)s --basic-pitch           # Download Basic Pitch transcription model
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )
    parser.add_argument(
        "--demucs",
        nargs="*",
        metavar="MODEL",
        help="Download Demucs models (default: all). Specify models: htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra, mdx_extra_q",
    )
    parser.add_argument(
        "--xtts",
        action="store_true",
        help="Download XTTS-v2 voice model (~1.8GB)",
    )
    parser.add_argument(
        "--basic-pitch",
        action="store_true",
        help="Download Basic Pitch transcription model",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Custom cache directory for models",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SoundLab Model Downloader")
    print("=" * 60)
    print()

    # Determine what to download
    download_demucs = (
        args.all or args.demucs is not None or (not args.xtts and not args.basic_pitch)
    )
    download_xtts = args.all or args.xtts
    download_bp = args.all or args.basic_pitch

    if download_demucs:
        models = args.demucs if args.demucs else None
        download_demucs_models(models)
        print()

    if download_bp:
        download_basic_pitch_model()
        print()

    if download_xtts:
        download_xtts_model(args.cache_dir)
        print()

    print("=" * 60)
    print("✅ Model download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
