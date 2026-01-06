"""Generate audio fixtures for testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


FIXTURES_DIR = Path(__file__).parent


def generate_sine_440hz_3s() -> None:
    """Generate a 440Hz sine wave, 3 seconds, 44100Hz."""
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    output_path = FIXTURES_DIR / "sine_440hz_3s.wav"
    sf.write(output_path, audio, sr)
    print(f"Generated: {output_path}")


def generate_silence_1s() -> None:
    """Generate 1 second of silence."""
    sr = 44100
    audio = np.zeros(sr, dtype=np.float32)

    output_path = FIXTURES_DIR / "silence_1s.wav"
    sf.write(output_path, audio, sr)
    print(f"Generated: {output_path}")


def generate_stereo_test() -> None:
    """Generate a stereo test file with different tones per channel."""
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Left channel: 440Hz, Right channel: 880Hz
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)

    stereo = np.stack([left, right], axis=1)

    output_path = FIXTURES_DIR / "stereo_test_2s.wav"
    sf.write(output_path, stereo, sr)
    print(f"Generated: {output_path}")


def generate_music_like() -> None:
    """Generate a music-like signal with multiple frequencies and beats."""
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Base frequencies (A minor chord: A, C, E)
    a = 0.2 * np.sin(2 * np.pi * 220 * t)  # A3
    c = 0.2 * np.sin(2 * np.pi * 261.63 * t)  # C4
    e = 0.2 * np.sin(2 * np.pi * 329.63 * t)  # E4

    # Add some rhythm (amplitude modulation at ~2Hz)
    rhythm = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)

    audio = (a + c + e) * rhythm

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    output_path = FIXTURES_DIR / "music_like_5s.wav"
    sf.write(output_path, audio.astype(np.float32), sr)
    print(f"Generated: {output_path}")


def generate_all_fixtures() -> None:
    """Generate all test fixtures."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    generate_sine_440hz_3s()
    generate_silence_1s()
    generate_stereo_test()
    generate_music_like()

    # Create a .gitkeep if fixtures don't exist
    gitkeep = FIXTURES_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

    print("\nAll fixtures generated successfully!")


if __name__ == "__main__":
    generate_all_fixtures()
