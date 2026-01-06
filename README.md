# SoundLab

[![CI Status](https://img.shields.io/github/actions/workflow/status/wyattwalsh/soundlab/ci.yml?branch=main&label=CI&logo=github)](https://github.com/wyattwalsh/soundlab/actions)
[![PyPI Version](https://img.shields.io/pypi/v/soundlab?logo=pypi&logoColor=white)](https://pypi.org/project/soundlab/)
[![Coverage](https://img.shields.io/codecov/c/github/wyattwalsh/soundlab?logo=codecov)](https://codecov.io/gh/wyattwalsh/soundlab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Production-ready music processing platform with stem separation, audio-to-MIDI transcription, effects processing, audio analysis, and voice generation capabilities. Built for researchers, music producers, and audio engineers.

## Features

### Core Audio Processing

- **Stem Separation** - Isolate vocals, drums, bass, and other instruments using state-of-the-art Demucs models (htdemucs, htdemucs_ft, htdemucs_6s)
- **Audio-to-MIDI Transcription** - Convert polyphonic audio to MIDI using Spotify's Basic Pitch with customizable onset/frame thresholds
- **Audio Analysis** - Extract musical features including BPM, key detection (Krumhansl-Schmuckler), loudness (LUFS), spectral features, and onset detection
- **Effects Processing** - Apply professional audio effects chains with dynamics (compressor, limiter, gate), EQ, time-based effects (reverb, delay, chorus), and creative processing

### Voice Generation (Optional)

- **Text-to-Speech** - Generate natural speech in 18+ languages using XTTS-v2
- **Voice Cloning** - Clone voices from 6-30 second audio samples
- **Singing Voice Conversion** - Transform vocals using RVC (Retrieval-based Voice Conversion)

### Additional Capabilities

- **Type-Safe** - Full type hints with Pydantic validation for all configurations
- **GPU Accelerated** - Automatic GPU detection with intelligent memory management and CPU fallback
- **Robust Error Handling** - Comprehensive exception hierarchy with automatic retry logic
- **Progress Tracking** - Real-time progress callbacks for long-running operations
- **Multiple Formats** - Support for WAV, MP3, FLAC, OGG, AIFF, M4A

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) provides fast, reliable dependency management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/wyattwalsh/soundlab.git
cd soundlab

# Sync dependencies
uv sync

# Or sync with optional features
uv sync --extra voice --extra notebook
```

### Using pip

```bash
# Basic installation
pip install soundlab

# With voice generation support
pip install soundlab[voice]

# With Gradio notebook interface
pip install soundlab[notebook]

# Install everything
pip install soundlab[all]
```

### Development Installation

```bash
# Clone and install in development mode
git clone https://github.com/wyattwalsh/soundlab.git
cd soundlab
uv sync --dev

# Run tests
uv run pytest tests/

# Lint and format
uv run ruff check .
uv run ruff format .
```

## Quick Start

### Stem Separation

Extract individual instrument stems from a mixed audio file:

```python
from soundlab.separation import StemSeparator, SeparationConfig, DemucsModel

# Configure separation
config = SeparationConfig(
    model=DemucsModel.HTDEMUCS_FT,  # Best quality model
    segment_length=7.8,              # Segment size for memory efficiency
    overlap=0.25,                    # Overlap between segments
    int24=True                       # 24-bit output quality
)

# Create separator and process
separator = StemSeparator(config)
result = separator.separate(
    audio_path="song.mp3",
    output_dir="output/stems"
)

# Access separated stems
print(f"Vocals: {result.vocals}")
print(f"Drums: {result.stems['drums']}")
print(f"Bass: {result.stems['bass']}")
print(f"Other: {result.stems['other']}")
print(f"Processing time: {result.processing_time_seconds:.1f}s")
```

### MIDI Transcription

Convert audio to MIDI with piano roll visualization:

```python
from soundlab.transcription import MIDITranscriber, TranscriptionConfig

# Configure transcription
config = TranscriptionConfig(
    onset_threshold=0.5,      # Note onset sensitivity (0.1-0.9)
    frame_threshold=0.3,      # Frame activation threshold
    minimum_note_length=0.058 # Minimum note duration in seconds
)

# Transcribe audio
transcriber = MIDITranscriber(config)
result = transcriber.transcribe(
    audio_path="melody.wav",
    output_path="output/melody.mid"
)

# Get transcription details
print(f"MIDI file: {result.midi_path}")
print(f"Notes detected: {len(result.notes)}")
print(f"Pitch range: {result.pitch_range[0]}-{result.pitch_range[1]} Hz")

# Generate piano roll visualization
result.save_piano_roll("output/piano_roll.png")
```

### Audio Analysis

Analyze musical features of an audio file:

```python
from soundlab.analysis import analyze_audio, AudioAnalyzer

# Perform comprehensive analysis
result = analyze_audio("song.mp3")

# BPM detection
print(f"Tempo: {result.tempo.bpm:.1f} BPM (confidence: {result.tempo.confidence:.2f})")

# Key detection
print(f"Key: {result.key.name}")  # e.g., "A minor"
print(f"Camelot: {result.key.camelot}")  # e.g., "8A" for DJ mixing

# Loudness analysis
print(f"Integrated loudness: {result.loudness.lufs:.1f} LUFS")
print(f"Dynamic range: {result.loudness.dynamic_range:.1f} dB")

# Spectral features
print(f"Spectral centroid: {result.spectral.centroid_mean:.1f} Hz")
print(f"Brightness: {result.spectral.brightness:.2f}")

# Onset detection
print(f"Onsets detected: {len(result.onsets.timestamps)}")
```

### Effects Chain

Apply professional audio effects:

```python
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig
from soundlab.effects.eq import HighPassFilterConfig
from soundlab.effects.time_based import ReverbConfig

# Build effects chain (fluent API)
chain = (
    EffectsChain()
    .add(HighPassFilterConfig(cutoff_frequency_hz=80))
    .add(CompressorConfig(
        threshold_db=-20,
        ratio=4.0,
        attack_ms=5.0,
        release_ms=100.0
    ))
    .add(ReverbConfig(
        room_size=0.5,
        damping=0.5,
        wet_level=0.3,
        dry_level=0.7
    ))
)

# Process audio file
output_path = chain.process(
    input_path="dry_vocals.wav",
    output_path="processed_vocals.wav"
)

print(f"Processed: {output_path}")
print(f"Effects applied: {len(chain.effects)}")
```

### Voice Generation (Optional)

Generate speech with voice cloning:

```python
from soundlab.voice import TextToSpeech, TTSConfig

# Configure TTS with voice cloning
config = TTSConfig(
    language="en",
    speaker_wav="voice_sample.wav",  # 6-30 second reference audio
)

# Generate speech
tts = TextToSpeech(config)
result = tts.synthesize(
    text="Welcome to SoundLab, the production-ready music processing platform.",
    output_path="output/speech.wav"
)

print(f"Generated: {result.audio_path}")
print(f"Duration: {result.duration_seconds:.1f}s")
```

## Architecture

SoundLab is designed as a modular Python package with clear separation of concerns:

```
soundlab/
├── core/           # Core data models and exceptions
├── separation/     # Stem separation (Demucs)
├── transcription/  # Audio-to-MIDI (Basic Pitch)
├── effects/        # Audio effects (Pedalboard)
├── analysis/       # Audio analysis (librosa)
├── voice/          # Voice generation (XTTS-v2, RVC)
├── io/             # Audio/MIDI I/O utilities
└── utils/          # GPU management, logging, retry logic
```

## Documentation

Full documentation is available at: [https://github.com/wyattwalsh/soundlab#readme](https://github.com/wyattwalsh/soundlab#readme)

- [API Reference](https://github.com/wyattwalsh/soundlab/tree/main/docs/api)
- [User Guide](https://github.com/wyattwalsh/soundlab/tree/main/docs/guides)
- [Examples](https://github.com/wyattwalsh/soundlab/tree/main/notebooks/examples)
- [Contributing Guide](https://github.com/wyattwalsh/soundlab/blob/main/CONTRIBUTING.md)

## Google Colab

Try SoundLab in your browser with our interactive Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wyattwalsh/soundlab/blob/main/notebooks/soundlab_studio.ipynb)

The notebook provides a Gradio interface for all features with no local installation required.

## Requirements

- Python 3.12 or higher
- Optional: CUDA-capable GPU for accelerated processing (CPU fallback available)
- Operating System: Linux, macOS, Windows

## Performance

Typical processing times on T4 GPU (3-minute song):

| Operation | Time |
|-----------|------|
| Stem separation (htdemucs_ft) | ~3 min |
| MIDI transcription | ~30 sec |
| Audio analysis (all features) | ~10 sec |
| Effects chain (10 effects) | ~5 sec |

## Technology Stack

- **Stem Separation**: [Demucs](https://github.com/facebookresearch/demucs) 4.0+
- **MIDI Transcription**: [Basic Pitch](https://github.com/spotify/basic-pitch) 0.3+
- **Audio Effects**: [Pedalboard](https://github.com/spotify/pedalboard) 0.9+
- **Audio Analysis**: [librosa](https://github.com/librosa/librosa) 0.10+
- **Voice Generation**: [Coqui TTS](https://github.com/coqui-ai/TTS) 0.22+
- **Type Safety**: [Pydantic](https://github.com/pydantic/pydantic) 2.7+

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

**Wyatt Walsh**

- GitHub: [@wyattwalsh](https://github.com/wyattwalsh)
- Repository: [github.com/wyattwalsh/soundlab](https://github.com/wyattwalsh/soundlab)

## Citation

If you use SoundLab in your research, please cite:

```bibtex
@software{walsh2026soundlab,
  author = {Walsh, Wyatt},
  title = {SoundLab: Production-Ready Music Processing Platform},
  year = {2026},
  url = {https://github.com/wyattwalsh/soundlab},
  version = {0.1.0}
}
```

## Acknowledgments

SoundLab builds upon excellent open-source projects:

- [Demucs](https://github.com/facebookresearch/demucs) by Meta Research for state-of-the-art source separation
- [Basic Pitch](https://github.com/spotify/basic-pitch) by Spotify for polyphonic pitch detection
- [Pedalboard](https://github.com/spotify/pedalboard) by Spotify for high-quality audio effects
- [librosa](https://github.com/librosa/librosa) for comprehensive audio analysis
- [Coqui TTS](https://github.com/coqui-ai/TTS) for neural text-to-speech

## Support

- Issues: [GitHub Issues](https://github.com/wyattwalsh/soundlab/issues)
- Discussions: [GitHub Discussions](https://github.com/wyattwalsh/soundlab/discussions)

---

Made with ❤️ for the audio ML community
