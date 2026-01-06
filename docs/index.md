# SoundLab

<div align="center" markdown>

**Production-ready music processing platform with stem separation, audio-to-MIDI transcription, effects processing, audio analysis, and voice generation capabilities.**

[![PyPI Version](https://img.shields.io/pypi/v/soundlab?logo=pypi&logoColor=white)](https://pypi.org/project/soundlab/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Status](https://img.shields.io/github/actions/workflow/status/wyattwalsh/soundlab/ci.yml?branch=main&label=CI&logo=github)](https://github.com/wyattwalsh/soundlab/actions)

[Get Started](getting-started/quickstart.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/wyattwalsh/soundlab){ .md-button }
[Try in Colab](https://colab.research.google.com/github/wyattwalsh/soundlab/blob/main/notebooks/soundlab_studio.ipynb){ .md-button }

</div>

---

## Features

<div class="grid cards" markdown>

-   :material-music-box-multiple: **Stem Separation**

    ---

    Isolate vocals, drums, bass, and other instruments using state-of-the-art Demucs models with GPU acceleration.

    [:octicons-arrow-right-24: Learn more](user-guide/separation.md)

-   :material-piano: **MIDI Transcription**

    ---

    Convert polyphonic audio to MIDI using Spotify's Basic Pitch with customizable onset and frame thresholds.

    [:octicons-arrow-right-24: Learn more](user-guide/transcription.md)

-   :material-waveform: **Audio Analysis**

    ---

    Extract musical features including BPM, key detection, loudness (LUFS), spectral features, and onset detection.

    [:octicons-arrow-right-24: Learn more](user-guide/analysis.md)

-   :material-tune: **Effects Processing**

    ---

    Apply professional audio effects chains with dynamics, EQ, time-based effects, and creative processing.

    [:octicons-arrow-right-24: Learn more](user-guide/effects.md)

-   :material-microphone-variant: **Voice Generation**

    ---

    Generate natural speech in 18+ languages with voice cloning and singing voice conversion capabilities.

    [:octicons-arrow-right-24: Learn more](user-guide/voice.md)

-   :material-speedometer: **GPU Accelerated**

    ---

    Automatic GPU detection with intelligent memory management and CPU fallback for maximum performance.

    [:octicons-arrow-right-24: View API](api/index.md)

</div>

---

## Quick Installation

=== "pip (Recommended)"

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

=== "uv (Fast)"

    ```bash
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Clone and sync
    git clone https://github.com/wyattwalsh/soundlab.git
    cd soundlab
    uv sync

    # With optional features
    uv sync --extra voice --extra notebook
    ```

=== "Development"

    ```bash
    # Clone repository
    git clone https://github.com/wyattwalsh/soundlab.git
    cd soundlab

    # Install with dev dependencies
    uv sync --dev

    # Run tests
    uv run pytest tests/
    ```

!!! tip "System Requirements"
    - Python 3.12 or higher
    - Optional: CUDA-capable GPU for accelerated processing (CPU fallback available)
    - Operating System: Linux, macOS, Windows

---

## Quick Start

### :material-music-box-multiple: Stem Separation

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
print(f"Processing time: {result.processing_time_seconds:.1f}s")
```

[:octicons-arrow-right-24: Full Separation Guide](user-guide/separation.md)

---

### :material-piano: MIDI Transcription

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

[:octicons-arrow-right-24: Full Transcription Guide](user-guide/transcription.md)

---

### :material-waveform: Audio Analysis

Analyze musical features of an audio file:

```python
from soundlab.analysis import analyze_audio

# Perform comprehensive analysis
result = analyze_audio("song.mp3")

# BPM detection
print(f"Tempo: {result.tempo.bpm:.1f} BPM")
print(f"Confidence: {result.tempo.confidence:.2f}")

# Key detection
print(f"Key: {result.key.name}")  # e.g., "A minor"
print(f"Camelot: {result.key.camelot}")  # e.g., "8A" for DJ mixing

# Loudness analysis
print(f"Integrated loudness: {result.loudness.lufs:.1f} LUFS")
print(f"Dynamic range: {result.loudness.dynamic_range:.1f} dB")

# Spectral features
print(f"Spectral centroid: {result.spectral.centroid_mean:.1f} Hz")
print(f"Brightness: {result.spectral.brightness:.2f}")
```

[:octicons-arrow-right-24: Full Analysis Guide](user-guide/analysis.md)

---

### :material-tune: Effects Chain

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

[:octicons-arrow-right-24: Full Effects Guide](user-guide/effects.md)

---

### :material-microphone-variant: Voice Generation

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

[:octicons-arrow-right-24: Full Voice Guide](user-guide/voice.md)

---

## Documentation Sections

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**

    ---

    New to SoundLab? Start here for installation instructions and your first project.

    [:octicons-arrow-right-24: Get Started](getting-started/index.md)

-   :material-book-open-variant: **User Guide**

    ---

    In-depth guides for each feature with examples and best practices.

    [:octicons-arrow-right-24: User Guide](user-guide/index.md)

-   :material-code-braces: **API Reference**

    ---

    Complete API documentation with type signatures and detailed descriptions.

    [:octicons-arrow-right-24: API Reference](api/index.md)

-   :material-notebook: **Examples**

    ---

    Real-world examples and integration patterns for common workflows.

    [:octicons-arrow-right-24: Examples](examples/index.md)

-   :material-github: **Contributing**

    ---

    Learn how to contribute to SoundLab with our development guide.

    [:octicons-arrow-right-24: Contributing](contributing/index.md)

-   :material-history: **Changelog**

    ---

    Track the latest features, improvements, and bug fixes.

    [:octicons-arrow-right-24: Changelog](changelog.md)

</div>

---

## Technology Stack

SoundLab is built on industry-leading open-source libraries:

<div class="grid" markdown>

- :simple-python: **[Demucs](https://github.com/facebookresearch/demucs)** - State-of-the-art source separation by Meta Research
- :material-music-note: **[Basic Pitch](https://github.com/spotify/basic-pitch)** - Polyphonic pitch detection by Spotify
- :material-equalizer: **[Pedalboard](https://github.com/spotify/pedalboard)** - High-quality audio effects by Spotify
- :material-chart-line: **[librosa](https://github.com/librosa/librosa)** - Comprehensive audio analysis toolkit
- :material-account-voice: **[Coqui TTS](https://github.com/coqui-ai/TTS)** - Neural text-to-speech with voice cloning
- :material-check-circle: **[Pydantic](https://github.com/pydantic/pydantic)** - Type-safe data validation

</div>

---

## Performance

Typical processing times on T4 GPU (3-minute song):

| Operation | Time | Notes |
|-----------|------|-------|
| Stem separation (htdemucs_ft) | ~3 min | Best quality model |
| MIDI transcription | ~30 sec | Polyphonic audio |
| Audio analysis (all features) | ~10 sec | BPM, key, loudness, spectral |
| Effects chain (10 effects) | ~5 sec | Real-time capable |

!!! note "Hardware Acceleration"
    SoundLab automatically detects and utilizes available GPU hardware. CPU fallback is always available for systems without CUDA support.

---

## Interactive Notebook

Try SoundLab directly in your browser without any installation:

<div align="center" markdown>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wyattwalsh/soundlab/blob/main/notebooks/soundlab_studio.ipynb)

The notebook provides a **Gradio interface** for all features with no local setup required.

</div>

---

## Resources

<div class="grid cards" markdown>

-   :fontawesome-brands-github: **GitHub Repository**

    ---

    View source code, report issues, and contribute to development.

    [:octicons-arrow-right-24: Visit Repository](https://github.com/wyattwalsh/soundlab)

-   :fontawesome-brands-python: **PyPI Package**

    ---

    Install SoundLab from the Python Package Index.

    [:octicons-arrow-right-24: View on PyPI](https://pypi.org/project/soundlab/)

-   :material-bug: **Issue Tracker**

    ---

    Report bugs, request features, or ask questions.

    [:octicons-arrow-right-24: Open Issue](https://github.com/wyattwalsh/soundlab/issues)

-   :material-chat: **Discussions**

    ---

    Join the community and share your projects.

    [:octicons-arrow-right-24: Start Discussion](https://github.com/wyattwalsh/soundlab/discussions)

</div>

---

## What's Next?

<div class="grid cards" markdown>

-   :material-download: **Install SoundLab**

    ---

    Get started with a simple pip install command.

    ```bash
    pip install soundlab
    ```

    [:octicons-arrow-right-24: Installation Guide](getting-started/installation.md)

-   :material-play-circle: **Quick Start Tutorial**

    ---

    Build your first audio processing pipeline in minutes.

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-application-brackets: **Explore Examples**

    ---

    Learn from real-world examples and integration patterns.

    [:octicons-arrow-right-24: View Examples](examples/index.md)

-   :material-api: **API Documentation**

    ---

    Dive deep into the complete API reference.

    [:octicons-arrow-right-24: API Reference](api/index.md)

</div>

---

<div align="center" markdown>

**Built with :material-heart: for the audio ML community**

[GitHub](https://github.com/wyattwalsh/soundlab) •
[PyPI](https://pypi.org/project/soundlab/) •
[Documentation](https://wyattowalsh.github.io/soundlab)

Copyright © 2026 [Wyatt Walsh](https://github.com/wyattwalsh) • [MIT License](https://opensource.org/licenses/MIT)

</div>
