# User Guide Overview

Welcome to the SoundLab user guide! This comprehensive guide will help you master all features of SoundLab, from basic audio processing to advanced voice generation.

## What is SoundLab?

SoundLab is a production-ready music processing platform that provides state-of-the-art audio processing capabilities through a clean, type-safe Python API. Built on industry-leading open-source tools, SoundLab offers:

- **Stem Separation** - Isolate vocals, drums, bass, and other instruments
- **MIDI Transcription** - Convert audio to MIDI with high accuracy
- **Audio Analysis** - Extract tempo, key, loudness, and spectral features
- **Effects Processing** - Apply professional audio effects chains
- **Voice Generation** - Text-to-speech and voice cloning (optional)

## Guide Structure

This user guide is organized into focused sections for each major feature:

### Core Guides

<div class="grid cards" markdown>

-   :material-package-variant-closed: **[Installation Guide](installation.md)**

    ---

    Complete installation instructions for all platforms, including uv, pip, development setup, and GPU configuration.

    [:octicons-arrow-right-24: Get started](installation.md)

-   :material-rocket-launch: **[Quick Start](quickstart.md)**

    ---

    Jump right in with basic examples covering all major features. Perfect for getting familiar with the API.

    [:octicons-arrow-right-24: Quick start](quickstart.md)

</div>

### Feature Guides

<div class="grid cards" markdown>

-   :material-music-box-multiple: **[Stem Separation](separation.md)**

    ---

    Isolate individual instruments and vocals from mixed audio using state-of-the-art Demucs models.

    [:octicons-arrow-right-24: Learn more](separation.md)

-   :material-piano: **[MIDI Transcription](transcription.md)**

    ---

    Convert polyphonic audio to MIDI using Spotify's Basic Pitch with customizable parameters.

    [:octicons-arrow-right-24: Learn more](transcription.md)

-   :material-chart-line: **[Audio Analysis](analysis.md)**

    ---

    Extract musical features including BPM, key, loudness, and spectral characteristics.

    [:octicons-arrow-right-24: Learn more](analysis.md)

-   :material-knob: **[Effects Processing](effects.md)**

    ---

    Build professional effects chains with dynamics, EQ, reverb, delay, and creative effects.

    [:octicons-arrow-right-24: Learn more](effects.md)

-   :material-microphone: **[Voice Generation](voice.md)**

    ---

    Generate natural speech in 18+ languages with voice cloning and singing voice conversion.

    [:octicons-arrow-right-24: Learn more](voice.md)

</div>

## Prerequisites

Before starting, ensure you have:

- Python 3.12 or higher
- Basic understanding of Python programming
- (Optional) CUDA-capable GPU for faster processing

!!! tip "First Time Users"
    If you're new to SoundLab, we recommend starting with the [Installation Guide](installation.md) followed by the [Quick Start](quickstart.md) guide.

## Key Concepts

### Type Safety

SoundLab uses [Pydantic](https://docs.pydantic.dev/) for configuration validation, ensuring type safety and catching errors early:

```python
from soundlab.separation import SeparationConfig, DemucsModel

# Type-checked configuration
config = SeparationConfig(
    model=DemucsModel.HTDEMUCS_FT,
    segment_length=7.8,  # Must be between 1.0 and 30.0
    overlap=0.25,        # Must be between 0.1 and 0.9
)
```

### GPU Acceleration

SoundLab automatically detects and uses available GPUs with intelligent memory management:

```python
# Automatic GPU detection
config = SeparationConfig(device="auto")  # Default

# Force CPU (useful for testing)
config = SeparationConfig(device="cpu")

# Force specific GPU
config = SeparationConfig(device="cuda:0")
```

### Progress Tracking

Long-running operations support progress callbacks:

```python
from soundlab.separation import StemSeparator

def progress_callback(step: str, percent: float):
    print(f"{step}: {percent:.1f}%")

separator = StemSeparator()
result = separator.separate(
    audio_path="song.mp3",
    output_dir="output/",
    progress_callback=progress_callback
)
```

### Error Handling

SoundLab provides comprehensive exception handling with automatic retry logic:

```python
from soundlab.core import SoundLabError
from soundlab.separation import StemSeparator

try:
    separator = StemSeparator()
    result = separator.separate("song.mp3", "output/")
except SoundLabError as e:
    print(f"Processing failed: {e}")
    # Handle error appropriately
```

## Common Patterns

### Configuration Reuse

Create reusable configurations for consistent processing:

```python
from soundlab.separation import SeparationConfig, DemucsModel

# Define once
PRODUCTION_CONFIG = SeparationConfig(
    model=DemucsModel.HTDEMUCS_FT,
    segment_length=7.8,
    shifts=1,
    int24=True
)

# Reuse everywhere
separator = StemSeparator(PRODUCTION_CONFIG)
```

### Batch Processing

Process multiple files efficiently:

```python
from pathlib import Path
from soundlab.separation import StemSeparator

separator = StemSeparator()
audio_files = Path("input/").glob("*.mp3")

for audio_file in audio_files:
    result = separator.separate(
        audio_path=audio_file,
        output_dir=f"output/{audio_file.stem}"
    )
    print(f"Processed: {audio_file.name}")
```

### Pipeline Composition

Combine multiple processing steps:

```python
from soundlab.separation import StemSeparator
from soundlab.transcription import MIDITranscriber
from soundlab.analysis import analyze_audio

# Separate stems
separator = StemSeparator()
stems = separator.separate("song.mp3", "output/stems/")

# Transcribe vocals to MIDI
transcriber = MIDITranscriber()
midi = transcriber.transcribe(
    audio_path=stems.vocals,
    output_path="output/vocals.mid"
)

# Analyze drums
drums_analysis = analyze_audio(stems.drums)
print(f"Drum tempo: {drums_analysis.tempo.bpm:.1f} BPM")
```

## Getting Help

### Documentation Resources

- **API Reference** - Detailed API documentation for all modules
- **Examples** - Jupyter notebooks with real-world examples
- **Contributing Guide** - Guidelines for contributing to SoundLab

### Community Support

- **GitHub Issues** - Report bugs or request features
- **GitHub Discussions** - Ask questions and share projects
- **Stack Overflow** - Tag questions with `soundlab`

!!! question "Need Help?"
    If you encounter issues or have questions:

    1. Check the relevant feature guide
    2. Review the [common issues](#common-issues) section
    3. Search [GitHub issues](https://github.com/wyattwalsh/soundlab/issues)
    4. Ask in [GitHub Discussions](https://github.com/wyattwalsh/soundlab/discussions)

## Common Issues

### GPU Memory

If you encounter GPU out-of-memory errors:

```python
# Reduce segment length for separation
config = SeparationConfig(segment_length=5.0)  # Default: 7.8

# Or force CPU processing
config = SeparationConfig(device="cpu")
```

### Audio Format Issues

Ensure your audio files are in supported formats:

- **Supported**: WAV, MP3, FLAC, OGG, AIFF, M4A
- **Recommended**: WAV or FLAC for best quality

```python
from soundlab.io import load_audio, save_audio

# Convert to supported format
audio, sr = load_audio("input.opus")
save_audio("output.wav", audio, sr)
```

### Import Errors

Voice generation requires optional dependencies:

```bash
# Install voice generation support
pip install soundlab[voice]

# Or with uv
uv sync --extra voice
```

## Performance Tips

### Optimization Strategies

1. **Use GPU when available** - 10-20x faster than CPU
2. **Adjust segment length** - Balance memory vs. speed
3. **Enable shifts for quality** - Trades speed for better separation
4. **Batch similar operations** - Reuse models across files
5. **Use appropriate models** - `htdemucs` is faster, `htdemucs_ft` is better quality

### Benchmarks

Typical processing times on T4 GPU (3-minute song):

| Operation | Time | Notes |
|-----------|------|-------|
| Stem separation (htdemucs_ft) | ~3 min | Best quality |
| Stem separation (htdemucs) | ~2 min | Faster |
| MIDI transcription | ~30 sec | GPU accelerated |
| Audio analysis (all features) | ~10 sec | Mostly CPU |
| Effects chain (10 effects) | ~5 sec | Real-time capable |

## Next Steps

Ready to get started? Here's the recommended learning path:

1. **[Install SoundLab](installation.md)** - Get everything set up
2. **[Quick Start](quickstart.md)** - Try basic examples
3. **Choose your feature**:
    - Music producers: [Separation](separation.md) → [Effects](effects.md)
    - Developers: [Analysis](analysis.md) → [Transcription](transcription.md)
    - Content creators: [Voice](voice.md)

!!! success "Ready to Build?"
    Head over to the [Quick Start guide](quickstart.md) to start processing audio with SoundLab!

---

**Questions or feedback?** Open an issue on [GitHub](https://github.com/wyattwalsh/soundlab/issues) or start a [discussion](https://github.com/wyattwalsh/soundlab/discussions).
