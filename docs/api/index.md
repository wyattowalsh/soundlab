# API Reference

Welcome to the SoundLab API documentation. This reference provides detailed information about all modules, classes, and functions available in the SoundLab library.

## Overview

SoundLab is a production-ready music processing platform that provides a comprehensive suite of tools for audio manipulation, analysis, and generation. The API is designed to be intuitive, type-safe, and efficient, with support for both CPU and GPU processing.

### Key Features

- **Type-Safe**: Full type hints with Pydantic validation for all configurations
- **GPU Accelerated**: Automatic GPU detection with intelligent memory management
- **Robust Error Handling**: Comprehensive exception hierarchy with automatic retry logic
- **Progress Tracking**: Real-time progress callbacks for long-running operations
- **Multiple Formats**: Support for WAV, MP3, FLAC, OGG, AIFF, M4A

## Module Organization

SoundLab is organized into several specialized modules, each focusing on a specific aspect of audio processing:

### [Core](core.md)
Core data models, configuration, and exceptions that form the foundation of SoundLab.

**Key Components:**
- `AudioSegment` - Core audio representation
- `AudioMetadata` - Audio file metadata
- `SoundLabConfig` - Global configuration
- `SoundLabError` - Exception hierarchy

**Use Cases:**
- Working with audio data structures
- Configuring global settings
- Handling errors and exceptions

---

### [Separation](separation.md)
Stem separation capabilities using state-of-the-art Demucs models.

**Key Components:**
- `StemSeparator` - Main separation interface
- `SeparationConfig` - Configuration for separation
- `DemucsModel` - Available model presets

**Use Cases:**
- Extracting vocals from mixed audio
- Isolating individual instruments (drums, bass, other)
- Creating stems for remixing

---

### [Transcription](transcription.md)
Audio-to-MIDI transcription using Spotify's Basic Pitch.

**Key Components:**
- `MIDITranscriber` - Main transcription interface
- `TranscriptionConfig` - Configuration for transcription
- `TranscriptionResult` - Results with note information

**Use Cases:**
- Converting audio melodies to MIDI
- Creating piano roll visualizations
- Extracting musical notation from recordings

---

### [Analysis](analysis.md)
Comprehensive audio analysis for extracting musical features.

**Key Components:**
- `analyze_audio()` - Main analysis function
- `detect_tempo()` - BPM detection
- `detect_key()` - Musical key detection
- `measure_loudness()` - LUFS loudness measurement

**Use Cases:**
- BPM and tempo detection
- Key and scale identification (Krumhansl-Schmuckler)
- Loudness analysis (LUFS/LKFS)
- Spectral feature extraction
- Onset detection

---

### [Effects](effects.md)
Professional audio effects processing using Pedalboard.

**Key Components:**
- `EffectsChain` - Chainable effects processor
- Dynamics: `CompressorConfig`, `LimiterConfig`, `GateConfig`
- EQ: `HighPassFilterConfig`, `LowPassFilterConfig`, `PeakFilterConfig`
- Time-based: `ReverbConfig`, `DelayConfig`, `ChorusConfig`
- Creative: `DistortionConfig`, `PhaserConfig`, `BitcrusherConfig`

**Use Cases:**
- Building professional effects chains
- Mastering and mixing automation
- Creative sound design
- Real-time audio processing

---

### [Voice](voice.md)
Voice generation capabilities including text-to-speech and voice conversion.

**Key Components:**
- `TextToSpeech` - Neural TTS with voice cloning
- `SingingVoiceConverter` - RVC-based voice conversion
- `TTSConfig` - TTS configuration
- `SVCConfig` - Voice conversion configuration

**Use Cases:**
- Generating natural speech in 18+ languages
- Cloning voices from short samples
- Converting singing voices
- Creating voiceovers and narration

---

### [I/O](io.md)
Audio and MIDI file input/output utilities.

**Key Components:**
- `load_audio()` - Load audio files
- `save_audio()` - Save audio files
- `load_midi()` - Load MIDI files
- `save_midi()` - Save MIDI files
- `get_audio_metadata()` - Extract metadata

**Use Cases:**
- Reading and writing audio files
- Working with multiple audio formats
- Handling MIDI files
- Extracting file metadata

---

### [Utils](utils.md)
Utility functions for GPU management, logging, progress tracking, and retry logic.

**Key Components:**
- GPU utilities: `get_device()`, `get_gpu_memory()`
- Logging: `setup_logging()`, `get_logger()`
- Progress: `ProgressCallback`, `ProgressTracker`
- Retry: `retry_with_backoff()`, `RetryConfig`

**Use Cases:**
- Managing GPU resources
- Configuring logging
- Tracking operation progress
- Implementing robust error recovery

---

## Quick Start

```python
import soundlab

# Basic stem separation
from soundlab.separation import StemSeparator
separator = StemSeparator()
result = separator.separate("song.mp3", "output/")

# Audio analysis
from soundlab.analysis import analyze_audio
analysis = analyze_audio("song.mp3")
print(f"Tempo: {analysis.tempo.bpm} BPM")
print(f"Key: {analysis.key.name}")

# MIDI transcription
from soundlab.transcription import MIDITranscriber
transcriber = MIDITranscriber()
result = transcriber.transcribe("melody.wav", "output/melody.mid")

# Effects processing
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig

chain = EffectsChain().add(CompressorConfig(threshold_db=-20, ratio=4.0))
chain.process("input.wav", "output.wav")
```

## Installation

```bash
# Basic installation
pip install soundlab

# With voice generation support
pip install soundlab[voice]

# Development installation
git clone https://github.com/wyattwalsh/soundlab.git
cd soundlab
uv sync --dev
```

## Type Safety

All SoundLab APIs use comprehensive type hints and Pydantic models for validation:

```python
from soundlab.separation import SeparationConfig, DemucsModel

# Type-safe configuration with validation
config = SeparationConfig(
    model=DemucsModel.HTDEMUCS_FT,  # Enum ensures valid model
    segment_length=7.8,              # Validated range
    overlap=0.25,                    # 0.0 to 1.0
    int24=True                       # Boolean flag
)
```

## Error Handling

SoundLab provides a comprehensive exception hierarchy:

```python
from soundlab import SoundLabError
from soundlab.core import AudioProcessingError, ModelLoadError

try:
    result = separator.separate("song.mp3", "output/")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except AudioProcessingError as e:
    print(f"Processing failed: {e}")
except SoundLabError as e:
    print(f"General error: {e}")
```

## Progress Tracking

Monitor long-running operations with progress callbacks:

```python
from soundlab.utils import ProgressCallback

def progress_callback(progress: float, message: str):
    print(f"{message}: {progress:.1%}")

separator = StemSeparator()
result = separator.separate(
    "song.mp3",
    "output/",
    progress_callback=progress_callback
)
```

## GPU Support

SoundLab automatically detects and uses GPU when available:

```python
from soundlab.utils import get_device, get_gpu_memory

device = get_device()  # Returns 'cuda' or 'cpu'
print(f"Using device: {device}")

if device == 'cuda':
    memory = get_gpu_memory()
    print(f"GPU memory: {memory.used_gb:.1f}GB / {memory.total_gb:.1f}GB")
```

## Additional Resources

- [GitHub Repository](https://github.com/wyattwalsh/soundlab)
- [Examples](https://github.com/wyattwalsh/soundlab/tree/main/notebooks/examples)
- [Google Colab Notebook](https://colab.research.google.com/github/wyattwalsh/soundlab/blob/main/notebooks/soundlab_studio.ipynb)
- [Contributing Guide](https://github.com/wyattwalsh/soundlab/blob/main/CONTRIBUTING.md)

## License

SoundLab is released under the MIT License. See [LICENSE](https://github.com/wyattwalsh/soundlab/blob/main/LICENSE) for details.
