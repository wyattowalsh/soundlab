# Separation

Stem separation capabilities using state-of-the-art Demucs models.

## Overview

The `soundlab.separation` module provides high-quality audio source separation, allowing you to extract individual instrument stems from mixed audio files. It uses Meta's Demucs models, which represent the current state-of-the-art in music source separation.

## Features

- **Multiple Models**: Support for htdemucs, htdemucs_ft, and htdemucs_6s variants
- **GPU Acceleration**: Automatic GPU utilization with fallback to CPU
- **Memory Efficient**: Segment-based processing for handling long audio files
- **High Quality**: 24-bit output with minimal artifacts
- **Progress Tracking**: Real-time callbacks for monitoring separation progress

## Available Models

- **htdemucs**: Baseline model (4 stems: vocals, drums, bass, other)
- **htdemucs_ft**: Fine-tuned version with improved quality (recommended)
- **htdemucs_6s**: 6-stem model (vocals, drums, bass, other, guitar, piano)

## Key Components

### StemSeparator
The main interface for stem separation. Handles model loading, audio processing, and output generation.

### SeparationConfig
Configuration options including:
- Model selection
- Segment length and overlap
- Output format and bit depth
- Device selection

### SeparationResult
Results containing:
- Paths to separated stem files
- Processing time and metadata
- Quality metrics

## Usage Examples

### Basic Separation

```python
from soundlab.separation import StemSeparator

# Create separator with defaults
separator = StemSeparator()

# Separate audio
result = separator.separate(
    audio_path="song.mp3",
    output_dir="output/stems"
)

# Access results
print(f"Vocals: {result.vocals}")
print(f"Drums: {result.stems['drums']}")
print(f"Bass: {result.stems['bass']}")
print(f"Other: {result.stems['other']}")
print(f"Processing time: {result.processing_time_seconds:.1f}s")
```

### Advanced Configuration

```python
from soundlab.separation import StemSeparator, SeparationConfig, DemucsModel

# Configure for best quality
config = SeparationConfig(
    model=DemucsModel.HTDEMUCS_FT,  # Fine-tuned model
    segment_length=7.8,              # Larger segments (more memory)
    overlap=0.25,                    # 25% overlap between segments
    int24=True,                      # 24-bit output
    shifts=2,                        # Multiple passes for quality
    device="cuda"                    # Explicit GPU usage
)

# Create separator
separator = StemSeparator(config)

# Separate with progress tracking
def on_progress(progress: float, message: str):
    print(f"{message}: {progress:.1%}")

result = separator.separate(
    audio_path="song.mp3",
    output_dir="output/stems",
    progress_callback=on_progress
)
```

### 6-Stem Separation

```python
from soundlab.separation import StemSeparator, SeparationConfig, DemucsModel

# Use 6-stem model
config = SeparationConfig(model=DemucsModel.HTDEMUCS_6S)
separator = StemSeparator(config)

result = separator.separate("song.mp3", "output/stems")

# Access all 6 stems
print(f"Vocals: {result.stems['vocals']}")
print(f"Drums: {result.stems['drums']}")
print(f"Bass: {result.stems['bass']}")
print(f"Guitar: {result.stems['guitar']}")
print(f"Piano: {result.stems['piano']}")
print(f"Other: {result.stems['other']}")
```

### Memory-Efficient Processing

```python
from soundlab.separation import StemSeparator, SeparationConfig

# Configure for low memory usage
config = SeparationConfig(
    segment_length=3.0,  # Smaller segments
    overlap=0.1,         # Less overlap
    int24=False          # 16-bit output
)

separator = StemSeparator(config)

# Process long audio file without running out of memory
result = separator.separate("long_podcast.mp3", "output/")
```

## Performance Tips

- **GPU Usage**: Use `device="cuda"` for 10-20x speedup on compatible GPUs
- **Quality vs Speed**:
  - Fast: `htdemucs`, `segment_length=10.0`, `shifts=1`
  - Best: `htdemucs_ft`, `segment_length=7.8`, `shifts=2`
- **Memory Management**: Reduce `segment_length` if running out of memory
- **Batch Processing**: Reuse the same `StemSeparator` instance for multiple files

## Typical Processing Times

On NVIDIA T4 GPU (3-minute song):

| Model | Time |
|-------|------|
| htdemucs | ~2 min |
| htdemucs_ft | ~3 min |
| htdemucs_6s | ~4 min |

On CPU (Intel i7):

| Model | Time |
|-------|------|
| htdemucs | ~20 min |
| htdemucs_ft | ~30 min |
| htdemucs_6s | ~40 min |

## API Reference

::: soundlab.separation
    options:
      show_source: true
      members: true
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: true
      show_bases: true
      show_inheritance_diagram: false
      group_by_category: true
      members_order: source
