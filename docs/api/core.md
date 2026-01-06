# Core

Core data models, configuration, and exceptions that form the foundation of SoundLab.

## Overview

The `soundlab.core` module provides the fundamental building blocks for the entire library:

- **Data Models**: Core representations of audio data and metadata
- **Configuration**: Global and module-specific configuration options
- **Exceptions**: Comprehensive error hierarchy for robust error handling
- **Type Definitions**: Common type aliases and enums used throughout the library

## Key Components

### AudioSegment
The primary data structure for representing audio data in memory. Provides methods for manipulation, resampling, and format conversion.

### AudioMetadata
Structured metadata information extracted from audio files, including sample rate, channels, duration, and format details.

### SoundLabConfig
Global configuration options that affect the behavior of all SoundLab operations, including device selection, logging, and resource management.

### Exception Hierarchy
- `SoundLabError` - Base exception for all SoundLab errors
- `AudioProcessingError` - Errors during audio processing
- `ModelLoadError` - Errors loading ML models
- `ValidationError` - Configuration validation errors
- `FileFormatError` - Unsupported file format errors
- `DeviceError` - GPU/device-related errors

## Usage Examples

### Working with AudioSegment

```python
from soundlab.core import AudioSegment
from soundlab.io import load_audio

# Load audio
audio = load_audio("song.mp3")

# Access properties
print(f"Sample rate: {audio.sample_rate} Hz")
print(f"Channels: {audio.channels}")
print(f"Duration: {audio.duration_seconds:.2f}s")
print(f"Shape: {audio.samples.shape}")

# Resample
audio_44k = audio.resample(44100)

# Convert to mono
audio_mono = audio.to_mono()
```

### Global Configuration

```python
from soundlab.core import SoundLabConfig, get_config

# Get current configuration
config = get_config()
print(f"Device: {config.device}")
print(f"Cache dir: {config.cache_dir}")

# Update configuration
config.device = "cpu"  # Force CPU mode
config.log_level = "DEBUG"  # Enable debug logging
```

### Error Handling

```python
from soundlab import SoundLabError
from soundlab.core import AudioProcessingError, ModelLoadError

try:
    # Some operation that might fail
    result = process_audio("input.wav")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
    # Try with a different model or download it
except AudioProcessingError as e:
    print(f"Processing error: {e}")
    # Handle processing failure
except SoundLabError as e:
    print(f"General SoundLab error: {e}")
    # Handle any other SoundLab error
```

## API Reference

::: soundlab.core
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
