# I/O

Audio and MIDI file input/output utilities.

## Overview

The `soundlab.io` module provides robust utilities for reading and writing audio and MIDI files. It handles multiple file formats, metadata extraction, and format conversion with consistent error handling.

## Features

- **Multiple Audio Formats**: WAV, MP3, FLAC, OGG, AIFF, M4A
- **MIDI Support**: Read and write MIDI files
- **Metadata Extraction**: Get detailed file information
- **Format Conversion**: Automatic format detection and conversion
- **Type-Safe**: Returns strongly-typed AudioSegment and MIDI objects
- **Error Handling**: Comprehensive validation and error messages

## Supported Formats

### Audio Formats
- **WAV**: Uncompressed audio (highest quality)
- **FLAC**: Lossless compression
- **MP3**: Lossy compression (widespread compatibility)
- **OGG/Vorbis**: Open-source lossy compression
- **AIFF**: Apple's uncompressed format
- **M4A/AAC**: Modern lossy compression

### MIDI
- Standard MIDI files (.mid, .midi)
- Multi-track support
- Tempo and time signature information

## Key Components

### load_audio()
Load audio files into memory as AudioSegment objects.

### save_audio()
Save AudioSegment objects to disk in various formats.

### load_midi()
Load MIDI files into structured MIDI objects.

### save_midi()
Save MIDI objects to disk.

### get_audio_metadata()
Extract metadata from audio files without loading full audio.

## Usage Examples

### Loading Audio

```python
from soundlab.io import load_audio

# Load audio file
audio = load_audio("song.mp3")

# Access properties
print(f"Sample rate: {audio.sample_rate} Hz")
print(f"Channels: {audio.channels}")
print(f"Duration: {audio.duration_seconds:.2f}s")
print(f"Shape: {audio.samples.shape}")
print(f"Format: {audio.format}")

# Audio data as numpy array
samples = audio.samples  # Shape: (channels, samples)
```

### Loading with Options

```python
from soundlab.io import load_audio

# Load at specific sample rate
audio = load_audio("song.mp3", sample_rate=44100)

# Load as mono
audio = load_audio("song.mp3", mono=True)

# Load specific duration
audio = load_audio(
    "song.mp3",
    offset=10.0,    # Start at 10 seconds
    duration=30.0   # Load 30 seconds
)

# Combine options
audio = load_audio(
    "song.mp3",
    sample_rate=48000,
    mono=True,
    offset=0,
    duration=60.0
)
```

### Saving Audio

```python
from soundlab.io import load_audio, save_audio

# Load audio
audio = load_audio("input.wav")

# Save in different formats
save_audio("output.mp3", audio.samples, audio.sample_rate)
save_audio("output.flac", audio.samples, audio.sample_rate)
save_audio("output.ogg", audio.samples, audio.sample_rate)
save_audio("output.m4a", audio.samples, audio.sample_rate)

# Save with quality settings
save_audio(
    "output.mp3",
    audio.samples,
    audio.sample_rate,
    bitrate="320k"  # High-quality MP3
)

save_audio(
    "output.ogg",
    audio.samples,
    audio.sample_rate,
    quality=10  # OGG quality (0-10)
)
```

### Format Conversion

```python
from soundlab.io import load_audio, save_audio

def convert_audio(input_path, output_path, **kwargs):
    """Convert audio between formats."""
    audio = load_audio(input_path)
    save_audio(output_path, audio.samples, audio.sample_rate, **kwargs)
    print(f"Converted {input_path} → {output_path}")

# Convert MP3 to WAV
convert_audio("song.mp3", "song.wav")

# Convert WAV to high-quality MP3
convert_audio("song.wav", "song.mp3", bitrate="320k")

# Convert to FLAC with resampling
audio = load_audio("song.mp3", sample_rate=48000)
save_audio("song.flac", audio.samples, audio.sample_rate)
```

### Metadata Extraction

```python
from soundlab.io import get_audio_metadata

# Get metadata without loading audio
metadata = get_audio_metadata("song.mp3")

print(f"Duration: {metadata.duration_seconds:.2f}s")
print(f"Sample rate: {metadata.sample_rate} Hz")
print(f"Channels: {metadata.channels}")
print(f"Bit depth: {metadata.bit_depth}")
print(f"Bitrate: {metadata.bitrate_kbps} kbps")
print(f"Format: {metadata.format}")
print(f"File size: {metadata.file_size_mb:.2f} MB")

# ID3 tags (if available)
if metadata.title:
    print(f"Title: {metadata.title}")
if metadata.artist:
    print(f"Artist: {metadata.artist}")
if metadata.album:
    print(f"Album: {metadata.album}")
```

### Loading MIDI

```python
from soundlab.io import load_midi

# Load MIDI file
midi = load_midi("melody.mid")

# Access MIDI information
print(f"Ticks per beat: {midi.ticks_per_beat}")
print(f"Number of tracks: {len(midi.tracks)}")
print(f"Duration: {midi.duration_seconds:.2f}s")

# Iterate through tracks
for i, track in enumerate(midi.tracks):
    print(f"\nTrack {i}: {track.name}")
    print(f"  Instrument: {track.instrument}")
    print(f"  Notes: {len(track.notes)}")

# Access notes
for note in midi.tracks[0].notes[:5]:  # First 5 notes
    print(f"Note: {note.pitch} at {note.start_time:.2f}s")
    print(f"  Duration: {note.duration:.2f}s")
    print(f"  Velocity: {note.velocity}")
```

### Saving MIDI

```python
from soundlab.io import load_midi, save_midi

# Load and modify MIDI
midi = load_midi("original.mid")

# Modify tempo
midi.tempo_bpm = 140

# Save modified MIDI
save_midi("modified.mid", midi)
print(f"Saved modified MIDI with tempo: {midi.tempo_bpm} BPM")
```

### Batch Loading

```python
from soundlab.io import load_audio
from pathlib import Path

# Load multiple files
audio_dir = Path("audio_files/")
audio_files = list(audio_dir.glob("*.wav"))

for audio_file in audio_files:
    audio = load_audio(str(audio_file))
    print(f"{audio_file.name}:")
    print(f"  Duration: {audio.duration_seconds:.1f}s")
    print(f"  Sample rate: {audio.sample_rate} Hz")
    print(f"  Channels: {audio.channels}")
```

### Batch Conversion

```python
from soundlab.io import load_audio, save_audio
from pathlib import Path

def batch_convert(input_dir, output_dir, output_format, **kwargs):
    """Convert all audio files in directory to specified format."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Find all audio files
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
        audio_files.extend(input_path.glob(f'*{ext}'))

    # Convert each file
    for audio_file in audio_files:
        output_file = output_path / f"{audio_file.stem}.{output_format}"

        audio = load_audio(str(audio_file))
        save_audio(str(output_file), audio.samples, audio.sample_rate, **kwargs)

        print(f"Converted: {audio_file.name} → {output_file.name}")

# Convert all files to 320k MP3
batch_convert("input/", "output/", "mp3", bitrate="320k")

# Convert all to FLAC
batch_convert("input/", "output_flac/", "flac")
```

### Working with Stems

```python
from soundlab.io import load_audio, save_audio
import numpy as np

# Load multiple stems
vocals = load_audio("vocals.wav")
drums = load_audio("drums.wav")
bass = load_audio("bass.wav")
other = load_audio("other.wav")

# Ensure same sample rate
assert vocals.sample_rate == drums.sample_rate == bass.sample_rate == other.sample_rate

# Mix stems
mixed = vocals.samples + drums.samples + bass.samples + other.samples

# Normalize to prevent clipping
max_val = np.abs(mixed).max()
if max_val > 1.0:
    mixed = mixed / max_val

# Save mix
save_audio("full_mix.wav", mixed, vocals.sample_rate)

# Save instrumental (no vocals)
instrumental = drums.samples + bass.samples + other.samples
save_audio("instrumental.wav", instrumental, vocals.sample_rate)
```

### Stream Processing

```python
from soundlab.io import load_audio, save_audio

def process_in_chunks(input_path, output_path, chunk_duration=30.0):
    """Process large audio files in chunks."""
    # Get total duration
    from soundlab.io import get_audio_metadata
    metadata = get_audio_metadata(input_path)
    total_duration = metadata.duration_seconds

    processed_chunks = []

    # Process in chunks
    offset = 0
    while offset < total_duration:
        # Load chunk
        audio = load_audio(
            input_path,
            offset=offset,
            duration=chunk_duration
        )

        # Process chunk (example: normalize)
        max_val = abs(audio.samples).max()
        if max_val > 0:
            normalized = audio.samples / max_val
        else:
            normalized = audio.samples

        processed_chunks.append(normalized)

        offset += chunk_duration
        print(f"Processed {offset:.1f}s / {total_duration:.1f}s")

    # Concatenate chunks
    import numpy as np
    full_audio = np.concatenate(processed_chunks, axis=1)

    # Save
    save_audio(output_path, full_audio, audio.sample_rate)
    print(f"Saved: {output_path}")

# Process large file
process_in_chunks("large_file.wav", "processed.wav")
```

### Export Presets

```python
from soundlab.io import save_audio

def export_for_streaming(samples, sample_rate, output_path):
    """Export optimized for streaming platforms."""
    save_audio(
        output_path,
        samples,
        sample_rate,
        bitrate="256k",  # Good quality/size balance
        format="mp3"
    )

def export_for_archive(samples, sample_rate, output_path):
    """Export lossless for archiving."""
    save_audio(
        output_path,
        samples,
        sample_rate,
        format="flac"
    )

def export_for_podcast(samples, sample_rate, output_path):
    """Export optimized for podcasts."""
    save_audio(
        output_path,
        samples,
        sample_rate,
        bitrate="128k",  # Lower bitrate for speech
        format="mp3"
    )
```

## Format Recommendations

### Lossless (Archiving, Production)
- **WAV**: Universal compatibility, no compression
- **FLAC**: Compression, smaller files, maintains quality

### Lossy (Distribution, Streaming)
- **MP3 320k**: High quality, widespread compatibility
- **MP3 256k**: Good quality, smaller files
- **MP3 128k**: Acceptable for speech/podcasts
- **OGG Vorbis**: Better quality than MP3 at same bitrate
- **M4A/AAC**: Modern format, good quality/size ratio

### Production Standards
- **Mastering**: 24-bit WAV or FLAC at 48kHz or 96kHz
- **Mixing**: 24-bit WAV at 48kHz
- **Streaming**: MP3 320k or AAC 256k at 44.1kHz
- **Podcasts**: MP3 128k or AAC 128k at 44.1kHz

## Error Handling

```python
from soundlab.io import load_audio, save_audio
from soundlab.core import FileFormatError, SoundLabError

try:
    audio = load_audio("song.mp3")
except FileNotFoundError:
    print("File not found")
except FileFormatError as e:
    print(f"Unsupported format: {e}")
except SoundLabError as e:
    print(f"Error loading audio: {e}")

try:
    save_audio("output.wav", samples, sample_rate)
except PermissionError:
    print("Cannot write to file")
except SoundLabError as e:
    print(f"Error saving audio: {e}")
```

## API Reference

::: soundlab.io
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
