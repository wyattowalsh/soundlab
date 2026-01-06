# Transcription

Audio-to-MIDI transcription using Spotify's Basic Pitch.

## Overview

The `soundlab.transcription` module provides polyphonic audio-to-MIDI transcription capabilities using Spotify's Basic Pitch model. Convert audio recordings into MIDI files with accurate note detection, timing, and pitch information.

## Features

- **Polyphonic Transcription**: Detect multiple simultaneous notes
- **High Accuracy**: Spotify's state-of-the-art Basic Pitch model
- **Piano Roll Visualization**: Generate visual representations of transcriptions
- **Customizable Thresholds**: Fine-tune onset and frame detection sensitivity
- **Note Information**: Extract detailed note events with timing and pitch
- **GPU Acceleration**: Automatic GPU utilization for faster processing

## Key Components

### MIDITranscriber
The main interface for audio-to-MIDI transcription. Handles model loading, audio processing, and MIDI generation.

### TranscriptionConfig
Configuration options including:
- Onset threshold (note start detection sensitivity)
- Frame threshold (note sustain detection)
- Minimum note length
- Device selection

### TranscriptionResult
Results containing:
- Path to generated MIDI file
- Note events with timing and pitch
- Pitch range information
- Piano roll visualization methods

## Usage Examples

### Basic Transcription

```python
from soundlab.transcription import MIDITranscriber

# Create transcriber with defaults
transcriber = MIDITranscriber()

# Transcribe audio to MIDI
result = transcriber.transcribe(
    audio_path="melody.wav",
    output_path="output/melody.mid"
)

# Access results
print(f"MIDI file: {result.midi_path}")
print(f"Notes detected: {len(result.notes)}")
print(f"Pitch range: {result.pitch_range[0]}-{result.pitch_range[1]} Hz")
print(f"Duration: {result.duration_seconds:.2f}s")
```

### Advanced Configuration

```python
from soundlab.transcription import MIDITranscriber, TranscriptionConfig

# Configure for sensitive detection
config = TranscriptionConfig(
    onset_threshold=0.3,      # Lower = more sensitive to note starts
    frame_threshold=0.2,      # Lower = more sustain detection
    minimum_note_length=0.03, # Shorter minimum note duration
    device="cuda"             # Use GPU
)

transcriber = MIDITranscriber(config)

# Transcribe with progress tracking
def on_progress(progress: float, message: str):
    print(f"{message}: {progress:.1%}")

result = transcriber.transcribe(
    audio_path="complex_melody.wav",
    output_path="output/melody.mid",
    progress_callback=on_progress
)
```

### Piano Roll Visualization

```python
from soundlab.transcription import MIDITranscriber

transcriber = MIDITranscriber()
result = transcriber.transcribe("melody.wav", "output/melody.mid")

# Generate piano roll image
result.save_piano_roll(
    output_path="output/piano_roll.png",
    figsize=(12, 6),
    dpi=150
)

# Get note information
for note in result.notes[:5]:  # First 5 notes
    print(f"Time: {note.start_time:.2f}s")
    print(f"Duration: {note.duration:.2f}s")
    print(f"Pitch: {note.pitch_hz:.1f} Hz ({note.note_name})")
    print(f"Velocity: {note.velocity}")
    print("---")
```

### Batch Transcription

```python
from soundlab.transcription import MIDITranscriber
from pathlib import Path

transcriber = MIDITranscriber()

# Transcribe multiple files
audio_files = Path("input/").glob("*.wav")

for audio_file in audio_files:
    output_path = f"output/{audio_file.stem}.mid"
    result = transcriber.transcribe(str(audio_file), output_path)
    print(f"Transcribed {audio_file.name}: {len(result.notes)} notes")
```

### Fine-Tuning Thresholds

```python
from soundlab.transcription import MIDITranscriber, TranscriptionConfig

# For vocals or sparse melodies (fewer notes, high confidence)
vocal_config = TranscriptionConfig(
    onset_threshold=0.6,      # Higher threshold
    frame_threshold=0.4,
    minimum_note_length=0.1   # Longer minimum duration
)

# For dense polyphonic music (more notes, lower confidence)
polyphonic_config = TranscriptionConfig(
    onset_threshold=0.3,      # Lower threshold
    frame_threshold=0.2,
    minimum_note_length=0.03  # Shorter minimum duration
)

# Use appropriate config
vocal_transcriber = MIDITranscriber(vocal_config)
poly_transcriber = MIDITranscriber(polyphonic_config)
```

### Working with Results

```python
from soundlab.transcription import MIDITranscriber

transcriber = MIDITranscriber()
result = transcriber.transcribe("piano.wav", "output/piano.mid")

# Analyze transcription
print(f"Total notes: {len(result.notes)}")
print(f"Unique pitches: {len(set(note.pitch_midi for note in result.notes))}")

# Filter notes by pitch range
high_notes = [n for n in result.notes if n.pitch_midi > 72]  # Above C5
low_notes = [n for n in result.notes if n.pitch_midi < 60]   # Below C4

print(f"High notes: {len(high_notes)}")
print(f"Low notes: {len(low_notes)}")

# Get note density (notes per second)
note_density = len(result.notes) / result.duration_seconds
print(f"Note density: {note_density:.1f} notes/second")
```

## Configuration Guidelines

### Onset Threshold (0.1 - 0.9)
- **Low (0.1-0.3)**: Detects more note starts, may include false positives
- **Medium (0.4-0.6)**: Balanced detection (recommended for most cases)
- **High (0.7-0.9)**: Only confident note starts, may miss quiet notes

### Frame Threshold (0.1 - 0.5)
- **Low (0.1-0.2)**: Longer note sustains, may include noise
- **Medium (0.3-0.4)**: Balanced sustain detection
- **High (0.5+)**: Shorter notes, cleaner MIDI

### Minimum Note Length (0.03 - 0.2 seconds)
- **Short (0.03-0.05s)**: Captures fast passages, may include artifacts
- **Medium (0.058-0.1s)**: Balanced (recommended)
- **Long (0.15-0.2s)**: Filters out very short notes

## Performance Tips

- **GPU Usage**: Use `device="cuda"` for 5-10x speedup
- **Audio Quality**: Higher quality input audio yields better transcriptions
- **Monophonic Sources**: Works best on single-instrument recordings
- **Pitch Range**: Optimized for piano range (27.5 Hz to 4186 Hz)

## Typical Processing Times

On NVIDIA T4 GPU:
- 30-second audio: ~5 seconds
- 3-minute song: ~30 seconds

On CPU (Intel i7):
- 30-second audio: ~30 seconds
- 3-minute song: ~3 minutes

## API Reference

::: soundlab.transcription
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
