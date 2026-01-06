# MIDI Transcription Guide

Learn how to convert audio to MIDI using Spotify's Basic Pitch model for polyphonic pitch detection and note transcription.

## Overview

MIDI transcription (also called audio-to-MIDI conversion) is the process of analyzing audio and converting detected pitches into MIDI notes. SoundLab uses Spotify's [Basic Pitch](https://github.com/spotify/basic-pitch) model, which supports:

- **Polyphonic transcription** - Multiple simultaneous notes
- **Accurate pitch detection** - MIDI pitch and frequency
- **Note timing** - Precise onset and duration
- **Confidence scores** - Reliability metrics

### Use Cases

- **Music notation** - Generate sheet music from recordings
- **Sampling** - Extract melodies for MIDI instruments
- **Analysis** - Study harmonic and melodic content
- **Learning** - Transcribe songs to learn
- **Remixing** - Recreate melodies in new arrangements
- **Game audio** - Extract musical patterns

## Quick Start

Basic transcription:

```python
from soundlab.transcription import MIDITranscriber

# Create transcriber with default settings
transcriber = MIDITranscriber()

# Transcribe audio to MIDI
result = transcriber.transcribe(
    audio_path="melody.wav",
    output_path="output/melody.mid"
)

# View results
print(f"MIDI file: {result.midi_path}")
print(f"Notes detected: {len(result.notes)}")
print(f"Duration: {result.duration:.2f}s")
print(f"Pitch range: {result.pitch_range[0]}-{result.pitch_range[1]}")

# Generate piano roll visualization
result.save_piano_roll("output/piano_roll.png")
```

## Configuration Options

### Detection Thresholds

#### onset_thresh

Controls note onset sensitivity (0.1-0.9):

```python
from soundlab.transcription import TranscriptionConfig, MIDITranscriber

# Conservative (fewer notes, fewer false positives)
config = TranscriptionConfig(onset_threshold=0.7)

# Default (balanced)
config = TranscriptionConfig(onset_threshold=0.5)

# Lenient (more notes, more false positives)
config = TranscriptionConfig(onset_threshold=0.3)

transcriber = MIDITranscriber(config)
```

**Guidelines**:
- **0.7-0.9**: Clean, solo instruments with clear attacks
- **0.5**: Default, works for most material
- **0.3-0.4**: Soft/legato playing, background melodies
- **0.1-0.2**: Experimental, captures very faint notes

!!! tip "Finding the Right Threshold"
    Start with 0.5 and adjust:

    - Too many extra notes? Increase onset_threshold
    - Missing notes? Decrease onset_threshold

#### frame_thresh

Frame-level activation threshold (0.1-0.9):

```python
# Conservative (shorter notes)
config = TranscriptionConfig(
    onset_threshold=0.5,
    frame_threshold=0.5
)

# Default (balanced)
config = TranscriptionConfig(
    onset_threshold=0.5,
    frame_threshold=0.3
)

# Lenient (longer notes)
config = TranscriptionConfig(
    onset_threshold=0.5,
    frame_threshold=0.1
)
```

**Guidelines**:
- Higher values: Shorter notes, cleaner transcription
- Lower values: Longer notes, may include decay/sustain
- Usually set lower than onset_threshold

### Note Parameters

#### minimum_note_length

Minimum note duration in milliseconds (10-200):

```python
# Short notes (fast passages, percussion)
config = TranscriptionConfig(minimum_note_length=30.0)

# Default (general use)
config = TranscriptionConfig(minimum_note_length=58.0)

# Long notes (sustained melodies)
config = TranscriptionConfig(minimum_note_length=100.0)
```

**Guidelines**:
- **30-50ms**: Fast piano runs, guitar solos
- **58ms**: Default, good for most music
- **100-200ms**: Slow melodies, vocals

### Frequency Range

Limit detection to specific frequency ranges:

```python
# Default (C1 to C7 - covers most instruments)
config = TranscriptionConfig(
    minimum_frequency=32.7,   # C1
    maximum_frequency=2093.0   # C7
)

# Piano range only (A0 to C8)
config = TranscriptionConfig(
    minimum_frequency=27.5,    # A0
    maximum_frequency=4186.0   # C8
)

# Vocals only (E2 to E6)
config = TranscriptionConfig(
    minimum_frequency=82.4,    # E2
    maximum_frequency=1318.5   # E6
)

# Bass only (E1 to E4)
config = TranscriptionConfig(
    minimum_frequency=41.2,    # E1
    maximum_frequency=329.6    # E4
)
```

**Common ranges**:

| Instrument | Min Hz | Max Hz | Min Note | Max Note |
|------------|--------|--------|----------|----------|
| Bass | 41.2 | 329.6 | E1 | E4 |
| Guitar | 82.4 | 1318.5 | E2 | E6 |
| Vocals | 82.4 | 1318.5 | E2 | E6 |
| Piano | 27.5 | 4186.0 | A0 | C8 |
| Violin | 196.0 | 3520.0 | G3 | A7 |

### Advanced Options

#### melodia_trick

Enable post-processing for monophonic melodies:

```python
# Enable for solo melodies (default)
config = TranscriptionConfig(melodia_trick=True)

# Disable for polyphonic music
config = TranscriptionConfig(melodia_trick=False)
```

#### include_pitch_bends

Include pitch bend information in MIDI:

```python
# Enable pitch bends (for vibrato, slides)
config = TranscriptionConfig(include_pitch_bends=True)

# Disable (default, simpler MIDI)
config = TranscriptionConfig(include_pitch_bends=False)
```

#### device

Control GPU usage:

```python
# Automatic GPU detection (default)
config = TranscriptionConfig(device="auto")

# Force CPU
config = TranscriptionConfig(device="cpu")

# Specific GPU
config = TranscriptionConfig(device="cuda:0")
```

## Usage Examples

### Example 1: Piano Transcription

Transcribe piano with high accuracy:

```python
from soundlab.transcription import TranscriptionConfig, MIDITranscriber

config = TranscriptionConfig(
    onset_threshold=0.5,
    frame_threshold=0.3,
    minimum_note_length=58.0,
    minimum_frequency=27.5,    # A0
    maximum_frequency=4186.0,  # C8
    melodia_trick=False         # Polyphonic
)

transcriber = MIDITranscriber(config)
result = transcriber.transcribe(
    audio_path="piano.wav",
    output_path="output/piano.mid"
)

print(f"Transcribed {len(result.notes)} notes")
print(f"Pitch range: {result.pitch_range}")
```

### Example 2: Vocal Melody

Extract melody from vocals:

```python
config = TranscriptionConfig(
    onset_threshold=0.6,
    frame_threshold=0.4,
    minimum_note_length=100.0,  # Longer notes for vocals
    minimum_frequency=82.4,     # E2
    maximum_frequency=1318.5,   # E6
    melodia_trick=True,         # Monophonic melody
    include_pitch_bends=True    # Capture vibrato
)

transcriber = MIDITranscriber(config)
result = transcriber.transcribe(
    audio_path="vocals.wav",
    output_path="output/vocals.mid"
)

# Generate piano roll
result.save_piano_roll(
    output_path="output/vocals_piano_roll.png",
    figsize=(12, 6)
)
```

### Example 3: Fast Guitar Solo

Capture fast notes with precision:

```python
config = TranscriptionConfig(
    onset_threshold=0.7,        # Clear attacks
    frame_threshold=0.5,
    minimum_note_length=30.0,   # Short notes
    minimum_frequency=82.4,     # E2
    maximum_frequency=1318.5,   # E6
    melodia_trick=False
)

transcriber = MIDITranscriber(config)
result = transcriber.transcribe(
    audio_path="guitar_solo.wav",
    output_path="output/guitar_solo.mid"
)
```

### Example 4: Bass Line Extraction

Focus on low frequencies:

```python
config = TranscriptionConfig(
    onset_threshold=0.6,
    frame_threshold=0.3,
    minimum_note_length=58.0,
    minimum_frequency=41.2,     # E1
    maximum_frequency=329.6,    # E4 - bass range only
    melodia_trick=False
)

transcriber = MIDITranscriber(config)
result = transcriber.transcribe(
    audio_path="bass.wav",
    output_path="output/bass.mid"
)
```

### Example 5: Transcribe from Separated Stems

Combine with stem separation for best results:

```python
from soundlab.separation import StemSeparator
from soundlab.transcription import TranscriptionConfig, MIDITranscriber

# Step 1: Separate stems
separator = StemSeparator()
stems = separator.separate("song.mp3", "output/stems/")

# Step 2: Transcribe vocals
vocal_config = TranscriptionConfig(
    onset_threshold=0.6,
    minimum_frequency=82.4,
    maximum_frequency=1318.5,
    melodia_trick=True
)
vocal_transcriber = MIDITranscriber(vocal_config)
vocal_midi = vocal_transcriber.transcribe(
    audio_path=stems.vocals,
    output_path="output/vocals.mid"
)

# Step 3: Transcribe bass
bass_config = TranscriptionConfig(
    onset_threshold=0.5,
    minimum_frequency=41.2,
    maximum_frequency=329.6
)
bass_transcriber = MIDITranscriber(bass_config)
bass_midi = bass_transcriber.transcribe(
    audio_path=stems.bass,
    output_path="output/bass.mid"
)

print(f"Vocal notes: {len(vocal_midi.notes)}")
print(f"Bass notes: {len(bass_midi.notes)}")
```

### Example 6: Batch Transcription

Transcribe multiple files:

```python
from pathlib import Path

transcriber = MIDITranscriber()

input_dir = Path("input/stems/")
output_dir = Path("output/midi/")
output_dir.mkdir(parents=True, exist_ok=True)

for audio_file in input_dir.glob("*.wav"):
    print(f"Transcribing: {audio_file.name}")

    result = transcriber.transcribe(
        audio_path=audio_file,
        output_path=output_dir / f"{audio_file.stem}.mid"
    )

    print(f"  Notes: {len(result.notes)}")
    print(f"  Duration: {result.duration:.2f}s")
    print(f"  Time: {result.processing_time_seconds:.2f}s")
```

### Example 7: Note Analysis

Analyze transcribed notes:

```python
result = transcriber.transcribe("melody.wav", "output/melody.mid")

# Get note statistics
pitches = [note.pitch for note in result.notes]
print(f"Unique pitches: {len(set(pitches))}")
print(f"Most common pitch: {max(set(pitches), key=pitches.count)}")

# Analyze note durations
durations_ms = [note.duration_ms for note in result.notes]
avg_duration = sum(durations_ms) / len(durations_ms)
print(f"Average note duration: {avg_duration:.1f} ms")

# Find longest note
longest_note = max(result.notes, key=lambda n: n.duration)
print(f"Longest note: {longest_note.pitch_name} at {longest_note.start_time:.2f}s")
print(f"  Duration: {longest_note.duration:.2f}s")

# Get notes in specific time range
chorus_notes = result.get_notes_in_range(30.0, 60.0)
print(f"Notes in chorus section: {len(chorus_notes)}")

# Analyze pitch range
note_names = [note.pitch_name for note in result.notes]
print(f"Note range: {result.notes[0].pitch_name} to {result.notes[-1].pitch_name}")
```

## Working with Results

### Note Objects

Each detected note is a `NoteEvent` with:

```python
result = transcriber.transcribe("melody.wav", "output/melody.mid")

for note in result.notes[:5]:
    print(f"Pitch: {note.pitch} ({note.pitch_name})")
    print(f"Frequency: {note.frequency:.2f} Hz")
    print(f"Start: {note.start_time:.3f}s")
    print(f"End: {note.end_time:.3f}s")
    print(f"Duration: {note.duration:.3f}s ({note.duration_ms:.1f}ms)")
    print(f"Velocity: {note.velocity}")
    print(f"Confidence: {note.confidence:.2f}")
    print()
```

### Piano Roll Visualization

Generate visual representation:

```python
result = transcriber.transcribe("melody.wav", "output/melody.mid")

# Basic piano roll
result.save_piano_roll("output/piano_roll.png")

# Customized piano roll
result.save_piano_roll(
    output_path="output/custom_roll.png",
    figsize=(16, 8),
    dpi=150
)
```

### MIDI Export

Work with the generated MIDI file:

```python
from soundlab.io import load_midi, save_midi

# Load MIDI
midi = load_midi("output/melody.mid")

# Modify and save
# (use python-midi or mido library for modifications)
save_midi("output/modified.mid", midi)
```

## Best Practices

### Input Preparation

For best transcription quality:

#### 1. Use Clean Audio

```python
from soundlab.separation import StemSeparator

# Separate instrument first
separator = StemSeparator()
stems = separator.separate("song.mp3", "output/stems/")

# Transcribe isolated instrument
transcriber = MIDITranscriber()
result = transcriber.transcribe(stems.vocals, "output/vocals.mid")
```

#### 2. Apply Preprocessing

```python
from soundlab.effects import EffectsChain
from soundlab.effects.eq import HighPassFilterConfig
from soundlab.effects.dynamics import CompressorConfig

# Clean up audio before transcription
chain = (
    EffectsChain()
    .add(HighPassFilterConfig(cutoff_frequency_hz=80))  # Remove rumble
    .add(CompressorConfig(threshold_db=-20, ratio=3.0)) # Even out dynamics
)

cleaned = chain.process("input.wav", "cleaned.wav")

# Transcribe cleaned audio
result = transcriber.transcribe(cleaned, "output/melody.mid")
```

#### 3. Match Frequency Range

```python
# Analyze audio first
from soundlab.analysis import analyze_audio

analysis = analyze_audio("melody.wav")
if analysis.spectral:
    print(f"Spectral centroid: {analysis.spectral.spectral_centroid:.1f} Hz")

# Adjust frequency range based on content
if analysis.spectral.spectral_centroid < 500:
    # Low-frequency content (bass)
    config = TranscriptionConfig(
        minimum_frequency=40.0,
        maximum_frequency=400.0
    )
elif analysis.spectral.spectral_centroid > 2000:
    # High-frequency content
    config = TranscriptionConfig(
        minimum_frequency=200.0,
        maximum_frequency=4000.0
    )
```

### Threshold Tuning

Iteratively adjust thresholds:

```python
def test_thresholds(audio_path, onset_values, frame_values):
    """Test different threshold combinations."""
    for onset in onset_values:
        for frame in frame_values:
            config = TranscriptionConfig(
                onset_threshold=onset,
                frame_threshold=frame
            )
            transcriber = MIDITranscriber(config)
            result = transcriber.transcribe(
                audio_path=audio_path,
                output_path=f"output/test_o{onset}_f{frame}.mid"
            )
            print(f"Onset: {onset:.1f}, Frame: {frame:.1f} -> {len(result.notes)} notes")

# Test range
test_thresholds(
    "melody.wav",
    onset_values=[0.3, 0.5, 0.7],
    frame_values=[0.1, 0.3, 0.5]
)
```

### Quality Validation

Verify transcription quality:

```python
result = transcriber.transcribe("melody.wav", "output/melody.mid")

# Check note count
if len(result.notes) == 0:
    print("Warning: No notes detected!")
    print("  Try lowering onset_threshold")
elif len(result.notes) > 1000:
    print("Warning: Too many notes detected!")
    print("  Try raising onset_threshold")

# Check pitch range
min_pitch, max_pitch = result.pitch_range
if max_pitch - min_pitch < 12:  # Less than one octave
    print("Warning: Very narrow pitch range")
    print("  Check minimum_frequency and maximum_frequency settings")

# Check note density
notes_per_second = len(result.notes) / result.duration
if notes_per_second > 20:
    print("Warning: Very dense transcription")
    print("  May have false positives, try raising thresholds")

# Check average confidence
if result.notes:
    avg_confidence = sum(n.confidence for n in result.notes) / len(result.notes)
    print(f"Average confidence: {avg_confidence:.2%}")
    if avg_confidence < 0.5:
        print("  Low confidence, results may be unreliable")
```

## Common Issues and Solutions

### No Notes Detected

**Problem**: Transcription produces zero notes.

**Solutions**:

```python
# 1. Lower onset threshold
config = TranscriptionConfig(onset_threshold=0.3)

# 2. Lower frame threshold
config = TranscriptionConfig(
    onset_threshold=0.5,
    frame_threshold=0.1
)

# 3. Expand frequency range
config = TranscriptionConfig(
    minimum_frequency=20.0,
    maximum_frequency=5000.0
)

# 4. Reduce minimum note length
config = TranscriptionConfig(minimum_note_length=30.0)

# 5. Check input audio
from soundlab.io import load_audio
import numpy as np

audio, sr = load_audio("melody.wav")
rms = np.sqrt(np.mean(audio ** 2))
print(f"Audio RMS: {20 * np.log10(rms):.1f} dB")
if rms < 0.001:
    print("Input audio is too quiet or silent!")
```

### Too Many False Positives

**Problem**: Transcription includes noise and artifacts.

**Solutions**:

```python
# 1. Raise onset threshold
config = TranscriptionConfig(onset_threshold=0.7)

# 2. Raise frame threshold
config = TranscriptionConfig(
    onset_threshold=0.7,
    frame_threshold=0.5
)

# 3. Increase minimum note length
config = TranscriptionConfig(minimum_note_length=100.0)

# 4. Narrow frequency range
config = TranscriptionConfig(
    minimum_frequency=100.0,
    maximum_frequency=2000.0
)

# 5. Use stem separation first
from soundlab.separation import StemSeparator

separator = StemSeparator()
stems = separator.separate("song.mp3", "output/stems/")
result = transcriber.transcribe(stems.vocals, "output/vocals.mid")
```

### Inaccurate Pitch Detection

**Problem**: Detected pitches don't match actual audio.

**Solutions**:

```python
# 1. Use separated/isolated audio
# See stem separation examples above

# 2. Verify input audio quality
from soundlab.io import get_audio_metadata

metadata = get_audio_metadata("melody.wav")
print(f"Sample rate: {metadata.sample_rate} Hz")
if metadata.sample_rate < 44100:
    print("Warning: Low sample rate may affect accuracy")

# 3. Check for pitch shifting in original audio
# Compare transcribed pitches with expected pitches

# 4. Adjust frequency range to expected content
config = TranscriptionConfig(
    minimum_frequency=200.0,  # Adjust to expected range
    maximum_frequency=800.0
)
```

### Notes Too Short/Long

**Problem**: Note durations don't match audio.

**Solutions**:

```python
# For too-short notes:
config = TranscriptionConfig(
    frame_threshold=0.1,        # Lower = longer notes
    minimum_note_length=80.0    # Higher = filter short notes
)

# For too-long notes:
config = TranscriptionConfig(
    frame_threshold=0.5,        # Higher = shorter notes
    minimum_note_length=30.0    # Lower = keep short notes
)
```

### Missing Notes

**Problem**: Known notes are not detected.

**Solutions**:

```python
# 1. Lower thresholds
config = TranscriptionConfig(
    onset_threshold=0.3,
    frame_threshold=0.1
)

# 2. Check frequency range
# Make sure note frequencies are within range
# E.g., for bass (41-330 Hz)

# 3. Preprocess audio
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig

chain = EffectsChain().add(
    CompressorConfig(threshold_db=-25, ratio=4.0)
)
compressed = chain.process("input.wav", "compressed.wav")
result = transcriber.transcribe(compressed, "output.mid")

# 4. Check input audio level
# Normalize if too quiet
from soundlab.io import load_audio, save_audio
import numpy as np

audio, sr = load_audio("input.wav")
audio_normalized = audio / np.max(np.abs(audio)) * 0.9
save_audio("normalized.wav", audio_normalized, sr)
```

## Advanced Techniques

### Multi-Pass Transcription

Transcribe different frequency ranges separately:

```python
# Pass 1: High frequencies (melody)
high_config = TranscriptionConfig(
    minimum_frequency=200.0,
    maximum_frequency=2000.0,
    onset_threshold=0.5
)
high_transcriber = MIDITranscriber(high_config)
high_result = high_transcriber.transcribe("audio.wav", "output/high.mid")

# Pass 2: Low frequencies (bass)
low_config = TranscriptionConfig(
    minimum_frequency=40.0,
    maximum_frequency=300.0,
    onset_threshold=0.6
)
low_transcriber = MIDITranscriber(low_config)
low_result = low_transcriber.transcribe("audio.wav", "output/low.mid")

# Combine results
# (use MIDI library to merge files)
```

### Confidence Filtering

Filter notes by confidence score:

```python
result = transcriber.transcribe("melody.wav", "output/melody.mid")

# Filter low-confidence notes
high_confidence_notes = [
    note for note in result.notes
    if note.confidence > 0.7
]

print(f"Original notes: {len(result.notes)}")
print(f"High confidence: {len(high_confidence_notes)}")

# Save filtered notes to new MIDI
# (requires MIDI library to rebuild file)
```

### Quantization

Align notes to musical grid:

```python
def quantize_notes(notes, grid_ms=125.0):
    """Quantize note timings to musical grid (e.g., 16th notes at 120 BPM)."""
    quantized = []
    for note in notes:
        # Quantize start time
        start_quantized = round(note.start_time * 1000 / grid_ms) * grid_ms / 1000

        # Quantize duration
        duration_quantized = round(note.duration * 1000 / grid_ms) * grid_ms / 1000

        # Create quantized note (pseudo-code)
        # In practice, use MIDI library to create new note
        quantized.append({
            'pitch': note.pitch,
            'start': start_quantized,
            'duration': duration_quantized,
            'velocity': note.velocity
        })

    return quantized

result = transcriber.transcribe("melody.wav", "output/melody.mid")
quantized = quantize_notes(result.notes, grid_ms=125.0)  # 16th notes at 120 BPM
```

## Performance Tips

### GPU Acceleration

```python
import torch

# Verify GPU usage
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Ensure GPU is used
config = TranscriptionConfig(device="auto")

# Typical GPU speedup: 5-10x over CPU
```

### Batch Processing Optimization

```python
# Reuse transcriber instance
transcriber = MIDITranscriber()  # Model loaded once

files = ["file1.wav", "file2.wav", "file3.wav"]

for file in files:
    result = transcriber.transcribe(file, f"output/{file.stem}.mid")
    # Model stays in memory between files
```

### Processing Time

Typical times for 3-minute audio:

| Device | Time | Notes |
|--------|------|-------|
| GPU (T4) | ~30 sec | Recommended |
| CPU (i9) | ~3 min | 6x slower |
| CPU (i5) | ~5 min | 10x slower |

## Next Steps

- **[Separation Guide](separation.md)** - Isolate instruments first for better transcription
- **[Analysis Guide](analysis.md)** - Analyze harmonic content
- **[Effects Guide](effects.md)** - Process audio before transcription

---

**Need help?** Check [GitHub Issues](https://github.com/wyattwalsh/soundlab/issues) or [Discussions](https://github.com/wyattwalsh/soundlab/discussions).
