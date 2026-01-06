# Quick Start Guide

Get up and running with SoundLab in minutes. This guide covers the basics of each major feature with practical examples.

## Prerequisites

Ensure you have SoundLab installed:

```bash
pip install soundlab
```

For voice generation features:

```bash
pip install soundlab[voice]
```

See the [Installation Guide](installation.md) for detailed setup instructions.

## Basic Usage

### Your First Separation

Separate a song into individual stems (vocals, drums, bass, other):

```python
from soundlab.separation import StemSeparator

# Create separator with default settings
separator = StemSeparator()

# Separate audio file
result = separator.separate(
    audio_path="song.mp3",
    output_dir="output/stems"
)

# Access individual stems
print(f"Vocals: {result.vocals}")
print(f"Drums: {result.drums}")
print(f"Bass: {result.bass}")
print(f"Other: {result.other}")
print(f"Processing time: {result.processing_time_seconds:.1f}s")
```

**Output structure:**
```
output/stems/
├── vocals.wav
├── drums.wav
├── bass.wav
└── other.wav
```

!!! tip "Model Selection"
    The default model (`htdemucs_ft`) provides the best quality. Use `htdemucs` for faster processing:
    ```python
    from soundlab.separation import SeparationConfig, DemucsModel

    config = SeparationConfig(model=DemucsModel.HTDEMUCS)
    separator = StemSeparator(config)
    ```

### Your First Transcription

Convert audio to MIDI:

```python
from soundlab.transcription import MIDITranscriber

# Create transcriber
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

!!! info "Best Results"
    Transcription works best with:

    - Monophonic or simple polyphonic audio
    - Clear, isolated instruments (use stem separation first!)
    - Minimal background noise

### Your First Analysis

Analyze musical features:

```python
from soundlab.analysis import analyze_audio

# Perform comprehensive analysis
result = analyze_audio("song.mp3")

# View results
print(f"Duration: {result.duration_seconds:.2f}s")
print(f"Sample Rate: {result.sample_rate} Hz")
print(f"Channels: {result.channels}")

# Tempo
if result.tempo:
    print(f"\nTempo: {result.tempo.bpm:.1f} BPM")
    print(f"Confidence: {result.tempo.confidence:.2%}")

# Key detection
if result.key:
    print(f"\nKey: {result.key.name}")
    print(f"Camelot: {result.key.camelot}")
    print(f"Open Key: {result.key.open_key}")

# Loudness
if result.loudness:
    print(f"\nLoudness: {result.loudness.integrated_lufs:.1f} LUFS")
    print(f"True Peak: {result.loudness.true_peak_db:.1f} dB")
    print(f"Broadcast Safe: {result.loudness.is_broadcast_safe}")

# Spectral features
if result.spectral:
    print(f"\nSpectral Centroid: {result.spectral.spectral_centroid:.1f} Hz")
    print(f"Brightness: {result.spectral.brightness}")

# Summary
print("\nSummary:")
for key, value in result.summary.items():
    print(f"  {key}: {value}")
```

### Your First Effects Chain

Apply professional audio effects:

```python
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig
from soundlab.effects.eq import HighPassFilterConfig
from soundlab.effects.time_based import ReverbConfig

# Build effects chain using fluent API
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
        wet_level=0.3,
        dry_level=0.7
    ))
)

# Process audio
output_path = chain.process(
    input_path="vocals_dry.wav",
    output_path="vocals_processed.wav"
)

print(f"Processed with {len(chain.effects)} effects")
print(f"Chain: {chain}")
```

### Your First Voice Generation

Generate speech with voice cloning (requires `soundlab[voice]`):

```python
from soundlab.voice import TextToSpeech, TTSConfig

# Configure with voice cloning
config = TTSConfig(
    language="en",
    speaker_wav="voice_sample.wav",  # 6-30 second reference
)

# Generate speech
tts = TextToSpeech(config)
result = tts.synthesize(
    text="Welcome to SoundLab. This is a demonstration of voice cloning.",
    output_path="output/speech.wav"
)

print(f"Generated: {result.audio_path}")
print(f"Duration: {result.duration_seconds:.1f}s")
print(f"Speaking rate: {result.words_per_minute:.0f} WPM")
```

!!! warning "Voice Cloning Requirements"
    For best results, use reference audio that is:

    - 6-30 seconds long (15s recommended)
    - Single speaker only
    - Clear speech without music
    - Minimal background noise

## Common Workflows

### Workflow 1: Separate and Transcribe

Extract vocals and convert to MIDI:

```python
from soundlab.separation import StemSeparator
from soundlab.transcription import MIDITranscriber

# Step 1: Separate vocals
separator = StemSeparator()
stems = separator.separate("song.mp3", "output/stems/")

# Step 2: Transcribe vocals to MIDI
transcriber = MIDITranscriber()
midi = transcriber.transcribe(
    audio_path=stems.vocals,
    output_path="output/vocals.mid"
)

print(f"Transcribed {len(midi.notes)} notes from vocals")
```

### Workflow 2: Analyze and Process

Analyze audio, then apply targeted effects:

```python
from soundlab.analysis import analyze_audio
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig, LimiterConfig

# Analyze audio
analysis = analyze_audio("vocals.wav")

# Build effects based on analysis
chain = EffectsChain()

# Add compression if too dynamic
if analysis.loudness and analysis.loudness.dynamic_range_db > 15:
    chain.add(CompressorConfig(
        threshold_db=-20,
        ratio=4.0,
        attack_ms=5.0,
        release_ms=100.0
    ))

# Add limiter if peaks are too high
if analysis.loudness and analysis.loudness.true_peak_db > -1.0:
    chain.add(LimiterConfig(threshold_db=-1.0))

# Process
output = chain.process("vocals.wav", "vocals_processed.wav")
print(f"Applied {len(chain.effects)} effects based on analysis")
```

### Workflow 3: Separate, Process, Mix

Professional mixing workflow:

```python
from pathlib import Path
from soundlab.separation import StemSeparator
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig
from soundlab.effects.eq import HighPassFilterConfig, PeakFilterConfig
from soundlab.effects.time_based import ReverbConfig
from soundlab.io import load_audio, save_audio
import numpy as np

# Step 1: Separate stems
separator = StemSeparator()
stems = separator.separate("song.mp3", "output/stems/")

# Step 2: Process each stem with appropriate effects
# Vocals chain
vocals_chain = (
    EffectsChain()
    .add(HighPassFilterConfig(cutoff_frequency_hz=80))
    .add(CompressorConfig(threshold_db=-18, ratio=3.0))
    .add(PeakFilterConfig(cutoff_frequency_hz=3000, gain_db=2.0))
    .add(ReverbConfig(room_size=0.3, wet_level=0.2))
)

# Drums chain
drums_chain = (
    EffectsChain()
    .add(CompressorConfig(threshold_db=-15, ratio=6.0, attack_ms=1.0))
    .add(HighPassFilterConfig(cutoff_frequency_hz=40))
)

# Process stems
vocals_processed = vocals_chain.process(stems.vocals, "output/vocals_fx.wav")
drums_processed = drums_chain.process(stems.drums, "output/drums_fx.wav")

print("Processing complete!")
print(f"Vocals: {vocals_processed}")
print(f"Drums: {drums_processed}")
```

### Workflow 4: Batch Processing

Process multiple files efficiently:

```python
from pathlib import Path
from soundlab.separation import StemSeparator

# Setup
input_dir = Path("input/")
output_dir = Path("output/")
separator = StemSeparator()

# Process all MP3 files
for audio_file in input_dir.glob("*.mp3"):
    print(f"\nProcessing: {audio_file.name}")

    # Separate stems
    result = separator.separate(
        audio_path=audio_file,
        output_dir=output_dir / audio_file.stem
    )

    print(f"  ✓ Vocals: {result.vocals.name}")
    print(f"  ✓ Drums: {result.drums.name}")
    print(f"  ✓ Time: {result.processing_time_seconds:.1f}s")

print("\nBatch processing complete!")
```

## Configuration Examples

### High-Quality Separation

For best quality (slower):

```python
from soundlab.separation import SeparationConfig, DemucsModel, StemSeparator

config = SeparationConfig(
    model=DemucsModel.HTDEMUCS_FT,  # Best quality model
    segment_length=10.0,             # Larger segments
    overlap=0.25,
    shifts=2,                        # Multiple shifts for quality
    int24=True                       # 24-bit output
)

separator = StemSeparator(config)
result = separator.separate("song.mp3", "output/")
```

### Fast Separation

For speed (good quality):

```python
config = SeparationConfig(
    model=DemucsModel.HTDEMUCS,  # Faster model
    segment_length=7.8,
    overlap=0.1,                  # Less overlap
    shifts=0,                     # No shifts
    float32=True                  # Faster processing
)

separator = StemSeparator(config)
result = separator.separate("song.mp3", "output/")
```

### Precise Transcription

For accurate MIDI:

```python
from soundlab.transcription import TranscriptionConfig, MIDITranscriber

config = TranscriptionConfig(
    onset_threshold=0.7,          # Higher = fewer false positives
    frame_threshold=0.5,          # Higher = more conservative
    minimum_note_length=100.0,    # Milliseconds
    minimum_frequency=32.7,       # C1
    maximum_frequency=2093.0,     # C7
)

transcriber = MIDITranscriber(config)
result = transcriber.transcribe("piano.wav", "output/piano.mid")
```

### Lenient Transcription

For capturing more notes:

```python
config = TranscriptionConfig(
    onset_threshold=0.3,          # Lower = more notes
    frame_threshold=0.1,          # Lower = more notes
    minimum_note_length=50.0,     # Shorter notes
)

transcriber = MIDITranscriber(config)
result = transcriber.transcribe("melody.wav", "output/melody.mid")
```

## Progress Tracking

Monitor long-running operations:

```python
from soundlab.separation import StemSeparator

def progress_callback(step: str, percent: float):
    """Display progress updates."""
    bar_length = 40
    filled = int(bar_length * percent / 100)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"\r{step}: [{bar}] {percent:.1f}%", end="", flush=True)

separator = StemSeparator()
result = separator.separate(
    audio_path="song.mp3",
    output_dir="output/",
    progress_callback=progress_callback
)
print("\nComplete!")
```

## Error Handling

Handle errors gracefully:

```python
from soundlab.core import SoundLabError
from soundlab.separation import StemSeparator

try:
    separator = StemSeparator()
    result = separator.separate("song.mp3", "output/")
    print(f"Success! Processed in {result.processing_time_seconds:.1f}s")

except FileNotFoundError as e:
    print(f"File not found: {e}")

except SoundLabError as e:
    print(f"Processing failed: {e}")
    # Could retry with different settings

except Exception as e:
    print(f"Unexpected error: {e}")
    raise
```

## Audio I/O

Work with audio data directly:

```python
from soundlab.io import load_audio, save_audio, get_audio_metadata
import numpy as np

# Load audio
audio, sample_rate = load_audio("input.mp3")
print(f"Shape: {audio.shape}, Sample rate: {sample_rate} Hz")

# Get metadata
metadata = get_audio_metadata("input.mp3")
print(f"Duration: {metadata.duration_seconds:.2f}s")
print(f"Channels: {metadata.channels}")
print(f"Format: {metadata.format.value}")

# Process audio (example: normalize)
audio_normalized = audio / np.max(np.abs(audio))

# Save processed audio
save_audio(
    path="output.wav",
    audio=audio_normalized,
    sample_rate=sample_rate,
    format="wav"
)
```

## Working with Results

### Stem Results

```python
result = separator.separate("song.mp3", "output/")

# Access stems
print(f"Vocals: {result.vocals}")
print(f"Drums: {result.drums}")

# Check available stems
print(f"Available stems: {result.stem_names}")

# Access by name
bass_path = result.stems["bass"]

# Process further
from soundlab.analysis import analyze_audio
drums_analysis = analyze_audio(result.drums)
print(f"Drum tempo: {drums_analysis.tempo.bpm:.1f} BPM")
```

### MIDI Results

```python
result = transcriber.transcribe("melody.wav", "output/melody.mid")

# View note details
for note in result.notes[:5]:  # First 5 notes
    print(f"{note.pitch_name}: {note.start_time:.2f}s - {note.end_time:.2f}s")
    print(f"  Frequency: {note.frequency:.1f} Hz")
    print(f"  Duration: {note.duration_ms:.0f} ms")
    print(f"  Velocity: {note.velocity}")

# Get notes in time range
notes_in_chorus = result.get_notes_in_range(30.0, 60.0)
print(f"Notes in chorus: {len(notes_in_chorus)}")

# Generate visualization
result.save_piano_roll("output/piano_roll.png")
```

### Analysis Results

```python
result = analyze_audio("song.mp3")

# Export summary
import json

summary = result.summary
with open("analysis.json", "w") as f:
    json.dump(summary, f, indent=2)

# Check specific features
if result.tempo:
    beat_times = result.tempo.beats
    print(f"First beat at: {beat_times[0]:.2f}s")

if result.onsets:
    onset_times = result.onsets.onset_times
    print(f"Onset density: {len(onset_times) / result.duration_seconds:.1f} per second")
```

## Performance Tips

### GPU Usage

```python
from soundlab.core import get_config

# Check GPU availability
config = get_config()
print(f"Using GPU: {config.use_gpu}")
print(f"Device: {config.device}")

# Force CPU for testing
from soundlab.separation import SeparationConfig
config = SeparationConfig(device="cpu")
```

### Memory Management

```python
# For limited memory, reduce segment length
config = SeparationConfig(segment_length=5.0)

# For high memory systems, increase segment length
config = SeparationConfig(segment_length=15.0)
```

### Batch Optimization

```python
# Reuse model instances for multiple files
separator = StemSeparator()  # Model loaded once

for file in files:
    result = separator.separate(file, f"output/{file.stem}/")
    # Model stays in memory, avoiding reload overhead
```

## Next Steps

Now that you understand the basics, explore detailed guides for each feature:

- **[Stem Separation](separation.md)** - Advanced separation techniques
- **[MIDI Transcription](transcription.md)** - Fine-tuning transcription
- **[Audio Analysis](analysis.md)** - Comprehensive analysis
- **[Effects Processing](effects.md)** - Building complex effect chains
- **[Voice Generation](voice.md)** - TTS and voice cloning

!!! tip "Learn by Example"
    Check out the [example notebooks](../../notebooks/examples/) for real-world use cases and advanced techniques.

---

**Questions?** Check the [User Guide Overview](index.md) or ask in [GitHub Discussions](https://github.com/wyattwalsh/soundlab/discussions).
