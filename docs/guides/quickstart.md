# Quickstart Guide

Get started with SoundLab in minutes.

---

## Installation

### From PyPI

```bash
pip install soundlab
```

### With Optional Extras

```bash
# Include voice generation (TTS + RVC)
pip install soundlab[voice]

# Include Gradio for notebook interfaces
pip install soundlab[notebook]

# Full installation
pip install soundlab[voice,notebook]
```

### From Source

```bash
git clone https://github.com/wyattowalsh/soundlab.git
cd soundlab
pip install -e ".[voice,notebook]"
```

---

## Python Version Compatibility

| Version | Status | Notes |
|---------|--------|-------|
| Python 3.10-3.11 | ✅ Full Support | Google Colab default |
| Python 3.12 | ✅ Full Support | Recommended for local development |
| Python 3.13+ | ⚠️ Limited | `audioop` deprecated; pydub MP3 support requires `audioop-lts` |

### Python 3.13+ Workaround

If using Python 3.13+, install the audioop compatibility package:

```bash
pip install audioop-lts
```

Alternatively, use WAV or FLAC formats which don't require audioop:

```python
from soundlab import save_audio, AudioFormat

save_audio(segment, "output.flac", format=AudioFormat.FLAC)
```

---

## Basic Usage

### Load Audio

```python
from soundlab import load_audio, get_audio_metadata

# Load audio file
audio = load_audio("song.mp3")

# Access properties
print(f"Duration: {audio.duration:.2f}s")
print(f"Sample rate: {audio.sample_rate} Hz")
print(f"Channels: {audio.channels}")

# Get detailed metadata
meta = get_audio_metadata("song.mp3")
print(meta)
```

### Stem Separation

Separate a song into vocals, drums, bass, and other instruments:

```python
from soundlab.separation import StemSeparator, SeparationConfig, DemucsModel

# Configure separation
config = SeparationConfig(
    model=DemucsModel.HTDEMUCS_FT,  # Best quality
    device="auto",                   # Auto-detect GPU/CPU
)

# Create separator and process
separator = StemSeparator(config=config)
result = separator.separate("song.mp3", output_dir="stems/")

# Access results
print(f"Vocals: {result.vocals}")
print(f"Drums: {result.stems['drums']}")
print(f"Bass: {result.stems['bass']}")
print(f"Other: {result.stems['other']}")
print(f"Processing time: {result.processing_time_seconds:.1f}s")
```

### Vocal Isolation

Isolate vocals from any song with a single command:

#### CLI
```bash
soundlab separate input.mp3 output/ --vocals-only
```

#### Python
```python
from soundlab.separation import StemSeparator, SeparationConfig

config = SeparationConfig(two_stems="vocals")
separator = StemSeparator(config)
result = separator.separate("song.mp3", "output/")

print(result.vocals)  # Path to isolated vocals
print(result.instrumental)  # Path to instrumental (computed on-demand)
```

**Available Models:**

| Model | Stems | Quality | Speed |
|-------|-------|---------|-------|
| `HTDEMUCS` | 4 | Good | Fast |
| `HTDEMUCS_FT` | 4 | Best | Slow |
| `HTDEMUCS_6S` | 6 | Good | Medium |
| `MDX_EXTRA` | 4 | Good | Fast |
| `MDX_EXTRA_Q` | 4 | Better | Medium |

### Audio-to-MIDI Transcription

Convert audio to MIDI notation:

```python
from soundlab.transcription import MIDITranscriber, TranscriptionConfig

# Configure transcription
config = TranscriptionConfig(
    onset_thresh=0.5,    # Note onset sensitivity
    frame_thresh=0.3,    # Frame activation threshold
    min_note_length=0.058,  # Minimum note duration
)

# Transcribe audio
transcriber = MIDITranscriber(config=config)
result = transcriber.transcribe("piano.wav", output_dir="midi/")

# Access notes
for note in result.notes[:10]:
    print(f"Pitch: {note.pitch}, Start: {note.start:.2f}s, Duration: {note.duration:.2f}s")

# Save MIDI file
print(f"MIDI saved to: {result.path}")
```

### Drum-to-MIDI Transcription

Convert drum tracks to MIDI for re-programming or analysis:

#### Step 1: Separate Drums
```python
from soundlab.separation import StemSeparator

separator = StemSeparator()
result = separator.separate("song.mp3", "stems/")
drum_path = result.stems["drums"]
```

#### Step 2: Transcribe to MIDI
```python
from soundlab.transcription import DrumTranscriber, DrumTranscriptionConfig

config = DrumTranscriptionConfig(onset_threshold=0.3)
transcriber = DrumTranscriber(config)
midi_result = transcriber.transcribe(drum_path, "output/")

print(midi_result.path)  # drums.mid
print(len(midi_result.notes))  # Number of detected hits
```

### Audio Analysis

Analyze tempo, key, loudness, and spectral features:

```python
from soundlab import analyze_audio, detect_tempo, detect_key, measure_loudness

# Full analysis
result = analyze_audio("song.mp3")
print(f"Tempo: {result.tempo.bpm:.1f} BPM")
print(f"Key: {result.key.key_name} ({result.key.camelot})")
print(f"Loudness: {result.loudness.integrated_lufs:.1f} LUFS")

# Individual analyses
tempo = detect_tempo("song.mp3")
print(f"Tempo: {tempo.bpm:.1f} BPM (confidence: {tempo.confidence:.2f})")

key = detect_key("song.mp3")
print(f"Key: {key.key} {key.mode.value} ({key.camelot})")

loudness = measure_loudness("song.mp3")
print(f"Integrated: {loudness.integrated_lufs:.1f} LUFS")
print(f"True Peak: {loudness.true_peak_dbfs:.1f} dBFS")
```

### Effects Processing

Apply audio effects using Pedalboard:

```python
from soundlab import load_audio, save_audio
from soundlab.effects import (
    EffectsChain,
    CompressorConfig,
    ReverbConfig,
    HighpassConfig,
    LimiterConfig,
)

# Load audio
audio = load_audio("vocals.wav")

# Build effects chain
chain = EffectsChain([
    HighpassConfig(cutoff_hz=80),           # Remove rumble
    CompressorConfig(threshold_db=-20, ratio=4.0),  # Dynamic control
    ReverbConfig(room_size=0.3, wet_level=0.2),     # Add space
    LimiterConfig(threshold_db=-1.0),       # Prevent clipping
])

# Apply effects
processed = chain.process(audio.samples, audio.sample_rate)

# Save result
save_audio(processed, audio.sample_rate, "vocals_processed.wav")
```

**Available Effects:**

| Category | Effects |
|----------|---------|
| **Dynamics** | `CompressorConfig`, `LimiterConfig`, `GateConfig`, `GainConfig` |
| **EQ** | `HighpassConfig`, `LowpassConfig` |
| **Time-based** | `ReverbConfig`, `DelayConfig` |
| **Creative** | `ChorusConfig`, `PhaserConfig`, `DistortionConfig` |

### Voice Generation (Optional)

Text-to-speech and voice conversion (requires `soundlab[voice]`):

```python
from soundlab.voice import XTTSWrapper, SVCConfig
from soundlab.voice.models import TTSConfig

# Text-to-Speech with XTTS-v2
tts = XTTSWrapper()
audio = tts.synthesize(
    text="Hello, welcome to SoundLab!",
    speaker_wav="reference_voice.wav",  # Clone this voice
    language="en",
)

# Save TTS output
from soundlab import save_audio
save_audio(audio, 24000, "tts_output.wav")
```

---

## Pipeline Integration

For production workflows, use the pipeline module:

```python
from soundlab.pipeline import (
    PipelineConfig,
    QAConfig,
    init_run,
    build_candidate_plans,
    score_separation,
    choose_best_candidate,
)

# Initialize run with checkpointing
run_id, paths = init_run("song.mp3", cache_root="/path/to/cache")

# Configure pipeline
config = PipelineConfig(
    qa=QAConfig(
        min_reconstruction_score=0.9,
        max_clipping_ratio=0.01,
    ),
)

# Build and evaluate candidates
plans = build_candidate_plans(config)
# ... run separation for each plan ...
# best = choose_best_candidate(scores)
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SOUNDLAB_LOG_LEVEL` | Logging level | `INFO` |
| `SOUNDLAB_GPU_MODE` | GPU mode: `auto`, `force_gpu`, `force_cpu` | `auto` |
| `SOUNDLAB_CACHE_DIR` | Model cache directory | `~/.cache/soundlab` |
| `SOUNDLAB_SEED` | Random seed for reproducibility | None |

### Logging

```python
from soundlab.utils import configure_logging

# Set log level
configure_logging(level="DEBUG")
```

### GPU Detection

```python
from soundlab.utils import get_device

device = get_device()  # Returns "cuda" or "cpu"
print(f"Using device: {device}")
```

---

## Next Steps

- [Colab Usage Guide](colab-usage.md) — Interactive notebook walkthrough
- [Extending SoundLab](extending.md) — Add custom effects and analyzers
- [API Reference](../api/) — Full API documentation
