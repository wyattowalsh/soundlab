# Stem Separation Guide

Learn how to isolate individual instruments and vocals from mixed audio using state-of-the-art Demucs models.

## Overview

Stem separation (also called source separation) is the process of splitting a mixed audio recording into individual components like vocals, drums, bass, and other instruments. SoundLab uses Facebook Research's [Demucs](https://github.com/facebookresearch/demucs) models to achieve state-of-the-art separation quality.

### Use Cases

- **Music Production**: Extract stems for remixing and sampling
- **Karaoke Creation**: Remove vocals to create backing tracks
- **Audio Analysis**: Analyze individual instruments separately
- **Transcription**: Isolate instruments before MIDI conversion
- **DJ Mixing**: Create acapellas and instrumentals
- **Audio Repair**: Remove specific instruments or artifacts

## Quick Start

Basic stem separation:

```python
from soundlab.separation import StemSeparator

# Create separator with default settings
separator = StemSeparator()

# Separate a song
result = separator.separate(
    audio_path="song.mp3",
    output_dir="output/stems"
)

# Access separated stems
print(f"Vocals: {result.vocals}")
print(f"Drums: {result.drums}")
print(f"Bass: {result.bass}")
print(f"Other: {result.other}")
```

## Demucs Models

SoundLab supports five Demucs models, each with different characteristics:

### Model Comparison

| Model | Stems | Quality | Speed | VRAM | Best For |
|-------|-------|---------|-------|------|----------|
| **htdemucs_ft** | 4 | Excellent | Slow | 8 GB | Production work |
| **htdemucs** | 4 | Very Good | Medium | 6 GB | General use |
| **htdemucs_6s** | 6 | Good | Slow | 10 GB | Multi-instrument |
| **mdx_extra** | 4 | Very Good | Medium | 6 GB | Alternative |
| **mdx_extra_q** | 4 | Good | Fast | 4 GB | Quick processing |

### htdemucs_ft (Recommended)

Fine-tuned version with the best separation quality:

```python
from soundlab.separation import SeparationConfig, DemucsModel, StemSeparator

config = SeparationConfig(model=DemucsModel.HTDEMUCS_FT)
separator = StemSeparator(config)
result = separator.separate("song.mp3", "output/")
```

**Stems produced**: vocals, drums, bass, other

**Characteristics**:
- Best overall separation quality
- Clean vocal extraction
- Minimal artifacts
- Slowest processing time
- Recommended for professional work

### htdemucs

Original hybrid transformer model:

```python
config = SeparationConfig(model=DemucsModel.HTDEMUCS)
separator = StemSeparator(config)
result = separator.separate("song.mp3", "output/")
```

**Stems produced**: vocals, drums, bass, other

**Characteristics**:
- Excellent quality, slightly below htdemucs_ft
- 20-30% faster than htdemucs_ft
- Good balance of speed and quality
- Recommended for batch processing

### htdemucs_6s (Experimental)

Six-stem model including piano and guitar:

```python
config = SeparationConfig(model=DemucsModel.HTDEMUCS_6S)
separator = StemSeparator(config)
result = separator.separate("song.mp3", "output/")

# Access additional stems
print(f"Piano: {result.stems['piano']}")
print(f"Guitar: {result.stems['guitar']}")
```

**Stems produced**: vocals, drums, bass, other, piano, guitar

**Characteristics**:
- Six separate stems
- Piano and guitar quality varies
- Best for acoustic/jazz music
- Requires more VRAM
- Still experimental

!!! warning "Piano/Guitar Quality"
    Piano and guitar separation can be unreliable, especially in dense mixes. Test with your specific audio before production use.

### MDX Models

Alternative architecture:

```python
# Standard MDX
config = SeparationConfig(model=DemucsModel.MDX_EXTRA)

# Quantized (faster)
config = SeparationConfig(model=DemucsModel.MDX_EXTRA_Q)
```

**Characteristics**:
- Different separation approach
- May work better on some material
- `mdx_extra_q` is faster but slightly lower quality

## Configuration Options

### Core Parameters

#### segment_length

Controls processing chunk size (in seconds):

```python
# Smaller segments (lower memory, slightly lower quality)
config = SeparationConfig(segment_length=5.0)

# Default (balanced)
config = SeparationConfig(segment_length=7.8)

# Larger segments (higher memory, potentially better quality)
config = SeparationConfig(segment_length=15.0)
```

**Guidelines**:
- **4-6**: Low memory GPUs (4-6 GB VRAM)
- **7-8**: Standard (6-8 GB VRAM) - Default
- **10-15**: High memory GPUs (12+ GB VRAM)
- **CPU**: Use 5-7 for reasonable performance

!!! tip "Finding the Right Balance"
    Larger segments can improve quality at transitions but require more memory. Start with default (7.8) and adjust if you encounter memory issues.

#### overlap

Overlap between segments (0.1-0.9):

```python
# Less overlap (faster, potential artifacts at boundaries)
config = SeparationConfig(overlap=0.1)

# Default (balanced)
config = SeparationConfig(overlap=0.25)

# More overlap (slower, smoother transitions)
config = SeparationConfig(overlap=0.5)
```

**Guidelines**:
- **0.1-0.2**: Fast processing, may have boundary artifacts
- **0.25**: Default, good balance
- **0.3-0.5**: Better quality, slower
- **>0.5**: Diminishing returns, much slower

#### shifts

Number of random shifts for quality improvement:

```python
# No shifts (fastest)
config = SeparationConfig(shifts=0)

# Default (good quality)
config = SeparationConfig(shifts=1)

# More shifts (better quality, much slower)
config = SeparationConfig(shifts=2)
```

**Performance impact**:
- `shifts=0`: 1x processing time
- `shifts=1`: 2x processing time (default)
- `shifts=2`: 3x processing time

!!! info "What are shifts?"
    Shifts apply small time offsets during processing and average results, reducing artifacts and improving consistency. Each shift doubles processing time.

### Output Options

#### Audio Quality

```python
# 32-bit float (highest quality, largest files)
config = SeparationConfig(float32=True, int24=False)

# 24-bit integer (excellent quality, smaller files) - Default
config = SeparationConfig(float32=False, int24=True)

# 16-bit integer (good quality, smallest files)
config = SeparationConfig(float32=False, int24=False)
```

#### MP3 Bitrate

For MP3 output:

```python
config = SeparationConfig(
    mp3_bitrate=320  # Range: 128-320 kbps
)
```

#### Two-Stem Mode

Extract only one stem (faster):

```python
# Extract only vocals
config = SeparationConfig(two_stems="vocals")

# Extract only drums
config = SeparationConfig(two_stems="drums")
```

**Available options**: `"vocals"`, `"drums"`, `"bass"`, `"other"`

**Benefits**:
- Faster processing (2-3x)
- Lower memory usage
- Same quality for target stem

### Device Selection

```python
# Automatic GPU detection (default)
config = SeparationConfig(device="auto")

# Force CPU
config = SeparationConfig(device="cpu")

# Specific GPU
config = SeparationConfig(device="cuda:0")

# Multiple GPUs (not currently supported)
config = SeparationConfig(device="cuda:1")
```

## Usage Examples

### Example 1: High-Quality Production

For professional music production:

```python
from soundlab.separation import SeparationConfig, DemucsModel, StemSeparator

config = SeparationConfig(
    model=DemucsModel.HTDEMUCS_FT,
    segment_length=10.0,
    overlap=0.25,
    shifts=2,
    int24=True
)

separator = StemSeparator(config)
result = separator.separate(
    audio_path="master.wav",
    output_dir="output/stems_hq"
)

print(f"High-quality separation complete in {result.processing_time_seconds:.1f}s")
```

### Example 2: Fast Batch Processing

Process multiple files quickly:

```python
from pathlib import Path

config = SeparationConfig(
    model=DemucsModel.HTDEMUCS,
    segment_length=7.8,
    overlap=0.1,
    shifts=0,
    float32=True  # Faster processing
)

separator = StemSeparator(config)

for audio_file in Path("input/").glob("*.mp3"):
    print(f"Processing: {audio_file.name}")
    result = separator.separate(
        audio_path=audio_file,
        output_dir=f"output/{audio_file.stem}"
    )
    print(f"  Time: {result.processing_time_seconds:.1f}s")
```

### Example 3: Vocal Extraction Only

Extract just vocals (acapella):

```python
config = SeparationConfig(
    model=DemucsModel.HTDEMUCS_FT,
    two_stems="vocals",  # Only extract vocals
    int24=True
)

separator = StemSeparator(config)
result = separator.separate(
    audio_path="song.mp3",
    output_dir="output/vocals_only"
)

print(f"Vocals saved to: {result.vocals}")
```

### Example 4: Instrumental (No Vocals)

Create instrumental/backing track:

```python
# Method 1: Extract "other" stem (everything except vocals)
config = SeparationConfig(two_stems="other")
separator = StemSeparator(config)
result = separator.separate("song.mp3", "output/")
instrumental = result.other

# Method 2: Mix non-vocal stems
from soundlab.io import load_audio, save_audio
import numpy as np

result = separator.separate("song.mp3", "output/")

# Load all non-vocal stems
drums, sr = load_audio(result.drums)
bass, _ = load_audio(result.bass)
other, _ = load_audio(result.other)

# Mix them together
instrumental = drums + bass + other

# Normalize to prevent clipping
instrumental = instrumental / np.max(np.abs(instrumental)) * 0.9

save_audio("instrumental.wav", instrumental, sr)
```

### Example 5: Drum Isolation for Analysis

Isolate drums for tempo/beat analysis:

```python
from soundlab.separation import StemSeparator
from soundlab.analysis import detect_tempo

# Separate drums
separator = StemSeparator()
result = separator.separate("song.mp3", "output/")

# Analyze drums specifically
tempo_result = detect_tempo(result.drums)
print(f"Drum tempo: {tempo_result.bpm:.1f} BPM")
print(f"Confidence: {tempo_result.confidence:.2%}")

# Beat times
for i, beat_time in enumerate(tempo_result.beats[:10], 1):
    print(f"Beat {i}: {beat_time:.2f}s")
```

### Example 6: Multi-Instrument Separation

Extract 6 stems including piano and guitar:

```python
config = SeparationConfig(model=DemucsModel.HTDEMUCS_6S)
separator = StemSeparator(config)
result = separator.separate("jazz_song.mp3", "output/")

# Access all stems
stems = result.stems
for stem_name, stem_path in stems.items():
    print(f"{stem_name}: {stem_path}")

# Expected output:
# vocals: output/vocals.wav
# drums: output/drums.wav
# bass: output/bass.wav
# other: output/other.wav
# piano: output/piano.wav
# guitar: output/guitar.wav
```

### Example 7: Custom Output Format

Save stems in specific format:

```python
from soundlab.io import load_audio, save_audio

separator = StemSeparator()
result = separator.separate("song.mp3", "output/temp/")

# Convert to FLAC with custom settings
for stem_name, stem_path in result.stems.items():
    audio, sr = load_audio(stem_path)
    save_audio(
        path=f"output/flac/{stem_name}.flac",
        audio=audio,
        sample_rate=sr,
        format="flac"
    )
```

## Progress Monitoring

### Simple Progress Bar

```python
def progress_callback(step: str, percent: float):
    bar_length = 50
    filled = int(bar_length * percent / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\r{step}: [{bar}] {percent:.1f}%", end="", flush=True)

separator = StemSeparator()
result = separator.separate(
    audio_path="song.mp3",
    output_dir="output/",
    progress_callback=progress_callback
)
print("\n✓ Complete!")
```

### Detailed Progress Logging

```python
from datetime import datetime

def detailed_progress(step: str, percent: float):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {step}: {percent:.1f}%")

separator = StemSeparator()
result = separator.separate(
    audio_path="song.mp3",
    output_dir="output/",
    progress_callback=detailed_progress
)
```

### tqdm Integration

```python
from tqdm import tqdm

class TqdmProgress:
    def __init__(self):
        self.pbar = None
        self.current_step = None

    def __call__(self, step: str, percent: float):
        if step != self.current_step:
            if self.pbar:
                self.pbar.close()
            self.current_step = step
            self.pbar = tqdm(total=100, desc=step)

        if self.pbar:
            self.pbar.n = percent
            self.pbar.refresh()

        if percent >= 100:
            if self.pbar:
                self.pbar.close()

progress = TqdmProgress()
separator = StemSeparator()
result = separator.separate(
    audio_path="song.mp3",
    output_dir="output/",
    progress_callback=progress
)
```

## Best Practices

### Choose the Right Model

=== "Production/Professional"
    ```python
    # Highest quality, worth the wait
    config = SeparationConfig(
        model=DemucsModel.HTDEMUCS_FT,
        shifts=1 or 2
    )
    ```

=== "General Use"
    ```python
    # Great balance of speed and quality
    config = SeparationConfig(
        model=DemucsModel.HTDEMUCS,
        shifts=1
    )
    ```

=== "Batch/Preview"
    ```python
    # Fast processing for many files
    config = SeparationConfig(
        model=DemucsModel.HTDEMUCS,
        shifts=0,
        overlap=0.1
    )
    ```

=== "Limited Memory"
    ```python
    # For low VRAM or CPU
    config = SeparationConfig(
        model=DemucsModel.MDX_EXTRA_Q,
        segment_length=5.0,
        device="cpu"
    )
    ```

### Input Audio Quality

For best results:

- **Use high-quality source audio** (WAV, FLAC, or high-bitrate MP3)
- **Avoid heavily compressed audio** (low-bitrate MP3s, YouTube rips)
- **Use mastered/mixed audio** (not raw tracks)
- **Avoid clipped audio** (check for distortion)

```python
# Check audio quality before processing
from soundlab.io import get_audio_metadata

metadata = get_audio_metadata("song.mp3")
print(f"Sample rate: {metadata.sample_rate} Hz")
print(f"Bit depth: {metadata.bit_depth}")
print(f"Duration: {metadata.duration_seconds:.2f}s")

# Recommend minimum 44.1 kHz, 16-bit
if metadata.sample_rate < 44100:
    print("Warning: Low sample rate may affect quality")
```

### Memory Management

Monitor and manage GPU memory:

```python
import torch

# Check available memory before processing
if torch.cuda.is_available():
    mem_total = torch.cuda.get_device_properties(0).total_memory
    mem_allocated = torch.cuda.memory_allocated(0)
    mem_free = mem_total - mem_allocated
    print(f"Free GPU memory: {mem_free / 1e9:.1f} GB")

    # Adjust segment length based on available memory
    if mem_free < 4e9:  # Less than 4 GB
        config = SeparationConfig(segment_length=5.0)
    elif mem_free > 12e9:  # More than 12 GB
        config = SeparationConfig(segment_length=15.0)
    else:
        config = SeparationConfig(segment_length=7.8)
```

### Quality Validation

Verify separation quality:

```python
from soundlab.analysis import analyze_audio
import numpy as np

# Separate stems
result = separator.separate("song.mp3", "output/")

# Analyze each stem
for stem_name, stem_path in result.stems.items():
    analysis = analyze_audio(stem_path)

    # Check for silence (potential separation failure)
    from soundlab.io import load_audio
    audio, sr = load_audio(stem_path)
    rms = np.sqrt(np.mean(audio ** 2))

    print(f"\n{stem_name}:")
    print(f"  RMS level: {20 * np.log10(rms):.1f} dB")
    print(f"  Duration: {analysis.duration_seconds:.2f}s")

    if rms < 0.001:
        print(f"  Warning: {stem_name} may be silent")
```

## Common Issues and Solutions

### GPU Out of Memory

**Problem**: CUDA out of memory error during processing.

**Solutions**:

```python
# 1. Reduce segment length
config = SeparationConfig(segment_length=5.0)

# 2. Use CPU
config = SeparationConfig(device="cpu")

# 3. Use smaller model
config = SeparationConfig(model=DemucsModel.MDX_EXTRA_Q)

# 4. Clear GPU cache before processing
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Poor Separation Quality

**Problem**: Stems have artifacts or bleed from other instruments.

**Solutions**:

```python
# 1. Use best quality model
config = SeparationConfig(model=DemucsModel.HTDEMUCS_FT)

# 2. Enable shifts
config = SeparationConfig(shifts=2)

# 3. Increase overlap
config = SeparationConfig(overlap=0.3)

# 4. Try different model architecture
config = SeparationConfig(model=DemucsModel.MDX_EXTRA)
```

### Slow Processing

**Problem**: Processing takes too long.

**Solutions**:

```python
# 1. Use faster model
config = SeparationConfig(model=DemucsModel.HTDEMUCS)

# 2. Disable shifts
config = SeparationConfig(shifts=0)

# 3. Reduce overlap
config = SeparationConfig(overlap=0.1)

# 4. Use two-stem mode if only one stem needed
config = SeparationConfig(two_stems="vocals")

# 5. Ensure GPU is being used
import torch
print(f"GPU available: {torch.cuda.is_available()}")
```

### Clipping/Distortion

**Problem**: Output stems have clipping artifacts.

**Solutions**:

```python
# 1. Use floating-point output
config = SeparationConfig(float32=True)

# 2. Normalize stems after separation
from soundlab.io import load_audio, save_audio
import numpy as np

for stem_name, stem_path in result.stems.items():
    audio, sr = load_audio(stem_path)

    # Check for clipping
    if np.max(np.abs(audio)) > 0.99:
        print(f"Normalizing {stem_name}...")
        audio = audio / np.max(np.abs(audio)) * 0.9
        save_audio(stem_path, audio, sr)
```

## Performance Benchmarks

Typical processing times for a 3-minute song:

### GPU (NVIDIA T4)

| Configuration | Time | Quality |
|--------------|------|---------|
| htdemucs_ft, shifts=2 | ~5 min | Excellent |
| htdemucs_ft, shifts=1 | ~3 min | Excellent |
| htdemucs, shifts=1 | ~2 min | Very Good |
| htdemucs, shifts=0 | ~1.5 min | Very Good |
| mdx_extra_q, shifts=0 | ~1 min | Good |

### CPU (Intel i9-12900K)

| Configuration | Time | Quality |
|--------------|------|---------|
| htdemucs_ft | ~45 min | Excellent |
| htdemucs | ~30 min | Very Good |
| mdx_extra_q | ~15 min | Good |

!!! tip "Performance Optimization"
    - GPU is 10-20x faster than CPU
    - Shifts double/triple processing time
    - Two-stem mode is 2-3x faster than four-stem
    - Longer segments slightly faster but use more memory

## Advanced Techniques

### Iterative Separation

Separate stems multiple times for ultra-clean isolation:

```python
# First pass: Extract vocals
config = SeparationConfig(
    model=DemucsModel.HTDEMUCS_FT,
    two_stems="vocals"
)
separator = StemSeparator(config)
result = separator.separate("song.mp3", "output/pass1/")

# Second pass: Separate remaining instruments from "other"
result2 = separator.separate(result.other, "output/pass2/")
```

### Ensemble Separation

Combine multiple models for best results:

```python
from soundlab.io import load_audio, save_audio
import numpy as np

# Separate with multiple models
configs = [
    SeparationConfig(model=DemucsModel.HTDEMUCS_FT),
    SeparationConfig(model=DemucsModel.MDX_EXTRA),
]

stems_list = []
for i, config in enumerate(configs):
    separator = StemSeparator(config)
    result = separator.separate("song.mp3", f"output/model{i}/")
    stems_list.append(result.stems)

# Average the results
for stem_name in ["vocals", "drums", "bass", "other"]:
    stem_audios = []
    for stems in stems_list:
        audio, sr = load_audio(stems[stem_name])
        stem_audios.append(audio)

    # Average
    avg_audio = np.mean(stem_audios, axis=0)
    save_audio(f"output/ensemble/{stem_name}.wav", avg_audio, sr)
```

## Next Steps

- **[Effects Processing](effects.md)** - Apply effects to separated stems
- **[MIDI Transcription](transcription.md)** - Convert stems to MIDI
- **[Audio Analysis](analysis.md)** - Analyze individual stems

---

**Need help?** Check the [common issues](#common-issues-and-solutions) or ask in [GitHub Discussions](https://github.com/wyattwalsh/soundlab/discussions).
