# Audio Analysis Guide

Learn how to extract musical features from audio including tempo, key, loudness, spectral characteristics, and onset detection.

## Overview

SoundLab provides comprehensive audio analysis capabilities built on [librosa](https://librosa.org/) and [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm), enabling you to extract:

- **Tempo/BPM** - Beat detection and tempo estimation
- **Key Detection** - Musical key using Krumhansl-Schmuckler algorithm
- **Loudness** - LUFS, true peak, and dynamic range
- **Spectral Features** - Centroid, bandwidth, rolloff, flatness
- **Onset Detection** - Transient and attack timing

### Use Cases

- **Music Organization** - Auto-tag tracks with BPM and key
- **DJ Mixing** - Find compatible tracks (Camelot wheel)
- **Mastering** - Measure and optimize loudness
- **Sound Design** - Analyze spectral characteristics
- **Music Information Retrieval** - Extract features for ML
- **Quality Control** - Verify audio specifications

## Quick Start

Comprehensive analysis:

```python
from soundlab.analysis import analyze_audio

# Analyze all features
result = analyze_audio("song.mp3")

# View results
print(f"Duration: {result.duration_seconds:.2f}s")
print(f"Tempo: {result.tempo.bpm:.1f} BPM")
print(f"Key: {result.key.name}")
print(f"Loudness: {result.loudness.integrated_lufs:.1f} LUFS")
print(f"Spectral Centroid: {result.spectral.spectral_centroid:.1f} Hz")
print(f"Onsets: {result.onsets.onset_count}")

# Get summary dict
summary = result.summary
for key, value in summary.items():
    print(f"{key}: {value}")
```

## Tempo Detection

Detect beats per minute (BPM) and beat times.

### Basic Usage

```python
from soundlab.analysis import detect_tempo

result = detect_tempo("song.mp3")

print(f"Tempo: {result.bpm:.1f} BPM")
print(f"Confidence: {result.confidence:.2%}")
print(f"Beat interval: {result.beat_interval:.3f}s")
print(f"Total beats: {result.beat_count}")

# Access beat timestamps
for i, beat_time in enumerate(result.beats[:10], 1):
    print(f"Beat {i}: {beat_time:.3f}s")
```

### Tempo Analysis Features

```python
result = detect_tempo("song.mp3")

# Calculate measures (assuming 4/4 time)
beats_per_measure = 4
measure_count = result.beat_count // beats_per_measure
print(f"Estimated measures: {measure_count}")

# Find downbeats (every 4th beat in 4/4)
downbeats = result.beats[::4]
print(f"Downbeats: {len(downbeats)}")

# Analyze tempo stability
beat_intervals = [
    result.beats[i+1] - result.beats[i]
    for i in range(len(result.beats) - 1)
]
import numpy as np
tempo_std = np.std(beat_intervals)
print(f"Tempo stability (std): {tempo_std:.4f}s")
if tempo_std < 0.01:
    print("Tempo is very stable (likely electronic)")
else:
    print("Tempo has variation (likely live/acoustic)")
```

### Use Cases

#### DJ Track Matching

```python
from soundlab.analysis import detect_tempo

track1_tempo = detect_tempo("track1.mp3")
track2_tempo = detect_tempo("track2.mp3")

bpm_diff = abs(track1_tempo.bpm - track2_tempo.bpm)
print(f"BPM difference: {bpm_diff:.1f}")

if bpm_diff < 3:
    print("✓ Tracks are mixable (similar tempo)")
elif bpm_diff < 10:
    print("~ Tracks may need pitch adjustment")
else:
    print("✗ Tracks are incompatible (large tempo difference)")
```

#### Metronome Sync

```python
result = detect_tempo("song.mp3")

# Generate click track aligned to beats
from soundlab.io import save_audio
import numpy as np

sample_rate = 44100
click_duration = 0.05  # 50ms clicks

audio = np.zeros(int(result.duration * sample_rate))

for beat_time in result.beats:
    click_start = int(beat_time * sample_rate)
    click_end = click_start + int(click_duration * sample_rate)
    if click_end < len(audio):
        # Generate click (1 kHz sine tone)
        t = np.linspace(0, click_duration, click_end - click_start)
        click = np.sin(2 * np.pi * 1000 * t) * 0.5
        audio[click_start:click_end] = click

save_audio("click_track.wav", audio, sample_rate)
```

## Key Detection

Detect musical key using Krumhansl-Schmuckler algorithm.

### Basic Usage

```python
from soundlab.analysis import detect_key

result = detect_key("song.mp3")

print(f"Key: {result.name}")                    # e.g., "A minor"
print(f"Note: {result.key.value}")              # e.g., "A"
print(f"Mode: {result.mode.value}")             # e.g., "minor"
print(f"Confidence: {result.confidence:.2%}")

# DJ mixing notations
print(f"Camelot: {result.camelot}")             # e.g., "8A"
print(f"Open Key: {result.open_key}")           # e.g., "1m"
```

### Key Relationships

```python
result = detect_key("song.mp3")

# View all key correlations
print("Key correlations:")
for key_name, correlation in sorted(
    result.all_correlations.items(),
    key=lambda x: x[1],
    reverse=True
)[:5]:
    print(f"  {key_name}: {correlation:.3f}")

# Compatible keys for mixing (Camelot wheel)
def get_compatible_keys(camelot):
    """Get harmonically compatible keys."""
    # Same key, adjacent numbers, same number opposite mode
    number = int(''.join(filter(str.isdigit, camelot)))
    letter = camelot[-1]

    compatible = [
        camelot,                           # Same key
        f"{number}{('A' if letter == 'B' else 'B')}",  # Relative major/minor
        f"{number+1 if number < 12 else 1}{letter}",   # +1
        f"{number-1 if number > 1 else 12}{letter}",   # -1
    ]
    return compatible

compatible = get_compatible_keys(result.camelot)
print(f"\nCompatible keys for {result.name} ({result.camelot}):")
for key in compatible:
    print(f"  {key}")
```

### Use Cases

#### Music Library Organization

```python
from pathlib import Path
from soundlab.analysis import detect_key
import json

library = Path("music_library/")
key_database = {}

for audio_file in library.glob("**/*.mp3"):
    result = detect_key(audio_file)
    key_database[str(audio_file)] = {
        "key": result.name,
        "camelot": result.camelot,
        "open_key": result.open_key,
        "confidence": result.confidence
    }

# Save database
with open("key_database.json", "w") as f:
    json.dump(key_database, f, indent=2)

# Find all tracks in A minor
a_minor_tracks = [
    path for path, data in key_database.items()
    if data["key"] == "A minor"
]
print(f"Tracks in A minor: {len(a_minor_tracks)}")
```

#### Harmonic Mixing

```python
def find_mixable_tracks(track_key, key_database):
    """Find tracks that mix harmonically with given track."""
    track_data = key_database[track_key]
    compatible_camelots = get_compatible_keys(track_data["camelot"])

    mixable = []
    for path, data in key_database.items():
        if path != track_key and data["camelot"] in compatible_camelots:
            mixable.append({
                "path": path,
                "key": data["key"],
                "camelot": data["camelot"]
            })

    return mixable

# Find tracks that mix with current track
mixable = find_mixable_tracks("current_track.mp3", key_database)
print(f"Found {len(mixable)} mixable tracks")
```

## Loudness Measurement

Measure loudness using ITU-R BS.1770-4 standard (LUFS).

### Basic Usage

```python
from soundlab.analysis import measure_loudness

result = measure_loudness("song.mp3")

print(f"Integrated Loudness: {result.integrated_lufs:.1f} LUFS")
print(f"True Peak: {result.true_peak_db:.1f} dBTP")
print(f"Loudness Range: {result.loudness_range:.1f} LU")
print(f"Dynamic Range: {result.dynamic_range_db:.1f} dB")

# Check broadcast standards
print(f"Broadcast Safe: {result.is_broadcast_safe}")
print(f"Streaming Optimized: {result.is_streaming_optimized}")
```

### Loudness Standards

```python
result = measure_loudness("song.mp3")

# Check against various standards
standards = {
    "Spotify": (-14.0, "LUFS"),
    "Apple Music": (-16.0, "LUFS"),
    "YouTube": (-14.0, "LUFS"),
    "TV Broadcast": (-23.0, "LUFS"),
    "Film": (-24.0, "LUFS"),
    "CD Maximum": (0.0, "dBTP"),
}

print(f"\nLoudness: {result.integrated_lufs:.1f} LUFS")
print(f"True Peak: {result.true_peak_db:.1f} dBTP")

for platform, (target, unit) in standards.items():
    if unit == "LUFS":
        diff = result.integrated_lufs - target
        if abs(diff) < 1.0:
            status = "✓"
        elif abs(diff) < 2.0:
            status = "~"
        else:
            status = "✗"
        print(f"{status} {platform}: {diff:+.1f} LU")
    elif unit == "dBTP":
        if result.true_peak_db <= target:
            print(f"✓ {platform}: Peak OK")
        else:
            print(f"✗ {platform}: Peak too high ({result.true_peak_db - target:+.1f} dB)")
```

### Use Cases

#### Mastering Target Check

```python
def check_mastering_targets(audio_path, target_lufs=-14.0, max_peak=-1.0):
    """Check if audio meets mastering targets."""
    result = measure_loudness(audio_path)

    print(f"Target: {target_lufs} LUFS, {max_peak} dBTP")
    print(f"Current: {result.integrated_lufs:.1f} LUFS, {result.true_peak_db:.1f} dBTP")

    lufs_diff = result.integrated_lufs - target_lufs
    peak_diff = result.true_peak_db - max_peak

    issues = []

    if lufs_diff < -2.0:
        issues.append(f"Too quiet ({lufs_diff:.1f} LU)")
    elif lufs_diff > 2.0:
        issues.append(f"Too loud ({lufs_diff:+.1f} LU)")

    if result.true_peak_db > max_peak:
        issues.append(f"Peak too high ({peak_diff:+.1f} dB)")

    if result.dynamic_range_db < 5.0:
        issues.append(f"Over-compressed (DR{result.dynamic_range_db:.1f})")

    if issues:
        print("\n✗ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ Mastering targets met!")
        return True

check_mastering_targets("master.wav")
```

#### Gain Adjustment Calculation

```python
def calculate_gain_adjustment(audio_path, target_lufs=-14.0):
    """Calculate gain needed to reach target loudness."""
    result = measure_loudness(audio_path)

    current_lufs = result.integrated_lufs
    gain_needed = target_lufs - current_lufs

    print(f"Current loudness: {current_lufs:.1f} LUFS")
    print(f"Target loudness: {target_lufs:.1f} LUFS")
    print(f"Gain adjustment needed: {gain_needed:+.1f} dB")

    # Check if adjustment would cause clipping
    new_peak = result.true_peak_db + gain_needed
    if new_peak > -1.0:
        print(f"Warning: Adjustment would clip (peak: {new_peak:.1f} dBTP)")
        safe_gain = -1.0 - result.true_peak_db
        print(f"Safe gain: {safe_gain:+.1f} dB (results in {current_lufs + safe_gain:.1f} LUFS)")

    return gain_needed

gain = calculate_gain_adjustment("song.mp3", target_lufs=-14.0)
```

## Spectral Analysis

Analyze frequency content and spectral characteristics.

### Basic Usage

```python
from soundlab.analysis import analyze_audio

result = analyze_audio("song.mp3")

if result.spectral:
    print(f"Spectral Centroid: {result.spectral.spectral_centroid:.1f} Hz")
    print(f"Spectral Bandwidth: {result.spectral.spectral_bandwidth:.1f} Hz")
    print(f"Spectral Rolloff: {result.spectral.spectral_rolloff:.1f} Hz")
    print(f"Spectral Flatness: {result.spectral.spectral_flatness:.3f}")
    print(f"Zero Crossing Rate: {result.spectral.zero_crossing_rate:.4f}")
    print(f"Brightness: {result.spectral.brightness}")
```

### Feature Interpretation

```python
result = analyze_audio("song.mp3")

if result.spectral:
    # Spectral centroid - "center of mass" of spectrum
    centroid = result.spectral.spectral_centroid
    if centroid < 1500:
        print("Sound character: Dark/warm")
    elif centroid < 3000:
        print("Sound character: Balanced")
    else:
        print("Sound character: Bright/harsh")

    # Spectral flatness - noisiness vs. tonality
    flatness = result.spectral.spectral_flatness
    if flatness < 0.1:
        print("Spectral character: Tonal (musical)")
    elif flatness < 0.5:
        print("Spectral character: Mixed")
    else:
        print("Spectral character: Noise-like")

    # Zero crossing rate - roughness/noisiness
    zcr = result.spectral.zero_crossing_rate
    if zcr < 0.05:
        print("Temporal character: Smooth/sustained")
    else:
        print("Temporal character: Percussive/noisy")
```

### Use Cases

#### Genre Classification Features

```python
def extract_genre_features(audio_path):
    """Extract features useful for genre classification."""
    result = analyze_audio(audio_path)

    features = {
        "tempo": result.tempo.bpm if result.tempo else None,
        "spectral_centroid": result.spectral.spectral_centroid if result.spectral else None,
        "spectral_flatness": result.spectral.spectral_flatness if result.spectral else None,
        "zero_crossing_rate": result.spectral.zero_crossing_rate if result.spectral else None,
        "dynamic_range": result.loudness.dynamic_range_db if result.loudness else None,
    }

    # Genre indicators
    if features["tempo"] and features["tempo"] > 140:
        print("Likely: Electronic/Dance")
    elif features["spectral_flatness"] and features["spectral_flatness"] > 0.5:
        print("Likely: Metal/Rock (distorted)")
    elif features["dynamic_range"] and features["dynamic_range"] > 10:
        print("Likely: Classical/Jazz (high dynamics)")

    return features

features = extract_genre_features("song.mp3")
```

#### Audio Quality Assessment

```python
def assess_audio_quality(audio_path):
    """Assess audio quality from spectral features."""
    result = analyze_audio(audio_path)

    issues = []

    # Check frequency content
    if result.spectral:
        if result.spectral.spectral_rolloff < 8000:
            issues.append("Limited high-frequency content (poor quality source?)")

        if result.spectral.spectral_centroid < 500:
            issues.append("Muddy/boomy (too much low-end)")

    # Check dynamic range
    if result.loudness and result.loudness.dynamic_range_db < 5:
        issues.append("Over-compressed (low dynamic range)")

    if issues:
        print("Quality issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Audio quality: Good")

assess_audio_quality("song.mp3")
```

## Onset Detection

Detect transients and attack times.

### Basic Usage

```python
from soundlab.analysis import analyze_audio

result = analyze_audio("drums.mp3")

if result.onsets:
    print(f"Onset count: {result.onsets.onset_count}")
    print(f"Average interval: {result.onsets.average_interval:.3f}s")

    # First 10 onsets
    for i, time in enumerate(result.onsets.onset_times[:10], 1):
        strength = result.onsets.onset_strengths[i-1]
        print(f"Onset {i}: {time:.3f}s (strength: {strength:.3f})")
```

### Use Cases

#### Drum Hit Detection

```python
from soundlab.separation import StemSeparator
from soundlab.analysis import analyze_audio

# Separate drums
separator = StemSeparator()
stems = separator.separate("song.mp3", "output/stems/")

# Analyze drum onsets
result = analyze_audio(stems.drums)

if result.onsets:
    # Analyze rhythm
    intervals = []
    for i in range(len(result.onsets.onset_times) - 1):
        interval = result.onsets.onset_times[i+1] - result.onsets.onset_times[i]
        intervals.append(interval)

    import numpy as np
    avg_interval = np.mean(intervals)
    std_interval = np.std(intervals)

    print(f"Average hit interval: {avg_interval:.3f}s")
    print(f"Rhythm consistency: {std_interval:.3f}s")

    if std_interval < 0.05:
        print("Rhythm: Very consistent (likely programmed)")
    else:
        print("Rhythm: Variable (likely live drums)")
```

#### Beat Grid Alignment

```python
def align_to_grid(onset_times, grid_interval=0.5):
    """Align onsets to rhythmic grid."""
    import numpy as np

    aligned = []
    for time in onset_times:
        aligned_time = np.round(time / grid_interval) * grid_interval
        aligned.append(aligned_time)

    return aligned

result = analyze_audio("percussion.wav")
if result.onsets:
    aligned_onsets = align_to_grid(result.onsets.onset_times, grid_interval=0.25)

    print("Original vs Aligned:")
    for orig, aligned in zip(result.onsets.onset_times[:5], aligned_onsets[:5]):
        diff = abs(aligned - orig)
        print(f"{orig:.3f}s -> {aligned:.3f}s (Δ{diff*1000:.1f}ms)")
```

## Comprehensive Analysis

Analyze all features at once:

```python
from soundlab.analysis import analyze_audio

result = analyze_audio("song.mp3")

# Basic info
print(f"Duration: {result.duration_seconds:.2f}s")
print(f"Sample Rate: {result.sample_rate} Hz")
print(f"Channels: {result.channels}")

# Tempo
if result.tempo:
    print(f"\nTempo: {result.tempo.bpm:.1f} BPM")
    print(f"Confidence: {result.tempo.confidence:.2%}")
    print(f"Beats detected: {result.tempo.beat_count}")

# Key
if result.key:
    print(f"\nKey: {result.key.name}")
    print(f"Camelot: {result.key.camelot}")
    print(f"Confidence: {result.key.confidence:.2%}")

# Loudness
if result.loudness:
    print(f"\nLoudness: {result.loudness.integrated_lufs:.1f} LUFS")
    print(f"True Peak: {result.loudness.true_peak_db:.1f} dBTP")
    print(f"Dynamic Range: {result.loudness.dynamic_range_db:.1f} dB")
    print(f"Broadcast Safe: {result.loudness.is_broadcast_safe}")

# Spectral
if result.spectral:
    print(f"\nSpectral Centroid: {result.spectral.spectral_centroid:.1f} Hz")
    print(f"Brightness: {result.spectral.brightness}")
    print(f"Spectral Flatness: {result.spectral.spectral_flatness:.3f}")

# Onsets
if result.onsets:
    print(f"\nOnsets: {result.onsets.onset_count}")
    print(f"Avg interval: {result.onsets.average_interval:.3f}s")

# Export summary
import json
with open("analysis.json", "w") as f:
    json.dump(result.summary, f, indent=2)
```

## Batch Analysis

Analyze multiple files:

```python
from pathlib import Path
from soundlab.analysis import analyze_audio
import json

def analyze_library(library_path, output_file="analysis_results.json"):
    """Analyze all audio files in a directory."""
    library = Path(library_path)
    results = {}

    for audio_file in library.glob("**/*.mp3"):
        print(f"Analyzing: {audio_file.name}")

        try:
            result = analyze_audio(audio_file)
            results[str(audio_file)] = result.summary
        except Exception as e:
            print(f"  Error: {e}")
            results[str(audio_file)] = {"error": str(e)}

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalyzed {len(results)} files")
    return results

# Analyze library
results = analyze_library("music_library/")

# Generate statistics
bpms = [data.get("bpm") for data in results.values() if "bpm" in data]
if bpms:
    import numpy as np
    print(f"\nLibrary BPM range: {min(bpms):.1f} - {max(bpms):.1f}")
    print(f"Average BPM: {np.mean(bpms):.1f}")
```

## Best Practices

### Optimize for Specific Features

```python
# For tempo detection only (faster)
from soundlab.analysis.tempo import detect_tempo
result = detect_tempo("song.mp3")

# For key detection only
from soundlab.analysis.key import detect_key
result = detect_key("song.mp3")

# For loudness measurement only
from soundlab.analysis.loudness import measure_loudness
result = measure_loudness("song.mp3")

# For all features (more comprehensive)
from soundlab.analysis import analyze_audio
result = analyze_audio("song.mp3")
```

### Handle Edge Cases

```python
from soundlab.core import SoundLabError

try:
    result = analyze_audio("song.mp3")

    # Check if features were extracted
    if result.tempo is None:
        print("Warning: Tempo detection failed")

    if result.key is None:
        print("Warning: Key detection failed")

except SoundLabError as e:
    print(f"Analysis error: {e}")
```

## Performance

Analysis times for 3-minute song:

| Feature | Time | Notes |
|---------|------|-------|
| Tempo | ~5 sec | Beat tracking |
| Key | ~3 sec | Chroma analysis |
| Loudness | ~2 sec | Fast |
| Spectral | ~3 sec | Per-frame analysis |
| Onsets | ~2 sec | Fast |
| **Complete** | ~10 sec | All features |

## Next Steps

- **[Separation Guide](separation.md)** - Analyze individual stems
- **[Effects Guide](effects.md)** - Process based on analysis
- **[Transcription Guide](transcription.md)** - Use key detection for transcription

---

**Questions?** Visit [GitHub Discussions](https://github.com/wyattwalsh/soundlab/discussions).
