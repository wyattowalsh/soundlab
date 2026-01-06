# Analysis

Comprehensive audio analysis for extracting musical features.

## Overview

The `soundlab.analysis` module provides tools for extracting musical and acoustic features from audio files. Built on librosa, it offers tempo detection, key detection, loudness measurement, spectral analysis, and onset detection.

## Features

- **Tempo Detection**: BPM detection with confidence scoring
- **Key Detection**: Musical key identification using Krumhansl-Schmuckler algorithm
- **Loudness Measurement**: ITU-R BS.1770-4 compliant LUFS measurement
- **Spectral Analysis**: Extract spectral centroid, bandwidth, rolloff, and brightness
- **Onset Detection**: Detect note onsets and transients
- **Camelot System**: DJ-friendly key notation for harmonic mixing

## Key Components

### AudioAnalyzer
The main analysis interface that performs comprehensive audio feature extraction.

### analyze_audio()
Convenience function for one-shot analysis of all features.

### detect_tempo()
Dedicated BPM detection with multiple estimation methods.

### detect_key()
Musical key detection with Camelot notation.

### measure_loudness()
LUFS loudness measurement with dynamic range analysis.

## Usage Examples

### Comprehensive Analysis

```python
from soundlab.analysis import analyze_audio

# Perform full analysis
result = analyze_audio("song.mp3")

# Tempo information
print(f"BPM: {result.tempo.bpm:.1f}")
print(f"Confidence: {result.tempo.confidence:.2f}")

# Key detection
print(f"Key: {result.key.name}")           # e.g., "A minor"
print(f"Camelot: {result.key.camelot}")    # e.g., "8A"
print(f"Root note: {result.key.root_note}") # e.g., "A"
print(f"Mode: {result.key.mode}")          # "major" or "minor"

# Loudness
print(f"Integrated LUFS: {result.loudness.lufs:.1f}")
print(f"Peak dB: {result.loudness.peak_db:.1f}")
print(f"Dynamic range: {result.loudness.dynamic_range:.1f} dB")

# Spectral features
print(f"Spectral centroid: {result.spectral.centroid_mean:.1f} Hz")
print(f"Bandwidth: {result.spectral.bandwidth_mean:.1f} Hz")
print(f"Brightness: {result.spectral.brightness:.2f}")

# Onsets
print(f"Onsets detected: {len(result.onsets.timestamps)}")
print(f"Average onset strength: {result.onsets.strength_mean:.2f}")
```

### Tempo Detection

```python
from soundlab.analysis import detect_tempo

# Detect BPM
result = detect_tempo("song.mp3")

print(f"Primary BPM: {result.bpm:.1f}")
print(f"Confidence: {result.confidence:.2f}")

# Alternative tempos (half-time, double-time)
if result.alternative_bpms:
    print("Alternative tempos:")
    for alt_bpm in result.alternative_bpms:
        print(f"  {alt_bpm:.1f} BPM")

# Tempo stability (how consistent)
if result.tempo_stability:
    print(f"Stability: {result.tempo_stability:.2f}")
```

### Key Detection

```python
from soundlab.analysis import detect_key

# Detect musical key
result = detect_key("melody.mp3")

print(f"Key: {result.name}")              # "C# major"
print(f"Root: {result.root_note}")        # "C#"
print(f"Mode: {result.mode}")             # "major"
print(f"Confidence: {result.confidence:.2f}")

# Camelot wheel for DJs
print(f"Camelot: {result.camelot}")       # "12B"

# Compatible keys for mixing
compatible = result.get_compatible_keys()
print(f"Compatible keys: {', '.join(compatible)}")

# Key change detection
if result.key_changes:
    print("Key changes detected:")
    for timestamp, key in result.key_changes:
        print(f"  {timestamp:.1f}s: {key}")
```

### Loudness Measurement

```python
from soundlab.analysis import measure_loudness

# Measure LUFS (ITU-R BS.1770-4)
result = measure_loudness("song.mp3")

print(f"Integrated LUFS: {result.lufs:.1f}")
print(f"Loudness range: {result.loudness_range:.1f} LU")
print(f"Peak level: {result.peak_db:.1f} dBFS")
print(f"True peak: {result.true_peak_db:.1f} dBTP")
print(f"Dynamic range: {result.dynamic_range:.1f} dB")

# Check mastering targets
streaming_target = -14.0  # Spotify, Apple Music target
if result.lufs > streaming_target:
    print(f"⚠️  Too loud for streaming ({result.lufs:.1f} LUFS)")
elif result.lufs < streaming_target - 2:
    print(f"ℹ️  Could be louder ({result.lufs:.1f} LUFS)")
else:
    print(f"✓ Good for streaming ({result.lufs:.1f} LUFS)")
```

### Spectral Analysis

```python
from soundlab.analysis import AudioAnalyzer

analyzer = AudioAnalyzer()
result = analyzer.analyze_spectral("audio.wav")

# Frequency content
print(f"Spectral centroid: {result.centroid_mean:.1f} Hz")
print(f"Bandwidth: {result.bandwidth_mean:.1f} Hz")
print(f"Rolloff: {result.rolloff_mean:.1f} Hz")

# Perceptual features
print(f"Brightness: {result.brightness:.2f}")  # 0-1 scale
print(f"Flatness: {result.flatness:.2f}")      # Noise vs tonal

# Time-varying features
for i, (centroid, time) in enumerate(zip(result.centroid, result.times)):
    print(f"Frame {i}: {centroid:.1f} Hz at {time:.2f}s")
```

### Onset Detection

```python
from soundlab.analysis import AudioAnalyzer

analyzer = AudioAnalyzer()
result = analyzer.detect_onsets("drums.wav")

# Get onset times
print(f"Total onsets: {len(result.timestamps)}")
print(f"Average strength: {result.strength_mean:.2f}")

# Access individual onsets
for onset_time, strength in zip(result.timestamps, result.strengths):
    print(f"Onset at {onset_time:.3f}s (strength: {strength:.2f})")

# Calculate inter-onset intervals (IOIs)
if len(result.timestamps) > 1:
    intervals = [result.timestamps[i+1] - result.timestamps[i]
                 for i in range(len(result.timestamps)-1)]
    avg_interval = sum(intervals) / len(intervals)
    print(f"Average IOI: {avg_interval:.3f}s")
    print(f"Estimated tempo from IOIs: {60/avg_interval:.1f} BPM")
```

### Batch Analysis

```python
from soundlab.analysis import analyze_audio
from pathlib import Path
import pandas as pd

# Analyze multiple files
results = []
audio_files = Path("audio_library/").glob("*.mp3")

for audio_file in audio_files:
    analysis = analyze_audio(str(audio_file))
    results.append({
        'filename': audio_file.name,
        'bpm': analysis.tempo.bpm,
        'key': analysis.key.name,
        'camelot': analysis.key.camelot,
        'lufs': analysis.loudness.lufs,
        'brightness': analysis.spectral.brightness
    })

# Create DataFrame
df = pd.DataFrame(results)
print(df)

# Export to CSV
df.to_csv("audio_analysis.csv", index=False)
```

### Custom Analysis Pipeline

```python
from soundlab.analysis import AudioAnalyzer
from soundlab.io import load_audio

# Load audio once
audio = load_audio("song.mp3")

# Create analyzer
analyzer = AudioAnalyzer()

# Run specific analyses
tempo = analyzer.detect_tempo(audio)
key = analyzer.detect_key(audio)
loudness = analyzer.measure_loudness(audio)

print(f"{tempo.bpm:.1f} BPM in {key.name} at {loudness.lufs:.1f} LUFS")
```

## Analysis Parameters

### Tempo Detection
- **hop_length**: Time resolution for onset detection (default: 512)
- **start_bpm**: Initial tempo guess (default: 120)

### Key Detection
- **window_size**: Analysis window in seconds (default: 4096 samples)
- **correlation_method**: 'krumhansl' or 'temperley' (default: 'krumhansl')

### Loudness
- **block_size**: Integration block size in ms (default: 400)
- **overlap**: Block overlap percentage (default: 75%)

### Spectral
- **n_fft**: FFT size (default: 2048)
- **hop_length**: Hop size in samples (default: 512)

## Reference Values

### Loudness Targets (LUFS)
- Spotify: -14 LUFS
- Apple Music: -16 LUFS
- YouTube: -13 to -15 LUFS
- Broadcast: -23 LUFS (EBU R128)
- Cinema: -24 LUFS (SMPTE ST 2067-3)

### Dynamic Range
- Highly compressed: < 6 dB
- Modern mastering: 6-10 dB
- Good dynamics: 10-15 dB
- Excellent dynamics: > 15 dB

### Camelot Wheel
```
Inner (Minor)    Outer (Major)
8A = A♭ minor    8B = B major
9A = E♭ minor    9B = F# major
10A = B♭ minor   10B = D♭ major
11A = F minor    11B = A♭ major
12A = C minor    12B = E♭ major
1A = G minor     1B = B♭ major
2A = D minor     2B = F major
3A = A minor     3B = C major
4A = E minor     4B = G major
5A = B minor     5B = D major
6A = F# minor    6B = A major
7A = C# minor    7B = E major
```

## Performance Tips

- **Batch Processing**: Analyze multiple features on the same audio load
- **Sample Rate**: Lower sample rates (22050 Hz) are sufficient for most analyses
- **File Format**: Use WAV or FLAC for best analysis accuracy
- **Duration**: Longer samples give more accurate key and tempo detection

## API Reference

::: soundlab.analysis
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
