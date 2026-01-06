# Effects

Professional audio effects processing using Pedalboard.

## Overview

The `soundlab.effects` module provides a comprehensive suite of professional audio effects processors built on Spotify's Pedalboard library. Create effects chains for mixing, mastering, and creative sound design.

## Features

- **Chainable Effects**: Fluent API for building complex effects chains
- **Professional Quality**: Studio-grade DSP algorithms from Pedalboard
- **Type-Safe Configuration**: Pydantic models for all effect parameters
- **Real-Time Capable**: Low-latency processing suitable for live use
- **Extensive Effects**: Dynamics, EQ, time-based, and creative effects

## Effect Categories

### Dynamics
Compressor, Limiter, Gate, Expander

### Equalization (EQ)
High-pass, Low-pass, Band-pass, Peak, Notch, Shelving

### Time-Based
Reverb, Delay, Chorus, Flanger, Phaser

### Creative
Distortion, Bitcrusher, Pitch Shift, Harmonizer

## Key Components

### EffectsChain
The main interface for building and processing effects chains. Supports fluent API for adding effects sequentially.

### Effect Configurations
Type-safe configuration classes for each effect:
- **Dynamics**: `CompressorConfig`, `LimiterConfig`, `GateConfig`
- **EQ**: `HighPassFilterConfig`, `LowPassFilterConfig`, `PeakFilterConfig`
- **Time-Based**: `ReverbConfig`, `DelayConfig`, `ChorusConfig`
- **Creative**: `DistortionConfig`, `PhaserConfig`, `BitcrusherConfig`

## Usage Examples

### Basic Effects Chain

```python
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig
from soundlab.effects.eq import HighPassFilterConfig

# Build effects chain
chain = (
    EffectsChain()
    .add(HighPassFilterConfig(cutoff_frequency_hz=80))
    .add(CompressorConfig(threshold_db=-20, ratio=4.0))
)

# Process audio
output_path = chain.process(
    input_path="vocals.wav",
    output_path="processed_vocals.wav"
)

print(f"Processed: {output_path}")
print(f"Effects applied: {len(chain.effects)}")
```

### Vocal Processing Chain

```python
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig, LimiterConfig
from soundlab.effects.eq import HighPassFilterConfig, PeakFilterConfig
from soundlab.effects.time_based import ReverbConfig, DelayConfig

# Professional vocal chain
chain = (
    EffectsChain()
    # Clean up low end
    .add(HighPassFilterConfig(cutoff_frequency_hz=100))
    # Add presence
    .add(PeakFilterConfig(
        center_frequency_hz=3000,
        q=1.0,
        gain_db=3.0
    ))
    # Control dynamics
    .add(CompressorConfig(
        threshold_db=-18,
        ratio=3.0,
        attack_ms=5.0,
        release_ms=100.0,
        knee_db=2.0
    ))
    # Add space
    .add(ReverbConfig(
        room_size=0.4,
        damping=0.5,
        wet_level=0.25,
        dry_level=0.75,
        width=1.0
    ))
    # Add depth
    .add(DelayConfig(
        delay_seconds=0.25,
        feedback=0.3,
        mix=0.15
    ))
    # Final limiting
    .add(LimiterConfig(
        threshold_db=-1.0,
        release_ms=50.0
    ))
)

# Process
chain.process("dry_vocal.wav", "mixed_vocal.wav")
```

### Mastering Chain

```python
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig, LimiterConfig
from soundlab.effects.eq import HighPassFilterConfig, PeakFilterConfig

# Mastering chain
chain = (
    EffectsChain()
    # Remove subsonic frequencies
    .add(HighPassFilterConfig(cutoff_frequency_hz=30))
    # Gentle compression for glue
    .add(CompressorConfig(
        threshold_db=-15,
        ratio=1.5,
        attack_ms=30.0,
        release_ms=200.0,
        knee_db=5.0
    ))
    # Add air/brightness
    .add(PeakFilterConfig(
        center_frequency_hz=12000,
        q=0.7,
        gain_db=1.5
    ))
    # Final limiting
    .add(LimiterConfig(
        threshold_db=-0.3,
        release_ms=100.0
    ))
)

# Process with target loudness
chain.process("mix.wav", "master.wav")
```

### Guitar Effects Chain

```python
from soundlab.effects import EffectsChain
from soundlab.effects.creative import DistortionConfig, ChorusConfig
from soundlab.effects.time_based import DelayConfig, ReverbConfig

# Guitar pedal board
chain = (
    EffectsChain()
    # Overdrive
    .add(DistortionConfig(
        drive_db=20.0,
        output_gain_db=-10.0
    ))
    # Chorus for width
    .add(ChorusConfig(
        rate_hz=1.5,
        depth=0.5,
        centre_delay_ms=7.0,
        feedback=0.25,
        mix=0.4
    ))
    # Delay
    .add(DelayConfig(
        delay_seconds=0.375,  # Dotted 8th at 120 BPM
        feedback=0.4,
        mix=0.3
    ))
    # Ambient reverb
    .add(ReverbConfig(
        room_size=0.7,
        damping=0.4,
        wet_level=0.3,
        dry_level=0.7
    ))
)

chain.process("guitar_dry.wav", "guitar_wet.wav")
```

### Parallel Processing

```python
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig
from soundlab.io import load_audio, save_audio
import numpy as np

# Load audio
audio = load_audio("vocals.wav")

# Dry path (no processing)
dry = audio.samples

# Wet path (heavy compression)
wet_chain = EffectsChain().add(CompressorConfig(
    threshold_db=-30,
    ratio=10.0,
    attack_ms=1.0,
    release_ms=50.0
))
wet = wet_chain.process_samples(audio.samples, audio.sample_rate)

# Mix dry and wet (New York compression)
mix_ratio = 0.3  # 30% wet, 70% dry
mixed = (dry * (1 - mix_ratio)) + (wet * mix_ratio)

# Save result
save_audio("vocals_parallel.wav", mixed, audio.sample_rate)
```

### Dynamic Effect Parameters

```python
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig

# Create multiple versions with different settings
settings = [
    ('gentle', CompressorConfig(threshold_db=-20, ratio=2.0)),
    ('medium', CompressorConfig(threshold_db=-18, ratio=4.0)),
    ('heavy', CompressorConfig(threshold_db=-15, ratio=8.0))
]

for name, config in settings:
    chain = EffectsChain().add(config)
    output = f"vocal_{name}.wav"
    chain.process("vocal.wav", output)
    print(f"Created {output}")
```

### Effect Presets

```python
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig
from soundlab.effects.eq import HighPassFilterConfig, PeakFilterConfig
from soundlab.effects.time_based import ReverbConfig

def create_vocal_preset():
    """Professional vocal preset."""
    return (
        EffectsChain()
        .add(HighPassFilterConfig(cutoff_frequency_hz=100))
        .add(PeakFilterConfig(center_frequency_hz=3000, q=1.0, gain_db=3.0))
        .add(CompressorConfig(threshold_db=-18, ratio=3.0))
        .add(ReverbConfig(room_size=0.4, wet_level=0.2))
    )

def create_podcast_preset():
    """Podcast/voiceover preset."""
    return (
        EffectsChain()
        .add(HighPassFilterConfig(cutoff_frequency_hz=80))
        .add(CompressorConfig(threshold_db=-20, ratio=4.0, attack_ms=5.0))
        .add(PeakFilterConfig(center_frequency_hz=5000, q=0.7, gain_db=2.0))
    )

# Use presets
vocal_chain = create_vocal_preset()
vocal_chain.process("vocals.wav", "vocals_processed.wav")

podcast_chain = create_podcast_preset()
podcast_chain.process("podcast.wav", "podcast_processed.wav")
```

### Batch Processing

```python
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig
from pathlib import Path

# Create chain once
chain = EffectsChain().add(CompressorConfig(threshold_db=-20, ratio=4.0))

# Process multiple files
input_dir = Path("raw_vocals/")
output_dir = Path("processed_vocals/")
output_dir.mkdir(exist_ok=True)

for audio_file in input_dir.glob("*.wav"):
    output_path = output_dir / audio_file.name
    chain.process(str(audio_file), str(output_path))
    print(f"Processed {audio_file.name}")
```

## Effect Reference

### Compressor
Controls dynamic range by reducing loud signals above a threshold.

**Parameters:**
- `threshold_db`: Level above which compression starts (-60 to 0 dB)
- `ratio`: Compression ratio (1:1 to 20:1)
- `attack_ms`: Attack time (0.1 to 100 ms)
- `release_ms`: Release time (10 to 1000 ms)
- `knee_db`: Soft knee (0 to 12 dB)

**Use Cases:** Vocals, drums, mastering

### Limiter
Prevents audio from exceeding a maximum level.

**Parameters:**
- `threshold_db`: Maximum output level (-20 to 0 dB)
- `release_ms`: Release time (10 to 500 ms)

**Use Cases:** Mastering, broadcast safety

### Gate
Attenuates signals below a threshold (removes noise).

**Parameters:**
- `threshold_db`: Gate opens above this level (-80 to 0 dB)
- `ratio`: Expansion ratio (1:1 to 10:1)
- `attack_ms`: Attack time (0.1 to 100 ms)
- `release_ms`: Release time (10 to 1000 ms)

**Use Cases:** Noise reduction, drum gating

### High-Pass Filter
Removes frequencies below cutoff (bass reduction).

**Parameters:**
- `cutoff_frequency_hz`: Cutoff frequency (20 to 20000 Hz)

**Use Cases:** Remove rumble, clean up vocals

### Reverb
Simulates acoustic space and ambience.

**Parameters:**
- `room_size`: Size of the simulated space (0 to 1)
- `damping`: High-frequency absorption (0 to 1)
- `wet_level`: Effect signal level (0 to 1)
- `dry_level`: Original signal level (0 to 1)
- `width`: Stereo width (0 to 1)

**Use Cases:** Adding space, depth, ambience

### Delay
Time-based echo effect.

**Parameters:**
- `delay_seconds`: Delay time (0.001 to 2.0 seconds)
- `feedback`: Number of repeats (0 to 0.95)
- `mix`: Wet/dry mix (0 to 1)

**Use Cases:** Slapback, rhythmic delays, doubling

### Distortion
Adds harmonic saturation and overdrive.

**Parameters:**
- `drive_db`: Amount of distortion (0 to 40 dB)
- `output_gain_db`: Output level adjustment (-40 to 0 dB)

**Use Cases:** Guitar, bass, creative processing

## Performance Tips

- **Chain Order Matters**: Apply EQ before compression for cleaner results
- **CPU Usage**: Complex effects chains may require more CPU
- **Batch Processing**: Reuse chains for multiple files
- **Preset Management**: Create and save effect chain templates

## Common Effect Orders

### Mixing
1. High-pass filter (cleanup)
2. Subtractive EQ (problem frequencies)
3. Compression (dynamics)
4. Additive EQ (enhancement)
5. Time-based effects (reverb, delay)

### Mastering
1. High-pass filter (subsonic removal)
2. Gentle compression (glue)
3. EQ (final tonal balance)
4. Limiting (loudness)

### Creative
1. Distortion/saturation
2. Modulation (chorus, flanger, phaser)
3. Delay
4. Reverb

## API Reference

::: soundlab.effects
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
