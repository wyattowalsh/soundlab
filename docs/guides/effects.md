# Effects Processing Guide

Learn how to apply professional audio effects including dynamics, EQ, reverb, delay, and creative processing using Spotify's Pedalboard.

## Overview

SoundLab provides a comprehensive effects processing system built on [Pedalboard](https://github.com/spotify/pedalboard), featuring:

- **Dynamics** - Compression, limiting, gating, gain
- **EQ** - Filters, shelves, parametric EQ
- **Time-Based** - Reverb, delay, chorus, phaser
- **Creative** - Distortion, clipping
- **Chainable** - Fluent API for building effect chains

### Use Cases

- **Vocal Processing** - Polish vocals with compression, EQ, and reverb
- **Mixing** - Balance and enhance individual tracks
- **Mastering** - Final polish with multi-band processing
- **Creative Effects** - Add character with distortion and modulation
- **Audio Repair** - Clean up recordings with gates and filters
- **Sound Design** - Create unique textures

## Quick Start

Basic effects chain:

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

## Dynamics Effects

Control the dynamic range of audio.

### Compressor

Reduce dynamic range by attenuating loud signals:

```python
from soundlab.effects import EffectsChain
from soundlab.effects.dynamics import CompressorConfig

# Gentle compression for vocals
vocal_comp = CompressorConfig(
    threshold_db=-20,      # Start compressing above -20 dB
    ratio=3.0,             # 3:1 compression ratio
    attack_ms=10.0,        # 10ms attack (fast)
    release_ms=100.0       # 100ms release
)

# Aggressive compression for drums
drum_comp = CompressorConfig(
    threshold_db=-15,      # Higher threshold
    ratio=8.0,             # Heavy compression
    attack_ms=1.0,         # Very fast attack
    release_ms=50.0        # Fast release
)

# Apply compression
chain = EffectsChain().add(vocal_comp)
chain.process("vocals.wav", "vocals_compressed.wav")
```

**Parameters**:

- **threshold_db** (-60 to 0): Level where compression starts
- **ratio** (1 to 20): Compression amount (4:1 = 4dB in → 1dB out)
- **attack_ms** (0.1 to 100): How fast compression engages
- **release_ms** (10 to 1000): How fast compression releases

**Use cases**:

- Vocals: threshold -20dB, ratio 3-4:1, attack 5-10ms, release 100ms
- Drums: threshold -15dB, ratio 6-8:1, attack 1ms, release 50ms
- Bass: threshold -18dB, ratio 4:1, attack 10ms, release 100ms
- Mix bus: threshold -10dB, ratio 2:1, attack 30ms, release 100ms

### Limiter

Prevent peaks from exceeding threshold:

```python
from soundlab.effects.dynamics import LimiterConfig

# Mastering limiter (prevent clipping)
limiter = LimiterConfig(
    threshold_db=-1.0,     # Maximum peak level
    release_ms=100.0       # Release time
)

# Broadcast limiter (strict compliance)
broadcast_limiter = LimiterConfig(
    threshold_db=-3.0,     # Conservative threshold
    release_ms=200.0       # Slower release
)

chain = EffectsChain().add(limiter)
chain.process("mix.wav", "limited.wav")
```

**Parameters**:

- **threshold_db** (-60 to 0): Maximum allowed peak
- **release_ms** (10 to 1000): Recovery time

**Use cases**:

- Mastering: threshold -1dB, release 100ms
- Broadcast: threshold -3dB, release 200ms
- Streaming: threshold -1dB, release 50ms

### Noise Gate

Remove low-level noise and bleed:

```python
from soundlab.effects.dynamics import GateConfig

# Vocal gate (remove room noise)
vocal_gate = GateConfig(
    threshold_db=-40,      # Close gate below -40dB
    ratio=10.0,            # Heavy gating
    attack_ms=1.0,         # Fast opening
    release_ms=100.0       # Smooth closing
)

# Drum gate (isolate hits)
drum_gate = GateConfig(
    threshold_db=-35,      # Higher threshold
    ratio=20.0,            # Very aggressive
    attack_ms=0.5,         # Instant opening
    release_ms=50.0        # Fast closing
)

chain = EffectsChain().add(vocal_gate)
chain.process("vocals.wav", "gated_vocals.wav")
```

**Parameters**:

- **threshold_db** (-80 to 0): Level below which gate closes
- **ratio** (1 to 20): How much to attenuate
- **attack_ms** (0.1 to 100): Opening speed
- **release_ms** (10 to 1000): Closing speed

### Gain

Simple volume adjustment:

```python
from soundlab.effects.dynamics import GainConfig

# Boost by 6dB
gain_up = GainConfig(gain_db=6.0)

# Reduce by 3dB
gain_down = GainConfig(gain_db=-3.0)

chain = EffectsChain().add(gain_up)
chain.process("quiet.wav", "louder.wav")
```

## EQ Effects

Shape frequency content.

### High-Pass Filter

Remove low frequencies:

```python
from soundlab.effects.eq import HighPassFilterConfig

# Remove rumble from vocals
vocal_hpf = HighPassFilterConfig(cutoff_frequency_hz=80)

# Remove sub-bass from guitars
guitar_hpf = HighPassFilterConfig(cutoff_frequency_hz=120)

# Aggressive high-pass
aggressive_hpf = HighPassFilterConfig(cutoff_frequency_hz=200)

chain = EffectsChain().add(vocal_hpf)
chain.process("vocals.wav", "filtered_vocals.wav")
```

**Common cutoff frequencies**:

- **40-80 Hz**: Vocals, guitars, most instruments
- **80-120 Hz**: Thin instruments, reduce mud
- **120-200 Hz**: Aggressive filtering, special effects

### Low-Pass Filter

Remove high frequencies:

```python
from soundlab.effects.eq import LowPassFilterConfig

# Warm up digital recordings
warmth_lpf = LowPassFilterConfig(cutoff_frequency_hz=15000)

# Create "phone" effect
phone_lpf = LowPassFilterConfig(cutoff_frequency_hz=3000)

# Extreme low-pass
bass_lpf = LowPassFilterConfig(cutoff_frequency_hz=500)

chain = EffectsChain().add(warmth_lpf)
chain.process("harsh.wav", "warm.wav")
```

### High Shelf

Boost or cut high frequencies:

```python
from soundlab.effects.eq import HighShelfConfig

# Add air/sparkle
air_boost = HighShelfConfig(
    cutoff_frequency_hz=8000,
    gain_db=3.0,           # Boost by 3dB
    q=0.707                # Smooth curve
)

# Reduce harshness
de_harsh = HighShelfConfig(
    cutoff_frequency_hz=6000,
    gain_db=-2.0,          # Cut by 2dB
    q=1.0
)

chain = EffectsChain().add(air_boost)
chain.process("dull.wav", "bright.wav")
```

**Parameters**:

- **cutoff_frequency_hz** (20-20000): Shelf frequency
- **gain_db** (-24 to 24): Boost or cut amount
- **q** (0.1-10): Curve steepness (0.707 is natural)

### Low Shelf

Boost or cut low frequencies:

```python
from soundlab.effects.eq import LowShelfConfig

# Add warmth/body
warmth_boost = LowShelfConfig(
    cutoff_frequency_hz=200,
    gain_db=3.0,
    q=0.707
)

# Reduce muddiness
mud_cut = LowShelfConfig(
    cutoff_frequency_hz=300,
    gain_db=-4.0,
    q=1.0
)

chain = EffectsChain().add(warmth_boost)
chain.process("thin.wav", "full.wav")
```

### Parametric EQ (Peak Filter)

Precise frequency control:

```python
from soundlab.effects.eq import PeakFilterConfig

# Boost presence (vocals)
presence_boost = PeakFilterConfig(
    cutoff_frequency_hz=3000,
    gain_db=3.0,
    q=2.0                  # Narrow boost
)

# Cut boxiness
box_cut = PeakFilterConfig(
    cutoff_frequency_hz=400,
    gain_db=-4.0,
    q=2.0
)

# Surgical notch (remove resonance)
notch = PeakFilterConfig(
    cutoff_frequency_hz=1000,
    gain_db=-12.0,
    q=10.0                 # Very narrow
)

chain = EffectsChain().add(presence_boost)
chain.process("vocals.wav", "present_vocals.wav")
```

**Common EQ targets**:

| Frequency | Description | Typical Adjustment |
|-----------|-------------|-------------------|
| 60-100 Hz | Sub-bass/rumble | Cut 3-6 dB |
| 200-400 Hz | Warmth/mud | Cut 2-4 dB if muddy |
| 800-1200 Hz | Boxiness | Cut 2-4 dB if boxy |
| 2-4 kHz | Presence/clarity | Boost 2-4 dB |
| 6-8 kHz | Brightness/air | Boost 2-3 dB |
| 10+ kHz | Sparkle | Boost 1-2 dB |

## Time-Based Effects

Add space and depth.

### Reverb

Create acoustic space:

```python
from soundlab.effects.time_based import ReverbConfig

# Subtle room reverb
room_reverb = ReverbConfig(
    room_size=0.3,         # Small room
    damping=0.5,           # Medium damping
    wet_level=0.2,         # 20% reverb
    dry_level=0.8,         # 80% dry
    width=1.0              # Full stereo width
)

# Large hall
hall_reverb = ReverbConfig(
    room_size=0.8,         # Large space
    damping=0.3,           # Less damping
    wet_level=0.4,         # More reverb
    dry_level=0.6,
    width=1.0
)

# Plate reverb (bright)
plate_reverb = ReverbConfig(
    room_size=0.5,
    damping=0.2,           # Bright character
    wet_level=0.3,
    dry_level=0.7,
    width=0.8
)

chain = EffectsChain().add(room_reverb)
chain.process("dry_vocals.wav", "reverb_vocals.wav")
```

**Parameters**:

- **room_size** (0-1): Space size
- **damping** (0-1): High-frequency absorption
- **wet_level** (0-1): Reverb amount
- **dry_level** (0-1): Original signal amount
- **width** (0-1): Stereo width

**Common settings**:

- Vocals: room_size 0.3-0.5, wet 0.2-0.3, damping 0.5
- Drums: room_size 0.2-0.4, wet 0.1-0.2, damping 0.6
- Guitars: room_size 0.4-0.6, wet 0.3-0.4, damping 0.4
- Snare: room_size 0.2, wet 0.3, damping 0.7 (gated reverb effect)

### Delay

Create echoes:

```python
from soundlab.effects.time_based import DelayConfig

# Eighth note delay (at 120 BPM: 60/120 * 0.5 = 0.25s)
eighth_delay = DelayConfig(
    delay_seconds=0.25,
    feedback=0.3,          # 3 repeats
    mix=0.3                # 30% delayed signal
)

# Slap delay (short)
slap_delay = DelayConfig(
    delay_seconds=0.08,    # 80ms
    feedback=0.1,          # Single repeat
    mix=0.2
)

# Long delay (ambient)
long_delay = DelayConfig(
    delay_seconds=0.75,
    feedback=0.5,          # Many repeats
    mix=0.4
)

chain = EffectsChain().add(eighth_delay)
chain.process("guitar.wav", "delayed_guitar.wav")
```

**Tempo-synced delays (BPM to seconds)**:

```python
def bpm_to_delay(bpm, note_division):
    """
    Convert BPM and note division to delay time.

    note_division: 1.0 = whole note, 0.5 = half note,
                   0.25 = quarter note, 0.125 = eighth note, etc.
    """
    beat_duration = 60.0 / bpm
    return beat_duration * note_division

# At 120 BPM
bpm = 120
quarter_note = DelayConfig(delay_seconds=bpm_to_delay(bpm, 0.25))  # 0.5s
eighth_note = DelayConfig(delay_seconds=bpm_to_delay(bpm, 0.125))  # 0.25s
dotted_eighth = DelayConfig(delay_seconds=bpm_to_delay(bpm, 0.1875))  # 0.375s
```

### Chorus

Create thickness and movement:

```python
from soundlab.effects.time_based import ChorusConfig

# Subtle chorus
subtle_chorus = ChorusConfig(
    rate_hz=1.0,           # Slow modulation
    depth=0.2,             # Subtle effect
    centre_delay_ms=7.0,
    feedback=0.0,
    mix=0.3
)

# Pronounced chorus
thick_chorus = ChorusConfig(
    rate_hz=2.0,           # Faster modulation
    depth=0.5,             # More obvious
    centre_delay_ms=10.0,
    feedback=0.2,
    mix=0.5
)

chain = EffectsChain().add(subtle_chorus)
chain.process("guitar.wav", "chorus_guitar.wav")
```

### Phaser

Sweeping filter effect:

```python
from soundlab.effects.time_based import PhaserConfig

# Classic phaser
classic_phaser = PhaserConfig(
    rate_hz=0.5,           # Slow sweep
    depth=0.5,
    centre_frequency_hz=1300,
    feedback=0.5,
    mix=0.5
)

# Intense phaser
intense_phaser = PhaserConfig(
    rate_hz=2.0,           # Fast sweep
    depth=0.8,             # Deep modulation
    centre_frequency_hz=1000,
    feedback=0.7,          # More pronounced
    mix=0.7
)

chain = EffectsChain().add(classic_phaser)
chain.process("synth.wav", "phased_synth.wav")
```

## Creative Effects

Add character and color.

### Distortion

Add harmonic saturation:

```python
from soundlab.effects.creative import DistortionConfig

# Subtle warmth
warmth = DistortionConfig(drive_db=10.0)

# Moderate overdrive
overdrive = DistortionConfig(drive_db=25.0)

# Heavy distortion
heavy = DistortionConfig(drive_db=40.0)

chain = EffectsChain().add(overdrive)
chain.process("guitar.wav", "overdriven_guitar.wav")
```

### Clipping

Hard limiting with character:

```python
from soundlab.effects.creative import ClippingConfig

# Soft clipping (saturation)
soft_clip = ClippingConfig(threshold_db=-6.0)

# Hard clipping
hard_clip = ClippingConfig(threshold_db=-3.0)

chain = EffectsChain().add(soft_clip)
chain.process("drums.wav", "clipped_drums.wav")
```

## Effect Chains

Combine multiple effects.

### Vocal Chain

Professional vocal processing:

```python
from soundlab.effects import EffectsChain
from soundlab.effects.eq import HighPassFilterConfig, PeakFilterConfig
from soundlab.effects.dynamics import CompressorConfig, LimiterConfig
from soundlab.effects.time_based import ReverbConfig, DelayConfig

vocal_chain = (
    EffectsChain()
    # 1. Clean up low end
    .add(HighPassFilterConfig(cutoff_frequency_hz=80))
    # 2. Reduce boxiness
    .add(PeakFilterConfig(cutoff_frequency_hz=400, gain_db=-3.0, q=2.0))
    # 3. Add presence
    .add(PeakFilterConfig(cutoff_frequency_hz=3000, gain_db=2.5, q=1.5))
    # 4. Compress
    .add(CompressorConfig(
        threshold_db=-20,
        ratio=4.0,
        attack_ms=5.0,
        release_ms=100.0
    ))
    # 5. Add subtle delay
    .add(DelayConfig(
        delay_seconds=0.125,  # Eighth note at 120 BPM
        feedback=0.2,
        mix=0.15
    ))
    # 6. Add reverb
    .add(ReverbConfig(
        room_size=0.4,
        damping=0.5,
        wet_level=0.25,
        dry_level=0.75
    ))
    # 7. Final limiter
    .add(LimiterConfig(threshold_db=-2.0, release_ms=100.0))
)

vocal_chain.process("vocals_dry.wav", "vocals_final.wav")
print(f"Applied {len(vocal_chain.effects)} effects")
```

### Guitar Chain

Rock guitar processing:

```python
guitar_chain = (
    EffectsChain()
    # 1. High-pass filter
    .add(HighPassFilterConfig(cutoff_frequency_hz=120))
    # 2. Overdrive
    .add(DistortionConfig(drive_db=20.0))
    # 3. EQ shaping
    .add(PeakFilterConfig(cutoff_frequency_hz=2500, gain_db=3.0, q=1.0))
    # 4. Compression
    .add(CompressorConfig(
        threshold_db=-15,
        ratio=3.0,
        attack_ms=10.0,
        release_ms=80.0
    ))
    # 5. Chorus for width
    .add(ChorusConfig(rate_hz=1.5, depth=0.3, mix=0.3))
    # 6. Delay
    .add(DelayConfig(delay_seconds=0.375, feedback=0.4, mix=0.3))
    # 7. Reverb
    .add(ReverbConfig(room_size=0.5, wet_level=0.2, dry_level=0.8))
)

guitar_chain.process("guitar_di.wav", "guitar_processed.wav")
```

### Mastering Chain

Final polish:

```python
mastering_chain = (
    EffectsChain()
    # 1. Subtle high-pass
    .add(HighPassFilterConfig(cutoff_frequency_hz=30))
    # 2. Low-end control
    .add(PeakFilterConfig(cutoff_frequency_hz=60, gain_db=-1.0, q=1.0))
    # 3. Gentle compression
    .add(CompressorConfig(
        threshold_db=-10,
        ratio=2.0,
        attack_ms=30.0,
        release_ms=100.0
    ))
    # 4. Brightness
    .add(HighShelfConfig(cutoff_frequency_hz=8000, gain_db=1.5))
    # 5. Final limiter
    .add(LimiterConfig(threshold_db=-0.5, release_ms=100.0))
)

mastering_chain.process("mix.wav", "master.wav")
```

### Drum Bus Chain

Glue drums together:

```python
drum_bus_chain = (
    EffectsChain()
    # 1. Remove rumble
    .add(HighPassFilterConfig(cutoff_frequency_hz=40))
    # 2. Parallel compression effect
    .add(CompressorConfig(
        threshold_db=-20,
        ratio=6.0,
        attack_ms=1.0,
        release_ms=50.0
    ))
    # 3. Add punch
    .add(PeakFilterConfig(cutoff_frequency_hz=100, gain_db=2.0, q=1.0))
    # 4. Add clarity
    .add(PeakFilterConfig(cutoff_frequency_hz=5000, gain_db=1.5, q=1.5))
    # 5. Saturation
    .add(DistortionConfig(drive_db=5.0))
)

drum_bus_chain.process("drums_mix.wav", "drums_glued.wav")
```

## Chain Management

### Building Chains

```python
from soundlab.effects import EffectsChain

# Method 1: Fluent API (recommended)
chain = (
    EffectsChain()
    .add(effect1)
    .add(effect2)
    .add(effect3)
)

# Method 2: From list
effects = [effect1, effect2, effect3]
chain = EffectsChain.from_configs(effects)

# Method 3: Incremental
chain = EffectsChain()
chain.add(effect1)
chain.add(effect2)
```

### Modifying Chains

```python
chain = (
    EffectsChain()
    .add(effect1)
    .add(effect2)
    .add(effect3)
)

# Insert at position
chain.insert(1, effect_new)  # Insert after effect1

# Remove effect
chain.remove(2)  # Remove third effect

# Clear all
chain.clear()

# Check contents
print(f"Effects: {len(chain)}")
print(f"Names: {chain.effect_names}")
print(f"Chain: {chain}")
```

### Preview Effects

Test on short segment:

```python
# Preview first 10 seconds
preview = chain.preview(
    input_path="full_song.wav",
    duration=10.0,
    start=0.0
)

# Play or analyze preview
from soundlab.io import save_audio
save_audio("preview.wav", preview, 44100)
```

### Process Arrays

Work with audio data directly:

```python
from soundlab.io import load_audio, save_audio

# Load audio
audio, sr = load_audio("input.wav")

# Process
processed = chain.process_array(audio, sr)

# Save
save_audio("output.wav", processed, sr)
```

## Best Practices

### Effect Order

Standard effect chain order:

1. **Gain staging** - Initial level adjustment
2. **Gates** - Remove noise before other processing
3. **EQ (subtractive)** - Remove problem frequencies
4. **Compression** - Control dynamics
5. **EQ (additive)** - Enhance desired frequencies
6. **Saturation/Distortion** - Add character
7. **Modulation** (Chorus, Phaser) - Add movement
8. **Time-based** (Delay, Reverb) - Add space
9. **Limiting** - Final level control

### A/B Testing

Compare processed vs. original:

```python
from soundlab.io import load_audio
import numpy as np

# Load original
original, sr = load_audio("input.wav")

# Process
chain = EffectsChain().add(effect)
processed = chain.process_array(original, sr)

# Calculate RMS difference
rms_orig = np.sqrt(np.mean(original ** 2))
rms_proc = np.sqrt(np.mean(processed ** 2))
print(f"Level change: {20 * np.log10(rms_proc / rms_orig):.1f} dB")

# Save both for comparison
save_audio("original.wav", original, sr)
save_audio("processed.wav", processed, sr)
```

### Gain Staging

Maintain proper levels throughout chain:

```python
from soundlab.effects.dynamics import GainConfig

# Compensate for heavy compression
chain = (
    EffectsChain()
    .add(CompressorConfig(threshold_db=-20, ratio=6.0))
    .add(GainConfig(gain_db=6.0))  # Make up gain
)
```

### Preset Management

Save and reuse configurations:

```python
import json
from pathlib import Path

# Save chain config
def save_chain_preset(chain, preset_name):
    configs = []
    for effect in chain.effects:
        configs.append({
            "type": effect.__class__.__name__,
            "params": effect.model_dump()
        })

    with open(f"{preset_name}.json", "w") as f:
        json.dump(configs, f, indent=2)

# Load preset
def load_chain_preset(preset_name):
    with open(f"{preset_name}.json") as f:
        configs = json.load(f)

    # Reconstruct chain (requires mapping class names to classes)
    # Implementation depends on your needs
    return chain

# Save current chain
save_chain_preset(vocal_chain, "vocal_preset")
```

## Common Issues and Solutions

### Clipping

```python
# Check for clipping
processed = chain.process_array(audio, sr)
if np.max(np.abs(processed)) > 0.99:
    print("Warning: Clipping detected!")

    # Add gain reduction
    chain_with_makeup = (
        chain
        .add(GainConfig(gain_db=-3.0))  # Reduce by 3dB
    )
```

### Phase Issues

```python
# Some effects (EQ, filters) can cause phase shifts
# Verify with null test
difference = original - processed
null_level = np.max(np.abs(difference))
if null_level < 0.001:
    print("Null test passed - no processing!")
```

## Performance

Processing times for 3-minute song:

| Effect Type | Time | Notes |
|------------|------|-------|
| Single effect | <1 sec | Real-time |
| 5-effect chain | ~2 sec | Very fast |
| 10-effect chain | ~5 sec | Still fast |
| Complex chain (20+) | ~10 sec | More processing |

## Next Steps

- **[Separation Guide](separation.md)** - Process separated stems
- **[Analysis Guide](analysis.md)** - Analyze before processing
- **[Quick Start](quickstart.md)** - More examples

---

**Questions?** Visit [GitHub Discussions](https://github.com/wyattwalsh/soundlab/discussions).
