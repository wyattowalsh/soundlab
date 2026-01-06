# Voice

Voice generation capabilities including text-to-speech and voice conversion.

## Overview

The `soundlab.voice` module provides advanced voice generation and conversion capabilities using state-of-the-art neural models. This is an optional feature that requires additional dependencies (`pip install soundlab[voice]`).

## Features

- **Text-to-Speech (TTS)**: Generate natural speech in 18+ languages using XTTS-v2
- **Voice Cloning**: Clone any voice from a 6-30 second audio sample
- **Singing Voice Conversion (SVC)**: Transform vocals using RVC models
- **Multi-Language Support**: Support for major languages and accents
- **GPU Acceleration**: Fast generation with automatic GPU utilization

## Key Components

### TextToSpeech
Neural text-to-speech with voice cloning capabilities using Coqui XTTS-v2.

### SingingVoiceConverter
RVC-based singing voice conversion for transforming vocal recordings.

### TTSConfig
Configuration for text-to-speech including language, speaker reference, and generation parameters.

### SVCConfig
Configuration for voice conversion including model selection and pitch shifting.

## Installation

Voice generation requires additional dependencies:

```bash
# Install with pip
pip install soundlab[voice]

# Or with uv
uv sync --extra voice
```

## Usage Examples

### Basic Text-to-Speech

```python
from soundlab.voice import TextToSpeech

# Create TTS instance
tts = TextToSpeech()

# Generate speech
result = tts.synthesize(
    text="Welcome to SoundLab, the production-ready music processing platform.",
    output_path="output/speech.wav"
)

print(f"Generated: {result.audio_path}")
print(f"Duration: {result.duration_seconds:.1f}s")
print(f"Sample rate: {result.sample_rate} Hz")
```

### Voice Cloning

```python
from soundlab.voice import TextToSpeech, TTSConfig

# Configure with voice sample
config = TTSConfig(
    language="en",
    speaker_wav="voice_sample.wav",  # 6-30 second reference audio
    temperature=0.7,                  # Expressiveness (0.1-1.0)
    speed=1.0                         # Speaking rate
)

# Create TTS with cloned voice
tts = TextToSpeech(config)

# Generate speech in the cloned voice
result = tts.synthesize(
    text="This is a demonstration of voice cloning technology.",
    output_path="output/cloned_speech.wav"
)

print(f"Cloned voice output: {result.audio_path}")
```

### Multi-Language TTS

```python
from soundlab.voice import TextToSpeech, TTSConfig

# Supported languages
languages = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'pl': 'Polish',
    'tr': 'Turkish',
    'ru': 'Russian',
    'nl': 'Dutch',
    'cs': 'Czech',
    'ar': 'Arabic',
    'zh-cn': 'Chinese (Simplified)',
    'ja': 'Japanese',
    'ko': 'Korean',
    'hi': 'Hindi'
}

# Generate in multiple languages
for lang_code, lang_name in languages.items():
    config = TTSConfig(language=lang_code)
    tts = TextToSpeech(config)

    result = tts.synthesize(
        text=f"Hello from {lang_name}!",
        output_path=f"output/hello_{lang_code}.wav"
    )
    print(f"Generated {lang_name}: {result.audio_path}")
```

### Long-Form TTS

```python
from soundlab.voice import TextToSpeech, TTSConfig

# Configure for long-form content
config = TTSConfig(
    language="en",
    speaker_wav="narrator_voice.wav",
    temperature=0.65,  # Slightly lower for consistency
    speed=0.95         # Slightly slower for clarity
)

tts = TextToSpeech(config)

# Long text (e.g., article or book chapter)
long_text = """
Chapter One: The Beginning

It was a dark and stormy night. The wind howled through the trees,
and rain lashed against the windows of the old mansion on the hill.
Inside, a lone figure sat by the fireplace, reading an ancient tome
that held secrets from a time long forgotten.
"""

# Generate
result = tts.synthesize(
    text=long_text,
    output_path="output/chapter_01.wav"
)

print(f"Narration: {result.audio_path}")
print(f"Duration: {result.duration_seconds / 60:.1f} minutes")
```

### Singing Voice Conversion

```python
from soundlab.voice import SingingVoiceConverter, SVCConfig

# Configure voice conversion
config = SVCConfig(
    model_path="models/singer_model.pth",
    pitch_shift=0,        # Semitones to shift (-12 to +12)
    index_rate=0.75,      # Feature retrieval strength (0-1)
    filter_radius=3,      # Median filtering for smoother output
    rms_mix_rate=0.25     # Volume envelope mixing (0-1)
)

# Create converter
converter = SingingVoiceConverter(config)

# Convert vocals
result = converter.convert(
    input_path="original_vocals.wav",
    output_path="converted_vocals.wav"
)

print(f"Converted: {result.audio_path}")
print(f"Processing time: {result.processing_time_seconds:.1f}s")
```

### Voice Conversion with Pitch Adjustment

```python
from soundlab.voice import SingingVoiceConverter, SVCConfig

# Adjust pitch for different vocal ranges
# Negative values: lower pitch (male → female)
# Positive values: higher pitch (female → male)

# Male to female conversion
male_to_female_config = SVCConfig(
    model_path="models/female_voice.pth",
    pitch_shift=-4,  # Lower by 4 semitones
    index_rate=0.8
)

# Female to male conversion
female_to_male_config = SVCConfig(
    model_path="models/male_voice.pth",
    pitch_shift=4,   # Raise by 4 semitones
    index_rate=0.8
)

# Use appropriate config
converter = SingingVoiceConverter(male_to_female_config)
result = converter.convert("male_vocals.wav", "female_vocals.wav")
```

### Batch Voice Cloning

```python
from soundlab.voice import TextToSpeech, TTSConfig
from pathlib import Path

# Multiple voice samples
voice_samples = {
    'narrator': 'voices/narrator.wav',
    'character_a': 'voices/character_a.wav',
    'character_b': 'voices/character_b.wav'
}

# Script with different speakers
script = [
    ('narrator', 'Once upon a time, in a land far away...'),
    ('character_a', 'Where are we going?'),
    ('character_b', 'To find the treasure, of course!'),
    ('narrator', 'And so their adventure began.')
]

# Generate each line with appropriate voice
output_dir = Path('output/dialogue')
output_dir.mkdir(exist_ok=True)

for i, (speaker, text) in enumerate(script):
    config = TTSConfig(
        language="en",
        speaker_wav=voice_samples[speaker]
    )
    tts = TextToSpeech(config)

    output_path = output_dir / f"{i:03d}_{speaker}.wav"
    result = tts.synthesize(text, str(output_path))
    print(f"Generated line {i+1}: {speaker}")
```

### Real-Time TTS Streaming

```python
from soundlab.voice import TextToSpeech, TTSConfig

# Configure for lower latency
config = TTSConfig(
    language="en",
    enable_streaming=True,  # Enable streaming mode
    chunk_size=512          # Smaller chunks for lower latency
)

tts = TextToSpeech(config)

# Generate with streaming callback
def on_audio_chunk(chunk, sample_rate):
    """Process audio chunks as they're generated."""
    # Play or process chunk in real-time
    print(f"Received chunk: {len(chunk)} samples")

result = tts.synthesize(
    text="This is a real-time streaming demonstration.",
    output_path="output/streaming.wav",
    streaming_callback=on_audio_chunk
)
```

### Voice Quality Control

```python
from soundlab.voice import TextToSpeech, TTSConfig

# High-quality settings (slower)
high_quality_config = TTSConfig(
    language="en",
    speaker_wav="voice.wav",
    temperature=0.65,     # Lower = more consistent
    repetition_penalty=5.0,  # Avoid repetitions
    length_penalty=1.0,   # Encourage natural length
    enable_text_splitting=True  # Better long-form handling
)

# Fast generation (lower quality)
fast_config = TTSConfig(
    language="en",
    speaker_wav="voice.wav",
    temperature=0.85,     # Higher = faster but less consistent
    enable_text_splitting=False
)

# Use appropriate config based on needs
tts_quality = TextToSpeech(high_quality_config)
tts_fast = TextToSpeech(fast_config)
```

## Configuration Parameters

### TTSConfig

**Language Settings:**
- `language`: Target language code (e.g., 'en', 'es', 'fr')
- `speaker_wav`: Path to voice sample for cloning (optional)

**Generation Parameters:**
- `temperature`: Expressiveness/randomness (0.1-1.0, default: 0.7)
- `speed`: Speaking rate (0.5-2.0, default: 1.0)
- `repetition_penalty`: Avoid repetitions (1.0-10.0, default: 5.0)
- `length_penalty`: Control output length (0.5-2.0, default: 1.0)

**Quality Settings:**
- `enable_text_splitting`: Better long-form handling (default: True)
- `enable_streaming`: Real-time generation (default: False)
- `chunk_size`: Streaming chunk size (default: 1024)

### SVCConfig

**Model Settings:**
- `model_path`: Path to RVC model file (.pth)
- `index_path`: Path to feature index file (optional)

**Processing Parameters:**
- `pitch_shift`: Pitch adjustment in semitones (-12 to +12)
- `index_rate`: Feature retrieval strength (0-1, default: 0.75)
- `filter_radius`: Median filtering radius (0-7, default: 3)
- `rms_mix_rate`: Volume envelope mixing (0-1, default: 0.25)

**Quality Settings:**
- `f0_method`: Pitch detection method ('crepe', 'harvest', 'dio')
- `crepe_hop_length`: CREPE hop length for pitch detection

## Supported Languages

### TTS (XTTS-v2)
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Polish (pl)
- Turkish (tr)
- Russian (ru)
- Dutch (nl)
- Czech (cs)
- Arabic (ar)
- Chinese/Simplified (zh-cn)
- Japanese (ja)
- Korean (ko)
- Hindi (hi)
- Hungarian (hu)
- Vietnamese (vi)

## Performance Tips

- **Voice Sample Quality**: Use clean, clear audio samples for best cloning results
- **Sample Length**: 6-30 seconds is optimal for voice cloning
- **GPU Acceleration**: Use GPU for 5-10x faster generation
- **Batch Processing**: Reuse TTS instances for multiple generations
- **Temperature**: Lower values (0.5-0.7) for consistency, higher (0.8-1.0) for variety

## Typical Generation Times

### TTS (XTTS-v2)

On NVIDIA T4 GPU:
- Short sentence (10 words): ~2 seconds
- Medium paragraph (50 words): ~5 seconds
- Long form (200 words): ~15 seconds

On CPU (Intel i7):
- Short sentence: ~10 seconds
- Medium paragraph: ~30 seconds
- Long form: ~2 minutes

### Voice Conversion (RVC)

On NVIDIA T4 GPU:
- 30-second vocal: ~5 seconds
- 3-minute song: ~30 seconds

On CPU:
- 30-second vocal: ~30 seconds
- 3-minute song: ~3 minutes

## Best Practices

### Voice Cloning
1. Use high-quality reference audio (no background noise)
2. Ensure reference audio is 6-30 seconds long
3. Use audio with emotional variety for better expressiveness
4. Match reference audio language to target language

### Voice Conversion
1. Use isolated vocal tracks (no instrumental bleeding)
2. Apply appropriate pitch shifting for vocal range matching
3. Experiment with `index_rate` for quality vs similarity balance
4. Use `filter_radius` for smoother output on noisy input

### Production Use
1. Validate generated audio before deployment
2. Consider ethical implications of voice cloning
3. Obtain proper consent when cloning voices
4. Use watermarking for generated content attribution

## API Reference

::: soundlab.voice
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
