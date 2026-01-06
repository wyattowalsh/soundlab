# Voice Generation Guide

Learn how to generate natural speech in 18+ languages using XTTS-v2, including voice cloning and singing voice conversion.

!!! info "Optional Feature"
    Voice generation requires additional dependencies. Install with:
    ```bash
    pip install soundlab[voice]
    ```

## Overview

SoundLab provides advanced voice generation capabilities using:

- **XTTS-v2** - Text-to-speech with voice cloning (18+ languages)
- **RVC** - Singing voice conversion (coming soon)

### Use Cases

- **Content Creation** - Generate voiceovers for videos
- **Accessibility** - Convert text to speech for visually impaired
- **Game Development** - Create character voices
- **Prototyping** - Quick voice demos before recording
- **Language Learning** - Hear text in multiple languages
- **Voice Cloning** - Replicate specific voices

## Installation

Voice generation requires additional dependencies:

```bash
# With pip
pip install soundlab[voice]

# With uv
uv sync --extra voice

# Verify installation
python -c "from soundlab.voice import TextToSpeech; print('✓ Voice module ready')"
```

## Text-to-Speech (TTS)

### Quick Start

Generate speech in English:

```python
from soundlab.voice import TextToSpeech, TTSConfig

# Basic TTS (using default voice)
tts = TextToSpeech()
result = tts.synthesize(
    text="Welcome to SoundLab. This is a demonstration of text-to-speech.",
    output_path="output/speech.wav"
)

print(f"Generated: {result.audio_path}")
print(f"Duration: {result.duration_seconds:.1f}s")
print(f"Speaking rate: {result.words_per_minute:.0f} WPM")
```

### Supported Languages

XTTS-v2 supports 18+ languages:

```python
from soundlab.voice import TTSConfig, TTSLanguage

# English
config = TTSConfig(language=TTSLanguage.ENGLISH)

# Spanish
config = TTSConfig(language=TTSLanguage.SPANISH)

# French
config = TTSConfig(language=TTSLanguage.FRENCH)

# Chinese
config = TTSConfig(language=TTSLanguage.CHINESE)
```

**Available languages**:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | `en` | Spanish | `es` |
| French | `fr` | German | `de` |
| Italian | `it` | Portuguese | `pt` |
| Polish | `pl` | Turkish | `tr` |
| Russian | `ru` | Dutch | `nl` |
| Czech | `cs` | Arabic | `ar` |
| Chinese | `zh-cn` | Japanese | `ja` |
| Hungarian | `hu` | Korean | `ko` |
| Hindi | `hi` | | |

### Generation Parameters

#### Temperature

Controls randomness and expressiveness:

```python
# Conservative (more consistent)
config = TTSConfig(temperature=0.3)

# Default (balanced)
config = TTSConfig(temperature=0.7)

# Creative (more varied)
config = TTSConfig(temperature=0.9)
```

**Guidelines**:
- **0.3-0.5**: Consistent, predictable output
- **0.6-0.8**: Natural, varied speech (recommended)
- **0.9-1.0**: Very expressive, less consistent

#### Speed

Adjust speaking rate:

```python
# Slow (0.5x speed)
config = TTSConfig(speed=0.5)

# Normal (default)
config = TTSConfig(speed=1.0)

# Fast (1.5x speed)
config = TTSConfig(speed=1.5)

# Very fast (2x speed)
config = TTSConfig(speed=2.0)
```

**Use cases**:
- **0.5-0.7**: Meditation, relaxation, educational content
- **0.8-1.0**: Normal speech, audiobooks
- **1.2-1.5**: Fast-paced content, summaries
- **1.5-2.0**: Speed reading, previews

#### Advanced Parameters

Fine-tune generation quality:

```python
config = TTSConfig(
    temperature=0.7,
    length_penalty=1.0,        # Controls utterance length
    repetition_penalty=2.0,    # Prevents repetition (1.0-10.0)
    top_k=50,                  # Sampling diversity (1-100)
    top_p=0.85                 # Nucleus sampling (0.0-1.0)
)
```

## Voice Cloning

Clone voices from short audio samples.

### Requirements

For best results, reference audio should:

- Be 6-30 seconds long (15 seconds recommended)
- Contain a single speaker
- Have clear speech without music
- Have minimal background noise
- Maintain consistent microphone distance
- Use natural speaking pace

```python
from soundlab.voice import VoiceCloningRequirements

requirements = VoiceCloningRequirements()
print("\nVoice Cloning Guidelines:")
for guideline in requirements.guidelines:
    print(f"  - {guideline}")
```

### Basic Voice Cloning

Clone a voice from reference audio:

```python
from soundlab.voice import TextToSpeech, TTSConfig
from pathlib import Path

# Prepare reference audio
reference_audio = Path("voice_samples/speaker1.wav")

# Configure with voice cloning
config = TTSConfig(
    language="en",
    speaker_wav=reference_audio,
    temperature=0.7
)

# Generate speech with cloned voice
tts = TextToSpeech(config)
result = tts.synthesize(
    text="This is a test of voice cloning. The voice should sound like the reference speaker.",
    output_path="output/cloned_speech.wav"
)

print(f"Cloned voice output: {result.audio_path}")
```

### Preparing Reference Audio

Clean up reference audio for better cloning:

```python
from soundlab.separation import StemSeparator
from soundlab.effects import EffectsChain
from soundlab.effects.eq import HighPassFilterConfig
from soundlab.effects.dynamics import CompressorConfig, GateConfig

# Step 1: If reference has music, separate vocals
separator = StemSeparator()
stems = separator.separate("reference_with_music.wav", "output/stems/")
reference = stems.vocals

# Step 2: Clean up audio
cleanup_chain = (
    EffectsChain()
    # Remove low-frequency noise
    .add(HighPassFilterConfig(cutoff_frequency_hz=80))
    # Remove background noise
    .add(GateConfig(threshold_db=-45, ratio=10.0))
    # Even out levels
    .add(CompressorConfig(threshold_db=-20, ratio=3.0))
)

cleaned = cleanup_chain.process(reference, "output/reference_clean.wav")

# Step 3: Use cleaned reference for cloning
config = TTSConfig(speaker_wav=cleaned)
```

### Multi-Sample Voice Cloning

Use multiple samples for better quality:

```python
from soundlab.io import load_audio, save_audio
import numpy as np

# Load multiple samples
samples = [
    "sample1.wav",
    "sample2.wav",
    "sample3.wav"
]

audio_segments = []
for sample in samples:
    audio, sr = load_audio(sample)
    audio_segments.append(audio)

# Concatenate samples
combined = np.concatenate(audio_segments)

# Trim to recommended length (15 seconds)
target_samples = int(15.0 * sr)
if len(combined) > target_samples:
    combined = combined[:target_samples]

# Save combined reference
save_audio("combined_reference.wav", combined, sr)

# Use for cloning
config = TTSConfig(speaker_wav="combined_reference.wav")
```

## Usage Examples

### Example 1: Multi-Language Content

Generate speech in multiple languages:

```python
from soundlab.voice import TextToSpeech, TTSConfig, TTSLanguage

texts = {
    TTSLanguage.ENGLISH: "Welcome to our service.",
    TTSLanguage.SPANISH: "Bienvenido a nuestro servicio.",
    TTSLanguage.FRENCH: "Bienvenue dans notre service.",
    TTSLanguage.GERMAN: "Willkommen bei unserem Service.",
    TTSLanguage.CHINESE: "欢迎使用我们的服务。"
}

for language, text in texts.items():
    config = TTSConfig(language=language)
    tts = TextToSpeech(config)

    result = tts.synthesize(
        text=text,
        output_path=f"output/welcome_{language.value}.wav"
    )

    print(f"{language.value}: {result.audio_path}")
```

### Example 2: Audiobook Generation

Convert text to audiobook with voice cloning:

```python
from pathlib import Path

# Load book text
with open("book.txt") as f:
    book_text = f.read()

# Split into chapters/segments (for memory management)
segments = book_text.split("\n\n")  # Split by paragraphs

# Configure with narrator voice
config = TTSConfig(
    language="en",
    speaker_wav="narrator_voice.wav",
    temperature=0.7,
    speed=1.0
)

tts = TextToSpeech(config)

# Generate each segment
output_dir = Path("output/audiobook/")
output_dir.mkdir(parents=True, exist_ok=True)

for i, segment in enumerate(segments, 1):
    if not segment.strip():
        continue

    print(f"Generating segment {i}/{len(segments)}")

    result = tts.synthesize(
        text=segment,
        output_path=output_dir / f"segment_{i:03d}.wav"
    )

    print(f"  Duration: {result.duration_seconds:.1f}s")

print(f"\nGenerated {len(segments)} segments")
```

### Example 3: Character Voices

Create different character voices:

```python
characters = {
    "narrator": "voices/narrator.wav",
    "hero": "voices/hero.wav",
    "villain": "voices/villain.wav"
}

dialogue = [
    ("narrator", "Once upon a time, in a distant land..."),
    ("hero", "I will save the kingdom!"),
    ("villain", "You'll never stop me!"),
    ("narrator", "And so the battle began...")
]

for i, (character, text) in enumerate(dialogue, 1):
    config = TTSConfig(
        language="en",
        speaker_wav=characters[character],
        temperature=0.8  # More expressive
    )

    tts = TextToSpeech(config)
    result = tts.synthesize(
        text=text,
        output_path=f"output/dialogue_{i:02d}_{character}.wav"
    )

    print(f"{character}: {text[:30]}...")
```

### Example 4: Batch TTS Processing

Process multiple texts efficiently:

```python
from pathlib import Path

# Load texts
texts_file = Path("texts_to_generate.txt")
with open(texts_file) as f:
    texts = [line.strip() for line in f if line.strip()]

# Create single TTS instance (reuses model)
tts = TextToSpeech()

output_dir = Path("output/batch/")
output_dir.mkdir(parents=True, exist_ok=True)

for i, text in enumerate(texts, 1):
    print(f"Generating {i}/{len(texts)}")

    result = tts.synthesize(
        text=text,
        output_path=output_dir / f"output_{i:03d}.wav"
    )

    print(f"  {result.duration_seconds:.1f}s, {result.words_per_minute:.0f} WPM")

print(f"\nGenerated {len(texts)} audio files")
```

### Example 5: Voice Comparison

Compare different voices for the same text:

```python
test_text = "This is a test to compare different voice characteristics."

voice_samples = [
    "voice1.wav",
    "voice2.wav",
    "voice3.wav"
]

for i, voice_sample in enumerate(voice_samples, 1):
    config = TTSConfig(speaker_wav=voice_sample)
    tts = TextToSpeech(config)

    result = tts.synthesize(
        text=test_text,
        output_path=f"output/comparison_voice{i}.wav"
    )

    print(f"Voice {i}: {result.words_per_minute:.0f} WPM")
```

### Example 6: Interactive Voice Response

Generate IVR prompts:

```python
prompts = {
    "welcome": "Thank you for calling. Please listen carefully as our menu options have changed.",
    "main_menu": "Press 1 for sales. Press 2 for support. Press 3 for billing.",
    "hold": "Please hold while we connect your call.",
    "goodbye": "Thank you for calling. Goodbye."
}

config = TTSConfig(
    language="en",
    temperature=0.5,  # More consistent
    speed=0.9         # Slightly slower
)

tts = TextToSpeech(config)

for prompt_name, prompt_text in prompts.items():
    result = tts.synthesize(
        text=prompt_text,
        output_path=f"output/ivr/{prompt_name}.wav"
    )
    print(f"Generated: {prompt_name}.wav")
```

## Voice Analysis

Analyze generated voice characteristics:

```python
from soundlab.analysis import analyze_audio
from soundlab.io import load_audio
import numpy as np

result = tts.synthesize(
    text="Sample text for analysis.",
    output_path="sample.wav"
)

# Analyze audio
analysis = analyze_audio("sample.wav")

# Speaking rate
print(f"Words per minute: {result.words_per_minute:.0f}")
print(f"Duration: {result.duration_seconds:.1f}s")

# Spectral characteristics
if analysis.spectral:
    print(f"Spectral centroid: {analysis.spectral.spectral_centroid:.0f} Hz")
    print(f"Brightness: {analysis.spectral.brightness}")

# Loudness
if analysis.loudness:
    print(f"Loudness: {analysis.loudness.integrated_lufs:.1f} LUFS")

# Pitch range (approximate)
audio, sr = load_audio("sample.wav")
# Pitch detection would require additional analysis
```

## Post-Processing

Enhance generated speech:

### Normalize Levels

```python
from soundlab.io import load_audio, save_audio
import numpy as np

# Generate speech
result = tts.synthesize(text="Test", output_path="raw.wav")

# Normalize
audio, sr = load_audio("raw.wav")
audio_normalized = audio / np.max(np.abs(audio)) * 0.9
save_audio("normalized.wav", audio_normalized, sr)
```

### Add Background Music

```python
from soundlab.io import load_audio, save_audio
import numpy as np

# Load voice and music
voice, sr = load_audio("speech.wav")
music, _ = load_audio("background.wav")

# Match lengths
min_length = min(len(voice), len(music))
voice = voice[:min_length]
music = music[:min_length]

# Mix (voice at full volume, music at 20%)
mixed = voice + (music * 0.2)

# Normalize
mixed = mixed / np.max(np.abs(mixed)) * 0.9

save_audio("with_music.wav", mixed, sr)
```

### Apply Effects

```python
from soundlab.effects import EffectsChain
from soundlab.effects.eq import HighPassFilterConfig, PeakFilterConfig
from soundlab.effects.dynamics import CompressorConfig
from soundlab.effects.time_based import ReverbConfig

# Generate speech
result = tts.synthesize(text="Test", output_path="raw.wav")

# Apply voice processing
voice_chain = (
    EffectsChain()
    .add(HighPassFilterConfig(cutoff_frequency_hz=80))
    .add(PeakFilterConfig(cutoff_frequency_hz=3000, gain_db=2.0))
    .add(CompressorConfig(threshold_db=-20, ratio=3.0))
    .add(ReverbConfig(room_size=0.3, wet_level=0.15))
)

processed = voice_chain.process("raw.wav", "processed.wav")
```

## Best Practices

### Reference Audio Quality

Prepare high-quality reference audio:

1. **Record in quiet environment**
2. **Use good microphone** (not phone/laptop mic)
3. **Maintain consistent distance** from mic
4. **Speak naturally** - not too fast or slow
5. **Include variety** - different words, intonations
6. **Avoid breathing sounds** and pops

### Text Preparation

Format text for better results:

```python
def prepare_text_for_tts(text):
    """Clean and format text for TTS."""
    import re

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Expand common abbreviations
    text = text.replace("Mr.", "Mister")
    text = text.replace("Dr.", "Doctor")
    text = text.replace("etc.", "et cetera")

    # Add pauses with punctuation
    text = text.replace("...", ", ")  # Convert ellipsis to pause

    # Remove special characters
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    return text.strip()

# Use cleaned text
clean_text = prepare_text_for_tts(original_text)
result = tts.synthesize(text=clean_text, output_path="output.wav")
```

### Memory Management

For large-scale generation:

```python
# Process in batches
def batch_tts(texts, batch_size=10):
    """Generate TTS in batches to manage memory."""
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Create TTS for this batch
        tts = TextToSpeech()

        for j, text in enumerate(batch):
            result = tts.synthesize(
                text=text,
                output_path=f"output/batch_{i+j}.wav"
            )

        # Clear model from memory
        del tts

        # Optional: garbage collection
        import gc
        gc.collect()
```

## Common Issues and Solutions

### Poor Voice Quality

**Problem**: Generated speech sounds robotic or unnatural.

**Solutions**:

```python
# 1. Adjust temperature (higher = more natural)
config = TTSConfig(temperature=0.8)

# 2. Use better reference audio
# - Ensure 15 seconds of clean speech
# - Single speaker only
# - No background noise

# 3. Adjust speed
config = TTSConfig(speed=1.0)  # Don't go too fast/slow
```

### Voice Doesn't Match Reference

**Problem**: Cloned voice differs from reference.

**Solutions**:

```python
# 1. Check reference audio length
# Should be 6-30 seconds, ideally 15 seconds

# 2. Clean reference audio (remove noise, music)
from soundlab.separation import StemSeparator
separator = StemSeparator()
stems = separator.separate("reference.wav", "output/")
# Use vocals stem

# 3. Try multiple reference samples
# Combine different samples of same speaker

# 4. Verify speaker_wav parameter
config = TTSConfig(
    speaker_wav="path/to/reference.wav"  # Ensure path is correct
)
```

### Slow Generation

**Problem**: TTS takes too long.

**Solutions**:

```python
# 1. Use GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")

# 2. Reuse TTS instance for multiple generations
tts = TextToSpeech()  # Create once
for text in texts:
    result = tts.synthesize(text, f"output_{i}.wav")
    # Don't recreate TTS each time

# 3. Process in batches
# See memory management section above
```

### Out of Memory

**Problem**: GPU/CPU runs out of memory.

**Solutions**:

```python
# 1. Force CPU if GPU memory is limited
config = TTSConfig(device="cpu")

# 2. Process shorter text segments
def chunk_text(text, max_words=100):
    """Split text into smaller chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

# 3. Clear GPU cache between generations
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## Performance

Generation times for typical use:

| Text Length | Time (GPU) | Time (CPU) |
|-------------|-----------|-----------|
| 1 sentence | ~2 sec | ~8 sec |
| 1 paragraph | ~5 sec | ~20 sec |
| 1 page | ~30 sec | ~2 min |
| 10 pages | ~5 min | ~20 min |

!!! tip "Optimization"
    - GPU is 4-5x faster than CPU
    - Reusing model instance saves loading time
    - Batch processing is most efficient

## Future Features

### Singing Voice Conversion (Coming Soon)

Transform singing voices using RVC:

```python
from soundlab.voice import SingingVoiceConverter, SVCConfig

# Configure voice conversion
config = SVCConfig(
    model_path="path/to/rvc_model.pth",
    pitch_shift_semitones=0,
    f0_method="rmvpe"
)

# Convert vocals
converter = SingingVoiceConverter(config)
result = converter.convert(
    audio_path="original_vocals.wav",
    output_path="converted_vocals.wav"
)
```

## Next Steps

- **[Separation Guide](separation.md)** - Extract vocals for voice conversion
- **[Effects Guide](effects.md)** - Process generated speech
- **[Quick Start](quickstart.md)** - More examples

---

**Questions?** Visit [GitHub Discussions](https://github.com/wyattwalsh/soundlab/discussions).
