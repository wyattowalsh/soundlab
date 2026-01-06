# SoundLab Examples

This directory contains example scripts demonstrating how to use SoundLab for various audio processing tasks.

## Prerequisites

Make sure you have SoundLab installed:

```bash
pip install soundlab
```

Or if you're developing locally:

```bash
cd packages/soundlab
pip install -e .
```

## Available Examples

### 1. Stem Separation (`separate_stems.py`)

Separate audio into individual stems (vocals, drums, bass, other) using the Demucs model.

**Basic Usage:**
```bash
python examples/separate_stems.py input.mp3
python examples/separate_stems.py input.mp3 -o output/ -m htdemucs_ft
python examples/separate_stems.py song.wav --two-stems vocals
```

**Features:**
- Multiple Demucs models (htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra, mdx_extra_q)
- Extract all stems or just one (two-stems mode)
- Configurable quality settings (shifts, segment length)
- GPU/CPU selection
- MP3 or WAV output

**Help:**
```bash
python examples/separate_stems.py --help
```

---

### 2. MIDI Transcription (`transcribe_to_midi.py`)

Transcribe audio to MIDI using the Basic Pitch model.

**Basic Usage:**
```bash
python examples/transcribe_to_midi.py input.mp3
python examples/transcribe_to_midi.py song.wav -o output.mid
python examples/transcribe_to_midi.py track.flac --onset 0.6 --save-pianoroll
```

**Features:**
- Adjustable detection thresholds (onset, frame)
- Minimum note length filtering
- Frequency range constraints
- Piano roll visualization
- Pitch bend support
- Detailed note statistics

**Help:**
```bash
python examples/transcribe_to_midi.py --help
```

---

### 3. Audio Analysis (`analyze_audio.py`)

Perform comprehensive audio analysis including tempo, key, loudness, spectral features, and onsets.

**Basic Usage:**
```bash
python examples/analyze_audio.py input.mp3
python examples/analyze_audio.py song.wav --no-spectral
python examples/analyze_audio.py track.flac --only tempo key loudness
```

**Features:**
- **Tempo**: BPM detection with confidence scores
- **Key**: Musical key detection (e.g., "A minor")
- **Loudness**: LUFS measurements (integrated, short-term, momentary)
- **Spectral**: Frequency content analysis (centroid, bandwidth, rolloff, etc.)
- **Onsets**: Beat and onset detection
- Selective analysis (enable/disable specific components)
- Formatted output with summary statistics

**Help:**
```bash
python examples/analyze_audio.py --help
```

---

### 4. Effects Processing (`apply_effects.py`)

Apply audio effects chains with preset configurations.

**Basic Usage:**
```bash
python examples/apply_effects.py input.wav output.wav --preset mastering
python examples/apply_effects.py vocals.mp3 processed.wav --preset vocal
python examples/apply_effects.py song.wav lofi.wav --preset lofi
python examples/apply_effects.py --list-presets
```

**Available Presets:**
- **mastering**: Professional mastering chain (EQ → Compression → Limiting)
- **vocal**: Vocal processing (HP filter → Compression → EQ → Reverb)
- **lofi**: Lo-fi aesthetic (Low-pass → Distortion → Chorus)
- **radio**: Radio/telephone effect (Bandpass → Compression → Distortion)
- **spacey**: Ambient/spacey atmosphere (Reverb → Delay → Chorus)

**Help:**
```bash
python examples/apply_effects.py --help
python examples/apply_effects.py --list-presets
```

---

### 5. Batch Processing (`batch_process.py`)

Process multiple audio files in a directory with stem separation, analysis, or effects.

**Basic Usage:**
```bash
# Analyze all files
python examples/batch_process.py audio_dir/ --mode analyze

# Separate all files
python examples/batch_process.py songs/ --mode separate -o stems/

# Apply effects to all files
python examples/batch_process.py tracks/ --mode effects -o processed/ --preset mastering
```

**Features:**
- **Analyze mode**: Comprehensive analysis with JSON report
- **Separate mode**: Batch stem separation
- **Effects mode**: Batch effects processing
- Progress tracking and error reporting
- Summary statistics
- Supports all common audio formats (WAV, MP3, FLAC, OGG, M4A, AAC, WMA)

**Help:**
```bash
python examples/batch_process.py --help
```

---

## Common Options

All scripts support these common options:

- `-v, --verbose`: Enable verbose logging for debugging
- `--help`: Show detailed help message with examples

## Supported Audio Formats

All examples support common audio formats:
- WAV
- MP3
- FLAC
- OGG
- M4A
- AAC
- WMA

## Tips

1. **GPU Acceleration**: If you have a CUDA-compatible GPU, the models will automatically use it for faster processing. You can force CPU mode with `--device cpu`.

2. **Quality vs Speed**:
   - For stem separation, `htdemucs_ft` provides the best quality but is slower
   - Increase `--shifts` for better quality (but longer processing time)

3. **Batch Processing**: Use `batch_process.py` for processing multiple files efficiently.

4. **Error Handling**: All scripts include comprehensive error handling and will show clear error messages. Use `-v` for detailed debugging information.

## Examples by Use Case

### Music Production Workflow

```bash
# 1. Separate stems
python examples/separate_stems.py song.mp3 -o stems/ -m htdemucs_ft

# 2. Analyze the original track
python examples/analyze_audio.py song.mp3

# 3. Apply mastering to the mix
python examples/apply_effects.py song.mp3 mastered.wav --preset mastering

# 4. Transcribe melody to MIDI
python examples/transcribe_to_midi.py stems/song/other.wav -o melody.mid
```

### Batch Music Analysis

```bash
# Analyze entire music library
python examples/batch_process.py ~/Music --mode analyze -o analysis/

# The JSON report includes BPM, key, loudness, and more for each file
```

### Vocal Processing Pipeline

```bash
# 1. Extract vocals
python examples/separate_stems.py song.mp3 --two-stems vocals

# 2. Process vocals
python examples/apply_effects.py separated/song/vocals.wav processed_vocals.wav --preset vocal

# 3. Transcribe vocal melody
python examples/transcribe_to_midi.py processed_vocals.wav vocals.mid
```

### Creating a Lo-Fi Remix

```bash
# 1. Separate the original
python examples/separate_stems.py original.mp3

# 2. Apply lo-fi effect to each stem
python examples/apply_effects.py separated/original/drums.wav lofi_drums.wav --preset lofi
python examples/apply_effects.py separated/original/bass.wav lofi_bass.wav --preset lofi
python examples/apply_effects.py separated/original/other.wav lofi_other.wav --preset lofi
```

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors during stem separation:
```bash
# Reduce segment length
python examples/separate_stems.py input.mp3 --segment 5.0

# Or force CPU mode
python examples/separate_stems.py input.mp3 --device cpu
```

### No GPU Detected

If you have a GPU but it's not being detected:
```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force GPU mode
python examples/separate_stems.py input.mp3 --device cuda
```

### Import Errors

If you get import errors:
```bash
# Make sure SoundLab is installed
pip install -e packages/soundlab

# Or install from PyPI
pip install soundlab
```

## Contributing

If you have ideas for new examples or improvements to existing ones, please:
1. Fork the repository
2. Create a new example following the existing pattern
3. Submit a pull request

## Support

For issues or questions:
- Open an issue on GitHub
- Check the main documentation
- See the API reference

## License

These examples are part of the SoundLab project and are licensed under the same terms.
