# Colab Usage Guide

Step-by-step guide to using SoundLab in Google Colab.

---

## Getting Started

### 1. Open the Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wyattowalsh/soundlab/blob/main/notebooks/soundlab_studio.ipynb)

Or upload `notebooks/soundlab_studio.ipynb` manually to Google Colab.

### 2. Enable GPU Runtime

For best performance, enable GPU acceleration:

1. Click **Runtime** → **Change runtime type**
2. Select **T4 GPU** (or better)
3. Click **Save**

<!-- Screenshot placeholder: runtime_settings.png -->

> [!NOTE]
> SoundLab works on CPU, but GPU acceleration provides 5-10x faster processing for stem separation.

---

## Notebook Walkthrough

### Cell 0: Resume From Checkpoint

Configure caching and resume previous runs:

```python
resume_run_id = ""  # Leave empty for new run, or enter previous run ID
cache_root = "/content/drive/MyDrive/soundlab_cache"
enable_drive_cache = True  # Persist models and checkpoints to Drive
```

<!-- Screenshot placeholder: cell0_checkpoint.png -->

**Key Options:**

| Option | Description |
|--------|-------------|
| `resume_run_id` | Enter a previous run ID to resume processing |
| `cache_root` | Directory for models and checkpoints |
| `enable_drive_cache` | Mount Google Drive for persistent storage |

> [!TIP]
> Enable Drive caching to avoid re-downloading models (~2GB) on each session.

### Cell 1: Header & Overview

Read the notebook overview and feature list. No configuration needed.

### Cell 2: Environment Setup

Configure runtime settings:

```python
gpu_mode = "auto"      # "auto", "force_gpu", "force_cpu"
log_level = "INFO"     # "DEBUG", "INFO", "WARNING", "ERROR"
output_base = "/content/soundlab_outputs"
deterministic = False  # Enable for reproducible results
random_seed = 42
```

<!-- Screenshot placeholder: cell2_environment.png -->

**Runtime Information Displayed:**

- GPU availability and model
- VRAM capacity
- Output directory paths

### Cell 3: Package Installation

Install SoundLab with optional extras:

```python
install_voice = False  # Enable TTS and voice conversion
install_from = "pypi"  # "pypi", "github_main", "github_dev"
force_reinstall = False
```

<!-- Screenshot placeholder: cell3_install.png -->

> [!IMPORTANT]
> First-time installation takes 2-3 minutes. Models download on first use.

---

## Processing Workflow

### Cell 4: Upload Audio

Upload your audio file using the Gradio interface:

<!-- Screenshot placeholder: cell4_upload.png -->

**Supported Formats:** WAV, MP3, FLAC, OGG, AIFF, M4A

**Metadata Displayed:**

- Duration
- Sample rate
- Channels (mono/stereo)
- File hash (for caching)

### Cell 5: Canonical Decode

Normalize audio to consistent format:

```python
target_sr = 44100      # Target sample rate
target_channels = 2    # 1 for mono, 2 for stereo
normalize_peak_db = -1.0  # Peak normalization level
```

<!-- Screenshot placeholder: cell5_decode.png -->

### Cell 6: Excerpt Selection

Select a portion of audio for quick testing:

```python
use_excerpt = True
excerpt_mode = "auto"  # "auto" or "manual"
excerpt_start = 0.0    # Start time (seconds)
excerpt_duration = 30.0  # Duration (seconds)
```

<!-- Screenshot placeholder: cell6_excerpt.png -->

> [!TIP]
> Use **auto** mode to automatically select the most energetic 30-second segment.

---

## Stem Separation

### Cell 7: Separation Configuration

Configure the Demucs model:

```python
model = "htdemucs_ft"  # Best quality
two_stems = None       # None for 4 stems, "vocals" for vocals/accompaniment
segment = 7.8          # Segment length (seconds)
overlap = 0.25         # Segment overlap ratio
shifts = 1             # Random shifts for better quality
output_format = "wav"  # "wav", "mp3", "flac"
```

<!-- Screenshot placeholder: cell7_separation.png -->

**Model Comparison:**

| Model | Quality | Speed | VRAM |
|-------|---------|-------|------|
| `htdemucs` | Good | ~45s | 4GB |
| `htdemucs_ft` | Best | ~3min | 6GB |
| `htdemucs_6s` | Good (6 stems) | ~1min | 5GB |

### Cell 8: Candidate Selection (Max-Quality Mode)

Run multiple separation strategies and select best results:

```python
enable_candidates = True
candidate_models = ["htdemucs", "htdemucs_ft"]
excerpt_qa = True  # Test on excerpt first
full_rerun = True  # Re-run best candidate on full track
```

<!-- Screenshot placeholder: cell8_candidates.png -->

---

## Transcription & Post-Processing

### Cell 9: Stem Post-Processing

Clean and prepare stems for transcription:

```python
min_silence_db = -60   # Silence threshold
alignment_safe = True  # Preserve timing alignment
create_mono_exports = True  # Create mono versions for transcription
```

<!-- Screenshot placeholder: cell9_postprocess.png -->

### Cell 10: Transcription

Convert stems to MIDI:

```python
transcribe_vocals = True
transcribe_bass = True
transcribe_other = True
onset_threshold = 0.5
frame_threshold = 0.3
min_note_length = 0.058
```

<!-- Screenshot placeholder: cell10_transcription.png -->

### Cell 11: MIDI Cleanup

Refine MIDI output:

```python
quantize = True
quantize_grid = 16     # 16th notes
quantize_strength = 0.5
filter_short_notes = True
min_note_duration = 0.05
```

<!-- Screenshot placeholder: cell11_midi_cleanup.png -->

---

## Review & Export

### Cell 12: QA Dashboard

Interactive preview and quality assessment:

<!-- Screenshot placeholder: cell12_qa_dashboard.png -->

**Features:**

- Audio players for each stem
- Residual audio (artifacts check)
- Reconstruction comparison
- QA metrics table
- Rerun controls

### Cell 13: Voice Generation (Optional)

Generate speech or convert voices:

```python
enable_tts = False
tts_text = "Hello, world!"
tts_language = "en"

enable_svc = False
svc_model_path = ""  # Path to RVC model
```

<!-- Screenshot placeholder: cell13_voice.png -->

### Cell 14: Export & Download

Package results for download:

```python
normalize_stems = True
normalize_target_lufs = -14.0
export_format = "wav"
create_zip = True
include_config = True
include_qa_report = True
```

<!-- Screenshot placeholder: cell14_export.png -->

**Export Contents:**

```
soundlab_export_<run_id>.zip
├── stems/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
├── midi/
│   ├── vocals.mid
│   ├── bass.mid
│   └── other.mid
├── config.json
├── qa_report.csv
└── manifest.json
```

### Cell 15: Cleanup

Free resources and clean temporary files:

```python
cleanup_stems = False
cleanup_midi = False
cleanup_all = False
clear_gpu_cache = True
```

<!-- Screenshot placeholder: cell15_cleanup.png -->

---

## Tips & Troubleshooting

### Memory Issues

If you encounter GPU out-of-memory errors:

1. Use a smaller model (`htdemucs` instead of `htdemucs_ft`)
2. Reduce segment length (e.g., `segment=5.0`)
3. Process shorter excerpts first
4. Run cleanup cell to free GPU memory

### Session Timeout

Colab sessions timeout after ~12 hours of inactivity:

1. Enable Google Drive caching (Cell 0)
2. Note your run ID from Cell 6
3. Resume using `resume_run_id` in Cell 0

### Slow Installation

First-time installation is slow due to dependencies:

1. Use Drive caching to persist installed packages
2. Consider using `github_main` for latest fixes

### Audio Quality Issues

If separation quality is poor:

1. Try `htdemucs_ft` model (best quality)
2. Increase `shifts` to 2-5 (slower but better)
3. Check input audio quality (44.1kHz+ recommended)

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run current cell |
| `Shift+Enter` | Run cell and advance |
| `Ctrl+M B` | Insert cell below |
| `Ctrl+M D` | Delete cell |
| `Ctrl+M Z` | Undo cell deletion |

---

## Next Steps

- [Quickstart Guide](quickstart.md) — Python API usage
- [Extending SoundLab](extending.md) — Custom effects and analyzers
- [GitHub Repository](https://github.com/wyattowalsh/soundlab) — Source code and issues
