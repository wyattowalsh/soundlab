# Colab Usage Guide

Walkthrough of [`notebooks/soundlab_studio.ipynb`](../../notebooks/soundlab_studio.ipynb). Control names below are the form variables in that notebook.

The repository policy page is the root [README.md](../../README.md). Mounting Drive, publishing a Gradio link, hosted uploads, and paid processing stay explicit opt-ins.

---

## Getting Started

### 1. Open the notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wyattowalsh/soundlab/blob/main/notebooks/soundlab_studio.ipynb)

Or upload `notebooks/soundlab_studio.ipynb` to Colab.

### 2. Choose a runtime

1. Click **Runtime** → **Change runtime type**
2. Select a GPU runtime when you want GPU separation
3. Click **Save**

The notebook also runs on CPU. It reports the device it actually sees. A GPU runtime type does not by itself prove the selected model fits in memory.

---

## Notebook walkthrough

### Resume

```python
resume_run_id = ""  # empty starts a new run
cache_root = "/content/drive/MyDrive/soundlab_cache"
enable_drive_cache = False
```

| Control | Role |
|---------|------|
| `resume_run_id` | Previous run id to reload. Empty starts fresh |
| `cache_root` | Directory for models and checkpoints |
| `enable_drive_cache` | Mount Google Drive when you want that cache to persist |

Leave `enable_drive_cache` off for a local session. Turn it on only when you want Drive storage.

### Environment

```python
gpu_mode = "auto"  # "auto", "force_gpu", "force_cpu"
log_level = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
output_base = "/content/soundlab_outputs"
deterministic = False
random_seed = 42
```

The cell prints GPU availability, GPU name, memory, the selected `gpu_mode`, and the output directory.

### Install

```python
install_voice = False
install_from = "github_main"  # "github_main" or "github_dev"
force_reinstall = False
```

The install cell builds `soundlab[notebook] @ git+https://github.com/wyattowalsh/soundlab.git#subdirectory=packages/soundlab`. `github_dev` pins that URL to the `dev` branch. `install_voice` adds the `voice` extra on the same package. The PyPI project named `soundlab` is a different project, and publication of this package is disabled.

### Upload

The upload cell accepts WAV, MP3, FLAC, OGG, AIFF, and M4A. It shows duration, sample rate, channel count, and a file hash.

### Canonical decode

```python
target_sample_rate = 44100
target_channels = 2  # 1 or 2
normalize_audio = False
normalization_level = -1.0  # peak level in dB, used when normalize_audio is true
```

`normalization_level` is the peak target in dB. Normalization runs only when `normalize_audio` is true.

### Excerpt

```python
use_excerpt = False
excerpt_start_seconds = 30.0
excerpt_duration_seconds = 30.0
auto_select_excerpt = False
```

Set `use_excerpt` false to keep the full track. Set `auto_select_excerpt` true to start the excerpt at the most energetic region of `excerpt_duration_seconds`. This cell prints the run id used later by `resume_run_id`.

---

## Stem separation

```python
separation_model = "htdemucs_ft"  # htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra, mdx_extra_q
two_stems = "none"  # none, vocals, drums, bass, other
segment_length = 7.8
overlap = 0.25
shifts = 1
output_format = "wav"  # wav, mp3, flac
mp3_bitrate = 320
float32_output = True
```

| `separation_model` | Stems |
|--------------------|-------|
| `htdemucs` | 4 |
| `htdemucs_ft` | 4 |
| `htdemucs_6s` | 6 |
| `mdx_extra` | 4 |
| `mdx_extra_q` | 4 |

`two_stems = "none"` separates the full stem set. Any other value keeps that stem and a complement. `output_format` chooses the stem container. `wav` can stay float when `float32_output` is on. `flac` and `mp3` are integer files, so leave `float32_output` off for those. `mp3_bitrate` is the MP3 rate in kilobits per second.

### Candidate selection

```python
enable_candidate_selection = True
excerpt_start = 30.0
excerpt_duration = 30.0
max_candidates = 3
min_overall_score = 0.7
max_reconstruction_error = 0.15
max_clipping_ratio = 0.01
rerun_full_on_best = True
```

`rerun_full_on_best` repeats the chosen candidate on the full track. Leave `enable_candidate_selection` false to keep the single separation from the previous cell.

---

## Transcription and post-processing

### Stem post-processing

```python
silence_threshold = 1e-4
enable_silence_filtering = False
clipping_threshold = 0.99
max_clipping_ratio = 0.01
prepare_mono_for_amt = True
save_mono_stems = True
```

### Transcription

```python
transcribe_vocals = False
transcribe_drums = False
transcribe_bass = True
transcribe_other = True
transcribe_piano = True
transcribe_guitar = True
onset_threshold = 0.5
frame_threshold = 0.3
min_note_length = 0.058
```

The same cell also has `bass_min_freq`, `bass_max_freq`, `default_min_freq`, `default_max_freq`, `enable_fallback`, and `fallback_lower_threshold`.

### MIDI cleanup

```python
min_note_duration = 0.02
min_velocity = 1
filter_short_notes = True
detect_tempo_from_audio = True
fallback_bpm = 120.0
enable_quantization = False
quantize_strength = 0.5
quantize_grid = "1/16"  # "1/4", "1/8", "1/16", "1/32"
enable_program_mapping = True
```

---

## Review and export

The QA cell previews stems, residual and reconstruction audio, metrics, and rerun controls. It has no extra form variables.

### Voice

Voice cells require `install_voice` in the install cell.

```python
enable_tts = False
tts_text = "Hello, this is a voice generation test."
tts_language = "en"
tts_speaker_wav = ""

enable_svc = False
svc_source_stem = "vocals"  # vocals or other
svc_pitch_shift = 0
svc_model_path = ""
svc_index_path = ""
```

### Export

```python
export_stems = True
export_midi = True
export_voice = True
export_analysis = True
export_config = True
export_qa_report = True
stem_format = "wav"  # wav, mp3, flac
normalize_stems = False
normalization_lufs = -14.0
create_zip_archive = True
zip_filename = "soundlab_export"
auto_download = True
```

When those sections have outputs, the archive name starts with `zip_filename` and can include `stems/`, `midi/`, voice files, analysis files, `config.json`, and `qa_report.csv`.

### Cleanup

```python
cleanup_stems = False
cleanup_midi = False
cleanup_voice = False
cleanup_analysis = False
cleanup_exports = False
cleanup_cache = False
clear_gpu_cache = True
reset_global_state = False
```

Each `cleanup_*` flag removes that output family. `clear_gpu_cache` frees CUDA memory. `reset_global_state` clears the notebook's in-memory run state.

---

## Troubleshooting

GPU memory:

1. Choose a smaller `separation_model`, such as `htdemucs`.
2. Lower `segment_length`. The notebook slider starts at 5.0.
3. Shorten `excerpt_duration_seconds` before a full-track run.
4. Run the cleanup cell with `clear_gpu_cache` enabled.

Session timeout:

1. Set `enable_drive_cache` if you want the cache on Drive.
2. Copy the run id printed by the excerpt cell.
3. Pass it as `resume_run_id` on the next session.

A local timeout does not prove that a remote job was cancelled.

Separation quality:

1. Try `separation_model = "htdemucs_ft"`.
2. Raise `shifts`. The slider allows 0 through 5.
3. Check the source sample rate. The decode cell's `target_sample_rate` defaults to 44100.

---

## Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run current cell |
| `Shift+Enter` | Run cell and advance |
| `Ctrl+M B` | Insert cell below |
| `Ctrl+M D` | Delete cell |
| `Ctrl+M Z` | Undo cell deletion |

---

## Next steps

- [Quickstart](quickstart.md)
- [Extending SoundLab](extending.md)
- [Repository](https://github.com/wyattowalsh/soundlab)
