<div align="center">

# 🎛️ SoundLab

[![CI](https://github.com/wyattowalsh/soundlab/actions/workflows/ci.yml/badge.svg)](https://github.com/wyattowalsh/soundlab/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/soundlab?color=blue)](https://pypi.org/project/soundlab/)
[![Python](https://img.shields.io/pypi/pyversions/soundlab)](https://pypi.org/project/soundlab/)
[![Coverage](https://codecov.io/gh/wyattowalsh/soundlab/branch/main/graph/badge.svg)](https://codecov.io/gh/wyattowalsh/soundlab)
[![License](https://img.shields.io/github/license/wyattowalsh/soundlab)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wyattowalsh/soundlab/blob/main/notebooks/soundlab_studio.ipynb)

**Production-ready music processing for stem separation, transcription, effects, and voice generation.**

[Documentation](https://wyattowalsh.github.io/soundlab) · [Examples](notebooks/examples/) · [Colab Notebook](https://colab.research.google.com/github/wyattowalsh/soundlab/blob/main/notebooks/soundlab_studio.ipynb)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎚️ **Stem Separation** | Demucs HTDemucs/HTDemucs-FT models for vocals, drums, bass, other |
| 🎹 **Audio-to-MIDI** | Basic Pitch transcription with configurable thresholds |
| 🎨 **Effects Processing** | Pedalboard-based EQ, compression, reverb, and creative effects |
| 📊 **Audio Analysis** | Tempo, key, loudness (LUFS), spectral features |
| 🗣️ **Voice Generation** | XTTS-v2 TTS and RVC voice conversion (optional) |
| 🔄 **Pipeline** | Checkpointed workflows with QA scoring and candidate selection |

---

## 📦 Installation

### From PyPI

```bash
pip install soundlab
```

### With Optional Extras

```bash
# Voice generation (TTS + RVC)
pip install soundlab[voice]

# Gradio interface for notebooks
pip install soundlab[notebook]

# Full installation
pip install soundlab[voice,notebook]
```

### From Source

```bash
git clone https://github.com/wyattowalsh/soundlab.git
cd soundlab
pip install -e ".[voice,notebook]"
```

> [!NOTE]
> Requires Python 3.12+. GPU recommended for stem separation.

---

## 🚀 Quick Start

### Stem Separation

```python
from soundlab.separation import StemSeparator, SeparationConfig, DemucsModel

# Configure and run separation
config = SeparationConfig(model=DemucsModel.HTDEMUCS_FT, device="auto")
separator = StemSeparator(config=config)
result = separator.separate("song.mp3", output_dir="stems/")

# Access stems
print(f"Vocals: {result.vocals}")
print(f"Drums: {result.stems['drums']}")
print(f"Processing time: {result.processing_time_seconds:.1f}s")
```

### Audio-to-MIDI Transcription

```python
from soundlab.transcription import MIDITranscriber, TranscriptionConfig

config = TranscriptionConfig(onset_thresh=0.5, frame_thresh=0.3)
transcriber = MIDITranscriber(config=config)
result = transcriber.transcribe("piano.wav", output_dir="midi/")

for note in result.notes[:5]:
    print(f"Pitch: {note.pitch}, Start: {note.start:.2f}s")
```

### Audio Analysis

```python
from soundlab import analyze_audio

result = analyze_audio("song.mp3")
print(f"Tempo: {result.tempo.bpm:.1f} BPM")
print(f"Key: {result.key.key_name} ({result.key.camelot})")
print(f"Loudness: {result.loudness.integrated_lufs:.1f} LUFS")
```

### Effects Processing

```python
from soundlab import load_audio, save_audio
from soundlab.effects import EffectsChain, CompressorConfig, ReverbConfig, LimiterConfig

audio = load_audio("vocals.wav")
chain = EffectsChain([
    CompressorConfig(threshold_db=-20, ratio=4.0),
    ReverbConfig(room_size=0.3, wet_level=0.2),
    LimiterConfig(threshold_db=-1.0),
])
processed = chain.process(audio.samples, audio.sample_rate)
save_audio(processed, audio.sample_rate, "vocals_processed.wav")
```

---

## 📓 Colab Notebook

Run SoundLab in Google Colab with an interactive UI:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wyattowalsh/soundlab/blob/main/notebooks/soundlab_studio.ipynb)

**Features:**
- GPU-accelerated processing
- Gradio interface for file upload/download
- Checkpoint resume for long sessions
- QA dashboard with audio previews

---

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Quickstart](docs/guides/quickstart.md) | Installation and basic usage |
| [Colab Usage](docs/guides/colab-usage.md) | Step-by-step notebook guide |
| [Extending](docs/guides/extending.md) | Add custom effects and analyzers |

---

## 🏗️ Project Structure

```
soundlab/
├── packages/soundlab/src/soundlab/
│   ├── analysis/      # Tempo, key, loudness, spectral
│   ├── effects/       # EQ, dynamics, time-based, creative
│   ├── separation/    # Demucs stem separation
│   ├── transcription/ # Basic Pitch audio-to-MIDI
│   ├── voice/         # TTS (XTTS-v2) and SVC (RVC)
│   ├── pipeline/      # Orchestration and QA
│   └── utils/         # GPU, logging, retry, progress
├── notebooks/         # Colab notebook and examples
├── tests/             # Unit and integration tests
└── docs/              # Documentation
```

---

## 🧪 Development

```bash
# Clone and install
git clone https://github.com/wyattowalsh/soundlab.git
cd soundlab
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Format and lint
uv run ruff format .
uv run ruff check .

# Type check
uv run ty check packages/soundlab/src
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## 📄 License

[MIT License](LICENSE) © Wyatt Walsh

---

<div align="center">

**[⬆ Back to Top](#-soundlab)**

</div>
