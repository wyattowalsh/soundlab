<div align="center">

# SoundLab

[![CI](https://github.com/wyattowalsh/soundlab/actions/workflows/ci.yml/badge.svg)](https://github.com/wyattowalsh/soundlab/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/wyattowalsh/soundlab/branch/main/graph/badge.svg)](https://codecov.io/gh/wyattowalsh/soundlab)
[![License](https://img.shields.io/github/license/wyattowalsh/soundlab)](../../LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wyattowalsh/soundlab/blob/main/notebooks/soundlab_studio.ipynb)

**Beta music-processing library for stem separation, transcription, effects, and voice generation.**

[Repository policy](../../README.md) · [Quickstart](../../docs/guides/quickstart.md) · [Colab notebook](../../notebooks/soundlab_studio.ipynb)

</div>

---

SoundLab is a beta package. `soundlab.__version__` is `0.1.0`. The package metadata version in `packages/soundlab/pyproject.toml` is `1.0.0`. PyPI publication is disabled. The PyPI project named `soundlab` is a different project.

The repository policy page is [`README.md`](../../README.md).

## Features

| Feature | Description |
|---------|-------------|
| Stem separation | Demucs models for vocals, drums, bass, other, and 6-stem sets |
| Audio-to-MIDI | Basic Pitch and CREPE transcription, selected by runtime |
| Effects | Pedalboard EQ, compression, reverb, and creative effects |
| Analysis | Tempo, key, loudness, and spectral features |
| Voice | Optional XTTS-v2 TTS and an RVC conversion interface |
| Pipeline | Candidate plans, QA scoring, and run artifacts |

Capability availability depends on the selected backend, Python version, model terms, and runtime resources.

## Installation

The workspace root project is `soundlab-workspace`. It does not depend on this library, so `uv sync` without `--all-packages` does not install it.

```bash
git clone https://github.com/wyattowalsh/soundlab.git
cd soundlab
uv sync --all-packages
```

The repository README also installs the notebook extra:

```bash
uv sync --all-packages --extra notebook
```

There is no `uv.lock` in this repository.

Extras are declared on `packages/soundlab` (`notebook`, `voice`, `transcription`, `visualization`, and `all`). A local editable install of the notebook extra, which does not select the `all` extra, is:

```bash
uv pip install -e "packages/soundlab[notebook]"
```

A git install needs the package subdirectory:

```bash
uv pip install "soundlab[notebook] @ git+https://github.com/wyattowalsh/soundlab.git#subdirectory=packages/soundlab"
```

The library requires Python 3.10+. The workspace root requires Python 3.12+. A GPU is recommended for stem separation.

## Quick start

### Stem separation

```python
from soundlab.separation import StemSeparator, SeparationConfig, DemucsModel

config = SeparationConfig(model=DemucsModel.HTDEMUCS_FT, device="auto")
separator = StemSeparator(config=config)
result = separator.separate("song.mp3", output_dir="stems/")

print(f"Vocals: {result.vocals}")
print(f"Drums: {result.stems['drums']}")
print(f"Processing time: {result.processing_time_seconds:.1f}s")
```

### Audio-to-MIDI

```python
from soundlab.transcription import MIDITranscriber, TranscriptionConfig

config = TranscriptionConfig(onset_thresh=0.5, frame_thresh=0.3)
transcriber = MIDITranscriber(config=config)
result = transcriber.transcribe("piano.wav", output_dir="midi/")

for note in result.notes[:5]:
    print(f"Pitch: {note.pitch}, Start: {note.start:.2f}s, End: {note.end:.2f}s")
```

### Analysis

```python
from soundlab import analyze_audio

result = analyze_audio("song.mp3")
print(f"Tempo: {result.tempo.bpm:.1f} BPM")
print(f"Key: {result.key.name} ({result.key.camelot})")
print(f"Loudness: {result.loudness.lufs:.1f} LUFS")
```

### Effects

```python
from soundlab import load_audio, save_audio
from soundlab.effects import EffectsChain, CompressorConfig, ReverbConfig, LimiterConfig

audio = load_audio("vocals.wav")
chain = (
    EffectsChain()
    .add(CompressorConfig(threshold_db=-20, ratio=4.0))
    .add(ReverbConfig(room_size=0.3, wet=0.2))
    .add(LimiterConfig(threshold_db=-1.0))
)
processed = chain.process_array(audio.samples, audio.sample_rate)
audio.samples = processed
save_audio(audio, "vocals_processed.wav")
```

`EffectsChain.process("vocals.wav", "vocals_processed.wav")` reads and writes paths. Each effect supplies its Pedalboard plugin through `to_plugin()`.

### Text-to-speech

The voice extra is optional. Install it from this package, for example `uv pip install -e "packages/soundlab[voice]"`.

```python
from pathlib import Path

from soundlab.voice import TTSGenerator
from soundlab.voice.models import TTSConfig

result = TTSGenerator().generate(
    TTSConfig(
        text="Hello, welcome to SoundLab!",
        language="en",
        speaker_wav=Path("reference_voice.wav"),
    )
)
print(result.audio_path)
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Quickstart](../../docs/guides/quickstart.md) | Installation and library usage |
| [Colab usage](../../docs/guides/colab-usage.md) | Studio notebook walkthrough |
| [Extending](../../docs/guides/extending.md) | Effects, analyzers, and separation backends |
| [Contributing](../../CONTRIBUTING.md) | Local setup and validation |
| [Changelog](../../CHANGELOG.md) | Release notes and current status |

Examples live under [`notebooks/examples/`](../../notebooks/examples/).

## Project structure

```text
soundlab/
├── packages/soundlab/src/soundlab/
│   ├── analysis/
│   ├── effects/
│   ├── separation/
│   ├── transcription/
│   ├── voice/
│   ├── pipeline/
│   └── utils/
├── notebooks/
├── tests/
└── docs/
```

## Development

```bash
uv sync --all-packages --extra notebook
uv run ruff format --check .
uv run ruff check .
uv run ty check packages/soundlab/src
uv run pytest tests/ -m "not slow and not gpu"
```

See [CONTRIBUTING.md](../../CONTRIBUTING.md).

## License

[MIT License](../../LICENSE). Third-party model weights and services keep their own terms.
