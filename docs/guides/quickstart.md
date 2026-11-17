# Quickstart Guide

Install SoundLab from this repository and call the current library API.

The repository policy page is the root [README.md](../../README.md). SoundLab is beta. `soundlab.__version__` is `0.1.0`. Package metadata in `packages/soundlab/pyproject.toml` is `1.0.0`. PyPI publication is disabled. The PyPI project named `soundlab` is a different project.

---

## Installation

The workspace root project is `soundlab-workspace`. It has no library dependency, so `uv sync` without `--all-packages` does not install SoundLab.

```bash
git clone https://github.com/wyattowalsh/soundlab.git
cd soundlab
uv sync --all-packages
```

To include the notebook extra, as the repository README does:

```bash
uv sync --all-packages --extra notebook
```

There is no `uv.lock` in this repository.

Extras live on `packages/soundlab`: `notebook`, `voice`, `transcription`, `visualization`, and `all`. The `all` extra selects transcription, voice, notebook, and visualization together. A local editable install of the notebook extra, which leaves `all` unselected, is:

```bash
uv pip install -e "packages/soundlab[notebook]"
```

Install another extra the same way, for example `packages/soundlab[voice]`. A git install needs the package subdirectory:

```bash
uv pip install "soundlab[notebook] @ git+https://github.com/wyattowalsh/soundlab.git#subdirectory=packages/soundlab"
```

Run those commands from the repository root. The root project does not declare extras.

---

## Python version

| Version | Where it applies |
|---------|------------------|
| Python 3.10+ | Library package (`packages/soundlab`) |
| Python 3.12+ | Workspace root (`soundlab-workspace`) and local `uv sync` |
| Python 3.13+ | MP3 decoding through pydub may need `audioop-lts` |

The `transcription` extra installs Basic Pitch on Python older than 3.12 and CREPE on Python 3.12+.

On Python 3.13+, WAV and FLAC avoid the `audioop` path:

```python
from soundlab import AudioFormat, save_audio

save_audio(segment, "output.flac", format=AudioFormat.FLAC)
```

---

## Basic usage

### Load audio

```python
from soundlab import get_audio_metadata, load_audio

audio = load_audio("song.mp3")
print(f"Duration: {audio.duration_seconds:.2f}s")
print(f"Sample rate: {audio.sample_rate} Hz")
print(f"Channels: {audio.channels}")

meta = get_audio_metadata("song.mp3")
print(meta.duration_seconds, meta.sample_rate, meta.channels)
```

`AudioSegment.duration_seconds` is the frame count on the time axis divided by the sample rate. `AudioMetadata.duration_seconds` is the container duration.

### Stem separation

```python
from soundlab.separation import DemucsModel, SeparationConfig, StemSeparator

config = SeparationConfig(
    model=DemucsModel.HTDEMUCS_FT,
    device="auto",
)
separator = StemSeparator(config=config)
result = separator.separate("song.mp3", output_dir="stems/")

print(f"Vocals: {result.vocals}")
print(f"Drums: {result.stems['drums']}")
print(f"Bass: {result.stems['bass']}")
print(f"Other: {result.stems['other']}")
print(f"Processing time: {result.processing_time_seconds:.1f}s")
```

### Vocal isolation

```bash
soundlab separate input.mp3 output/ --vocals-only
```

```python
from soundlab.separation import SeparationConfig, StemSeparator

config = SeparationConfig(two_stems="vocals")
separator = StemSeparator(config)
result = separator.separate("song.mp3", "output/")

print(result.vocals)
print(result.stems["no_vocals"])
print(result.instrumental)
```

`--vocals-only` sets `two_stems="vocals"`. That writes `vocals.wav` and `no_vocals.wav`. `StemResult.instrumental` returns an existing `no_vocals` or `instrumental` path and does not write a file. Call `result.export_instrumental()` to sum the non-vocal stems into `instrumental.wav`.

| Model | Stems |
|-------|-------|
| `HTDEMUCS` | vocals, drums, bass, other |
| `HTDEMUCS_FT` | vocals, drums, bass, other |
| `HTDEMUCS_6S` | those four, plus piano and guitar |
| `MDX_EXTRA` | vocals, drums, bass, other |
| `MDX_EXTRA_Q` | vocals, drums, bass, other |

### Audio-to-MIDI

```python
from soundlab.transcription import MIDITranscriber, TranscriptionConfig

config = TranscriptionConfig(
    onset_thresh=0.5,
    frame_thresh=0.3,
    min_note_length=0.058,
)
transcriber = MIDITranscriber(config=config)
result = transcriber.transcribe("piano.wav", output_dir="midi/")

for note in result.notes[:10]:
    duration = note.end - note.start
    print(f"Pitch: {note.pitch}, Start: {note.start:.2f}s, Duration: {duration:.2f}s")

print(f"MIDI saved to: {result.path}")
```

`NoteEvent` stores `start` and `end`. It has no `duration` field.

### Drum-to-MIDI

```python
from soundlab.separation import StemSeparator

separator = StemSeparator()
result = separator.separate("song.mp3", "stems/")
drum_path = result.stems["drums"]
```

`DrumTranscriber` takes `DrumConfig` from the drum backend. `onset_thresh` is the onset field.

```python
from soundlab.transcription.drum_backend import DrumConfig, DrumTranscriber

transcriber = DrumTranscriber(DrumConfig(onset_thresh=0.3))
midi_result = transcriber.transcribe(drum_path, "output/")

print(midi_result.path)
print(len(midi_result.notes))
```

### Analysis

`analyze_audio` accepts a path. `detect_tempo`, `detect_key`, and `measure_loudness` accept samples and a sample rate.

```python
from soundlab import analyze_audio, detect_key, detect_tempo, load_audio, measure_loudness

result = analyze_audio("song.mp3")
print(f"Tempo: {result.tempo.bpm:.1f} BPM")
print(f"Key: {result.key.name} ({result.key.camelot})")
print(f"Loudness: {result.loudness.lufs:.1f} LUFS")

segment = load_audio("song.mp3")
tempo = detect_tempo(segment.samples, segment.sample_rate)
print(f"Tempo: {tempo.bpm:.1f} BPM (confidence: {tempo.confidence:.2f})")

key = detect_key(segment.samples, segment.sample_rate)
print(f"Key: {key.key.value} {key.mode.value} ({key.camelot})")

loudness = measure_loudness(segment.samples, segment.sample_rate)
print(f"Integrated: {loudness.lufs:.1f} LUFS")
print(f"Peak: {loudness.peak:.3f}")
```

### Effects

Build the chain with `add`. Use `process_array` for samples, or `process` for file paths. `save_audio` takes the segment and the destination path.

```python
from soundlab import load_audio, save_audio
from soundlab.effects import (
    CompressorConfig,
    EffectsChain,
    HighpassConfig,
    LimiterConfig,
    ReverbConfig,
)

audio = load_audio("vocals.wav")
chain = (
    EffectsChain()
    .add(HighpassConfig(cutoff_frequency_hz=80))
    .add(CompressorConfig(threshold_db=-20, ratio=4.0))
    .add(ReverbConfig(room_size=0.3, wet=0.2))
    .add(LimiterConfig(threshold_db=-1.0))
)

processed = chain.process_array(audio.samples, audio.sample_rate)
audio.samples = processed
save_audio(audio, "vocals_processed.wav")

# File form: reads the source path and writes the destination.
chain.process("vocals.wav", "vocals_from_path.wav")
```

| Category | Config types |
|----------|----------------|
| Dynamics | `CompressorConfig`, `LimiterConfig`, `GateConfig`, `GainConfig` |
| EQ | `HighpassConfig`, `LowpassConfig` |
| Time-based | `ReverbConfig`, `DelayConfig` |
| Creative | `ChorusConfig`, `PhaserConfig`, `DistortionConfig` |

Each config implements `to_plugin()`. The chain calls that method while it builds the Pedalboard.

### Voice generation

Install the voice extra from this package (`packages/soundlab[voice]`). `TTSGenerator.generate` accepts a `TTSConfig` and returns a `TTSResult` whose audio is already written to `audio_path`.

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

---

## Pipeline

`init_run` takes a `PipelineConfig`, the source path, and a keyword-only cache root. It returns `RunArtifacts`.

```python
from pathlib import Path

from soundlab.pipeline import PipelineConfig, QAConfig, build_candidate_plans, init_run

config = PipelineConfig(
    qa=QAConfig(
        max_reconstruction_error=0.15,
        max_clipping_ratio=0.01,
    ),
)
artifacts = init_run(config, Path("song.mp3"), root=Path("/path/to/cache"))
plans = build_candidate_plans(config)

print(artifacts.run_id)
print(artifacts.output_dir)
print([plan.name for plan in plans])
```

`choose_best_candidate` selects from `CandidateScore` values. `score_separation` scores in-memory arrays: `score_separation(mix, stems, sample_rate, qa=config.qa)`.

---

## Configuration

| Variable | Role | Default |
|----------|------|---------|
| `SOUNDLAB_LOG_LEVEL` | Loguru level used by `configure_logging` | `INFO` |
| `SOUNDLAB_GPU_MODE` | Overrides the mode passed to `get_device` | unset; the function argument is used |
| `SOUNDLAB_CACHE_DIR` | Cache directory read by `SoundLabConfig` and TTS | `~/.cache/soundlab` |
| `SOUNDLAB_OUTPUT_DIR` | Voice output directory | `outputs` |
| `SOUNDLAB_RVC_ROOT` | External RVC checkout for `VoiceConverter` | unset |

```python
from soundlab.utils import configure_logging, get_device

configure_logging(level="DEBUG")
device = get_device("auto")
print(device)
```

`get_device` requires a mode. `"auto"` returns `"cuda"` when CUDA is available and `"cpu"` otherwise. `"force_gpu"` and `"force_cpu"` are also accepted. `SOUNDLAB_GPU_MODE` replaces the argument when it is set.

---

## Next steps

- [Colab usage](colab-usage.md)
- [Extending SoundLab](extending.md)
- [Contributing](../../CONTRIBUTING.md)
