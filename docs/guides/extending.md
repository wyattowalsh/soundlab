# Extending SoundLab

How to add an effect, an analyzer, a voice wrapper, or a separation backend.

Reusable behavior belongs in `packages/soundlab/src/soundlab/`. The studio notebook is the interactive entrypoint.

```text
soundlab/
├── analysis/      # tempo, key, loudness, spectral
├── effects/       # Pedalboard effect configs and EffectsChain
├── separation/    # Demucs StemSeparator
├── transcription/ # audio-to-MIDI
├── voice/         # TTS and SVC
├── pipeline/      # plans, QA, and backend protocols
└── core/          # audio models and errors
```

A config implements `to_plugin()`, and the chain calls it. Separation backends implement `SeparatorBackend.separate(mix_wav, output_dir)`.

---

## Add an effect

Subclass `EffectConfig` and implement `to_plugin()`. `EffectsChain._build_board` calls `effect.to_plugin()` for every config passed to `add`.

```python
from pydantic import Field
from pedalboard import Chorus

from soundlab.effects.models import EffectConfig


class FlangerConfig(EffectConfig):
    """Chorus settings used as a flanger approximation."""

    rate_hz: float = Field(default=0.5, ge=0.01, le=10.0)
    depth: float = Field(default=0.5, ge=0.0, le=1.0)
    feedback: float = Field(default=0.0, ge=0.0, le=1.0)
    mix: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_plugin(self) -> Chorus:
        return Chorus(
            rate_hz=self.rate_hz,
            depth=self.depth,
            feedback=self.feedback,
            mix=self.mix,
        )
```

`EffectConfig` is frozen. Export the new class from `soundlab.effects` and from the package `__init__` when it should be public.

Use the chain with `add`, then either samples or paths:

```python
import numpy as np

from soundlab import load_audio, save_audio
from soundlab.effects import EffectsChain

audio = load_audio("vocals.wav")
chain = EffectsChain().add(FlangerConfig(rate_hz=1.0, depth=0.8))

processed = chain.process_array(audio.samples, audio.sample_rate)
audio.samples = processed
save_audio(audio, "vocals_flanged.wav")

chain.process("vocals.wav", "vocals_flanged.wav")
```

```python
def test_flanger_processing() -> None:
    chain = EffectsChain().add(FlangerConfig(rate_hz=1.0, depth=0.8))
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)
    processed = chain.process_array(audio, 44100)

    assert processed.shape == audio.shape
    assert not np.allclose(processed, audio)
```

---

## Add an analyzer

Define a frozen result model, then a function that returns it. `analyze_audio` currently assembles tempo, key, loudness, and spectral results. Call the new function from there when it should be part of that aggregate.

```python
from pydantic import BaseModel, ConfigDict, Field


class DynamicRangeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    crest_factor_db: float = Field(description="Peak-to-RMS ratio in dB")
    dynamic_range_db: float = Field(description="Loud-section to quiet-section difference")
    compression_ratio: float = Field(description="Estimated compression ratio")
```

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def analyze_dynamics(
    y: NDArray[np.float32],
    sr: int = 44100,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> DynamicRangeResult:
    if y.ndim == 2:
        y = np.mean(y, axis=0)

    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y**2))
    crest_factor_db = 20 * np.log10(peak / (rms + 1e-10))

    n_frames = (len(y) - frame_length) // hop_length + 1
    frame_rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start : start + frame_length]
        frame_rms[i] = np.sqrt(np.mean(frame**2))

    loud = np.percentile(frame_rms, 95)
    quiet = np.percentile(frame_rms, 5)
    dynamic_range_db = 20 * np.log10((loud + 1e-10) / (quiet + 1e-10))

    if crest_factor_db < 6:
        compression_ratio = 8.0
    elif crest_factor_db < 10:
        compression_ratio = 4.0
    elif crest_factor_db < 14:
        compression_ratio = 2.0
    else:
        compression_ratio = 1.0

    return DynamicRangeResult(
        crest_factor_db=float(crest_factor_db),
        dynamic_range_db=float(dynamic_range_db),
        compression_ratio=compression_ratio,
    )
```

The `sr` argument is unused in this crest-factor sketch. Pass the real sample rate through when the measurement depends on it.

Export `DynamicRangeResult` and `analyze_dynamics` from `soundlab.analysis` if they are public. Test synthetic signals, including mono and stereo shapes. `AudioSegment.samples` is channels-first when the file has more than one channel.

---

## Add a voice wrapper

Voice code is optional and lives under `soundlab.voice`. The built-in text-to-speech path is `TTSGenerator.generate(TTSConfig)`. A new wrapper should take an explicit device mode because `get_device` requires one.

```python
from pydantic import BaseModel, ConfigDict, Field


class CustomVoiceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_path: str
    sample_rate: int = Field(default=22050)
    device: str = Field(default="auto")
```

```python
from soundlab.utils.gpu import get_device


class CustomVoiceWrapper:
    def __init__(self, config: CustomVoiceConfig) -> None:
        self.config = config
        self._model = None
        self._device = get_device(config.device)

    def synthesize(self, text: str) -> None:
        raise NotImplementedError(f"Load {self.config.model_path} on {self._device} for {text!r}")
```

`get_device("auto")` returns `"cuda"` when CUDA is available and `"cpu"` otherwise. `"force_gpu"` and `"force_cpu"` are valid modes. `SOUNDLAB_GPU_MODE` overrides the argument when it is set.

Keep model downloads and hosted inference opt-in. Reference secrets by name and do not print them.

---

## Add a separation backend

Implement the `SeparatorBackend` protocol. Its method is `separate(mix_wav, output_dir)` and it returns a map of stem name to path.

```python
from pathlib import Path

from soundlab.pipeline.interfaces import SeparatorBackend


class CustomSeparator:
    def separate(self, mix_wav: Path, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Write stem files under output_dir and return their paths.
        raise NotImplementedError(f"Separate {mix_wav}")


def _check_protocol(backend: SeparatorBackend) -> SeparatorBackend:
    return backend
```

`StemSeparator.separate(audio_path, output_dir)` is the Demucs implementation. It returns a `StemResult`, not the protocol's `dict[str, Path]`. Call that method when you want Demucs. Implement `SeparatorBackend.separate(mix_wav, output_dir)` when a caller needs the path map, and construct that object at the call site.

---

## Tests

- Check config bounds and frozen models.
- Process a synthetic `float32` array and compare shape.
- Mark long or GPU tests with `@pytest.mark.slow` or `@pytest.mark.gpu`.
- Disabled operations should not add validation requirements.

From the repository root, after `uv sync --all-packages`:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check packages/soundlab/src
uv run pytest tests/ -m "not slow and not gpu"
```

There is no `uv.lock` in this repository.

---

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md). Open a focused pull request with the tests that cover the new behavior.
