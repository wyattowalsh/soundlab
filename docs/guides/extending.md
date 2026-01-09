# Extending SoundLab

Guide to adding custom effects, analyzers, and voice models.

---

## Architecture Overview

SoundLab uses a modular architecture with clear extension points:

```
soundlab/
├── analysis/      # Audio analysis (tempo, key, loudness, spectral)
├── effects/       # Audio effects (EQ, dynamics, time-based, creative)
├── separation/    # Stem separation (Demucs backends)
├── transcription/ # Audio-to-MIDI (Basic Pitch)
├── voice/         # Voice generation (TTS, SVC)
├── pipeline/      # Orchestration and QA
└── core/          # Base types and configuration
```

Each module follows consistent patterns:

1. **Config models** — Pydantic models for configuration
2. **Result models** — Pydantic models for outputs
3. **Processing classes** — Stateful processors with lazy loading
4. **Utility functions** — Stateless helpers

---

## Adding Custom Effects

### 1. Define Effect Configuration

Create a Pydantic model for your effect parameters:

```python
# soundlab/effects/models.py
from pydantic import BaseModel, Field

class FlangerConfig(BaseModel):
    """Configuration for flanger effect."""

    rate_hz: float = Field(default=0.5, ge=0.01, le=10.0, description="LFO rate in Hz")
    depth: float   = Field(default=0.5, ge=0.0, le=1.0, description="Effect depth")
    feedback: float = Field(default=0.5, ge=-1.0, le=1.0, description="Feedback amount")
    mix: float     = Field(default=0.5, ge=0.0, le=1.0, description="Wet/dry mix")

    class Config:
        frozen = True
```

### 2. Create Plugin Factory

Map your config to a Pedalboard plugin:

```python
# soundlab/effects/creative.py
from pedalboard import Chorus  # Flanger can be approximated with Chorus

from soundlab.effects.models import FlangerConfig


def create_flanger(config: FlangerConfig):
    """Create a flanger effect plugin.

    Note: Pedalboard doesn't have native Flanger, so we use Chorus
    with short delay times to approximate the effect.
    """
    return Chorus(
        rate_hz=config.rate_hz,
        depth=config.depth,
        mix=config.mix,
        feedback=config.feedback,
    )
```

### 3. Register in Effects Chain

Add your effect to the chain's plugin registry:

```python
# soundlab/effects/chain.py
from soundlab.effects.creative import create_flanger
from soundlab.effects.models import FlangerConfig

# In EffectsChain._create_plugin method:
def _create_plugin(self, config: EffectConfig):
    """Create a Pedalboard plugin from config."""
    match config:
        case FlangerConfig():
            return create_flanger(config)
        # ... other effects ...
        case _:
            raise ValueError(f"Unknown effect config: {type(config)}")
```

### 4. Export from Module

Add to the public API:

```python
# soundlab/effects/__init__.py
from soundlab.effects.models import FlangerConfig

__all__ = [
    # ... existing exports ...
    "FlangerConfig",
]
```

### 5. Add Tests

```python
# tests/unit/test_effects_flanger.py
import numpy as np
import pytest

from soundlab.effects import EffectsChain, FlangerConfig


class TestFlangerEffect:
    def test_flanger_config_defaults(self):
        config = FlangerConfig()
        assert config.rate_hz == 0.5
        assert config.depth == 0.5
        assert config.mix == 0.5

    def test_flanger_config_validation(self):
        with pytest.raises(ValueError):
            FlangerConfig(rate_hz=100.0)  # Exceeds max

    def test_flanger_processing(self):
        config = FlangerConfig(rate_hz=1.0, depth=0.8)
        chain = EffectsChain([config])

        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)
        processed = chain.process(audio, 44100)

        assert processed.shape == audio.shape
        # Flanger should modulate the signal
        assert not np.allclose(processed, audio)
```

---

## Adding Custom Analyzers

### 1. Define Result Model

```python
# soundlab/analysis/models.py
from pydantic import BaseModel, Field


class DynamicRangeResult(BaseModel):
    """Result from dynamic range analysis."""

    crest_factor_db: float = Field(description="Peak-to-RMS ratio in dB")
    dynamic_range_db: float = Field(description="Difference between loud and quiet sections")
    compression_ratio: float = Field(description="Estimated compression ratio")

    class Config:
        frozen = True
```

### 2. Implement Analysis Function

```python
# soundlab/analysis/dynamics.py
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from soundlab.analysis.models import DynamicRangeResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


def analyze_dynamics(
    y: NDArray[np.float32],
    sr: int = 44100,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> DynamicRangeResult:
    """Analyze dynamic range of audio signal.

    Args:
        y: Audio signal (mono or stereo, will be converted to mono)
        sr: Sample rate
        frame_length: Analysis frame length
        hop_length: Hop length between frames

    Returns:
        DynamicRangeResult with crest factor, dynamic range, and compression estimate
    """
    # Convert to mono if stereo
    if y.ndim == 2:
        y = np.mean(y, axis=0)

    # Calculate peak and RMS
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y**2))

    # Crest factor (peak-to-RMS ratio)
    crest_factor_db = 20 * np.log10(peak / (rms + 1e-10))

    # Frame-wise RMS for dynamic range
    n_frames = (len(y) - frame_length) // hop_length + 1
    frame_rms = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_length
        frame = y[start : start + frame_length]
        frame_rms[i] = np.sqrt(np.mean(frame**2))

    # Dynamic range (difference between 95th and 5th percentile in dB)
    loud = np.percentile(frame_rms, 95)
    quiet = np.percentile(frame_rms, 5)
    dynamic_range_db = 20 * np.log10((loud + 1e-10) / (quiet + 1e-10))

    # Estimate compression ratio based on crest factor
    # Lower crest factor suggests more compression
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

### 3. Integrate with analyze_audio

```python
# soundlab/analysis/__init__.py
from soundlab.analysis.dynamics import analyze_dynamics
from soundlab.analysis.models import DynamicRangeResult

# In analyze_audio function, add:
# dynamics = analyze_dynamics(y, sr)
# result.dynamics = dynamics

__all__ = [
    # ... existing exports ...
    "DynamicRangeResult",
    "analyze_dynamics",
]
```

### 4. Add Tests

```python
# tests/unit/test_analysis_dynamics.py
import numpy as np
import pytest

from soundlab.analysis.dynamics import analyze_dynamics


class TestDynamicRangeAnalysis:
    def test_sine_wave_crest_factor(self):
        """Pure sine wave has ~3dB crest factor."""
        sr = 44100
        t = np.linspace(0, 1, sr, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)

        result = analyze_dynamics(sine, sr)

        # Sine wave crest factor is sqrt(2) ≈ 3.01 dB
        assert 2.5 < result.crest_factor_db < 3.5

    def test_compressed_signal(self):
        """Heavily limited signal has low crest factor."""
        sr = 44100
        t = np.linspace(0, 1, sr, dtype=np.float32)
        # Square-ish wave (heavily compressed)
        square = np.sign(np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        result = analyze_dynamics(square, sr)

        # Square wave has ~0dB crest factor
        assert result.crest_factor_db < 1.0
        assert result.compression_ratio >= 4.0

    def test_stereo_input(self):
        """Stereo input is converted to mono."""
        sr = 44100
        mono = np.random.randn(sr).astype(np.float32)
        stereo = np.stack([mono, mono])

        result_mono = analyze_dynamics(mono, sr)
        result_stereo = analyze_dynamics(stereo, sr)

        assert abs(result_mono.crest_factor_db - result_stereo.crest_factor_db) < 0.1
```

---

## Adding Voice Models

### 1. Define Configuration

```python
# soundlab/voice/models.py
from pathlib import Path
from pydantic import BaseModel, Field


class CustomVoiceConfig(BaseModel):
    """Configuration for custom voice model."""

    model_path: Path = Field(description="Path to model checkpoint")
    speaker_embedding: Path | None = Field(default=None, description="Speaker embedding file")
    sample_rate: int = Field(default=22050, description="Output sample rate")
    device: str = Field(default="auto", description="Processing device")

    class Config:
        frozen = True


class CustomVoiceResult(BaseModel):
    """Result from custom voice synthesis."""

    audio: list[float] = Field(description="Generated audio samples")
    sample_rate: int = Field(description="Audio sample rate")
    duration: float = Field(description="Audio duration in seconds")
    model_name: str = Field(description="Model used for synthesis")
```

### 2. Implement Voice Wrapper

```python
# soundlab/voice/custom.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from soundlab.utils.gpu import get_device
from soundlab.utils.retry import gpu_retry
from soundlab.voice.models import CustomVoiceConfig, CustomVoiceResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CustomVoiceWrapper:
    """Wrapper for custom voice model."""

    def __init__(self, config: CustomVoiceConfig | None = None) -> None:
        self.config = config or CustomVoiceConfig()
        self._model = None
        self._device = None

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is not None:
            return

        self._device = get_device() if self.config.device == "auto" else self.config.device

        # Load your custom model here
        # self._model = YourModel.load(self.config.model_path)
        # self._model.to(self._device)
        raise NotImplementedError("Implement model loading for your voice model")

    @gpu_retry(max_attempts=3)
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
    ) -> CustomVoiceResult:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize
            speaker_id: Speaker ID for multi-speaker models

        Returns:
            CustomVoiceResult with generated audio
        """
        self._load_model()

        # Implement synthesis logic
        # audio = self._model.synthesize(text, speaker_id)

        # Placeholder
        duration = len(text) * 0.1  # Rough estimate
        n_samples = int(duration * self.config.sample_rate)
        audio = np.zeros(n_samples, dtype=np.float32)

        return CustomVoiceResult(
            audio=audio.tolist(),
            sample_rate=self.config.sample_rate,
            duration=duration,
            model_name=str(self.config.model_path),
        )
```

### 3. Export from Module

```python
# soundlab/voice/__init__.py
from soundlab.voice.custom import CustomVoiceWrapper
from soundlab.voice.models import CustomVoiceConfig, CustomVoiceResult

__all__ = [
    # ... existing exports ...
    "CustomVoiceConfig",
    "CustomVoiceResult",
    "CustomVoiceWrapper",
]
```

---

## Adding Pipeline Backends

### 1. Implement Protocol

```python
# soundlab/pipeline/backends/custom_separator.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from soundlab.pipeline.interfaces import SeparatorBackend

if TYPE_CHECKING:
    from soundlab.separation.models import SeparationConfig, StemResult


class CustomSeparator(SeparatorBackend):
    """Custom stem separation backend."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model = None

    def separate(
        self,
        audio_path: Path,
        output_dir: Path,
        config: SeparationConfig,
    ) -> StemResult:
        """Separate audio into stems.

        Args:
            audio_path: Input audio file
            output_dir: Directory for output stems
            config: Separation configuration

        Returns:
            StemResult with paths to separated stems
        """
        # Implement your separation logic
        raise NotImplementedError("Implement separation for your model")

    def supports_model(self, model_name: str) -> bool:
        """Check if this backend supports the given model."""
        return model_name.startswith("custom_")
```

### 2. Register Backend

```python
# soundlab/pipeline/candidates.py
from soundlab.pipeline.backends.custom_separator import CustomSeparator

# In build_candidate_plans or pipeline initialization:
def get_separator_backend(model_name: str) -> SeparatorBackend:
    """Get appropriate backend for model."""
    if model_name.startswith("custom_"):
        return CustomSeparator(model_path=Path(model_name))
    else:
        from soundlab.separation import StemSeparator
        return StemSeparator()
```

---

## Testing Guidelines

### Unit Tests

- Test configuration validation (bounds, types)
- Test processing with synthetic signals
- Mock external dependencies (models, GPU)

### Integration Tests

- Test end-to-end workflows
- Use `@pytest.mark.slow` for long-running tests
- Use fixtures for shared test data

### Example Test Structure

```python
# tests/unit/test_your_module.py
import numpy as np
import pytest


class TestYourConfig:
    """Tests for configuration model."""

    def test_defaults(self):
        """Test default values."""
        pass

    def test_validation(self):
        """Test parameter validation."""
        pass


class TestYourProcessor:
    """Tests for processor class."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        pass

    def test_basic_processing(self, processor):
        """Test basic processing."""
        pass

    @pytest.mark.slow
    def test_full_pipeline(self, processor):
        """Test full pipeline (slow)."""
        pass
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your extension with tests
4. Run validation:
   ```bash
   uv run ruff format .
   uv run ruff check .
   uv run pytest tests/ -v
   ```
5. Submit a pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for full guidelines.
