# SoundLab PRD

**Repository:** [github.com/wyattowalsh/soundlab](https://github.com/wyattowalsh/soundlab)  
**Version:** 2.0  
**Date:** 2026-01-06  
**Status:** Final Specification

---

## 1. Executive Summary

SoundLab is a production-ready music processing platform consisting of:

1. **`soundlab` Python package** — A modular, well-tested library providing stem separation, audio-to-MIDI transcription, effects processing, audio analysis, and voice generation capabilities
2. **Google Colab notebook** — A thin orchestration layer that imports `soundlab` and provides interactive UI via Colab Forms + Gradio

This architecture ensures the notebook remains lean (~500 lines) while all business logic, error handling, and complex processing lives in the properly-tested `soundlab` package.

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Separation of concerns** | Notebook = UI/orchestration; Package = logic/processing |
| **Testability** | All processing logic in package with pytest coverage |
| **Maintainability** | Single source of truth for algorithms |
| **Extensibility** | Plugin architecture for new processors |
| **Type safety** | Full type hints with Pydantic models |
| **Reproducibility** | Locked dependencies via uv.lock |

---

## 2. Repository Structure

```
soundlab/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Test, lint, type-check on PR
│       ├── release.yml               # PyPI publish on tag
│       └── colab-test.yml            # Periodic Colab compatibility check
├── .python-version                   # 3.12
├── pyproject.toml                    # Workspace root configuration
├── uv.lock                           # Cross-platform lockfile (committed)
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
│
├── packages/
│   └── soundlab/                     # Main library package
│       ├── pyproject.toml            # Package-specific config
│       ├── py.typed                  # PEP 561 marker
│       └── src/
│           └── soundlab/
│               ├── __init__.py       # Public API exports
│               ├── _version.py       # Dynamic versioning
│               ├── py.typed
│               │
│               ├── core/             # Core abstractions
│               │   ├── __init__.py
│               │   ├── audio.py      # AudioSegment, AudioMetadata models
│               │   ├── config.py     # Global configuration management
│               │   ├── exceptions.py # Custom exception hierarchy
│               │   └── types.py      # Type aliases, protocols
│               │
│               ├── separation/       # Stem separation module
│               │   ├── __init__.py
│               │   ├── demucs.py     # Demucs wrapper
│               │   ├── models.py     # SeparationConfig, StemResult
│               │   └── utils.py      # Segment processing, overlap-add
│               │
│               ├── transcription/    # Audio-to-MIDI module
│               │   ├── __init__.py
│               │   ├── basic_pitch.py
│               │   ├── models.py     # TranscriptionConfig, MIDIResult
│               │   └── visualization.py  # Piano roll rendering
│               │
│               ├── effects/          # Audio effects module
│               │   ├── __init__.py
│               │   ├── chain.py      # EffectsChain orchestration
│               │   ├── dynamics.py   # Compressor, Limiter, Gate
│               │   ├── eq.py         # Filters, EQ
│               │   ├── time_based.py # Reverb, Delay, Chorus
│               │   ├── creative.py   # Distortion, Phaser, etc.
│               │   └── models.py     # Effect configs as Pydantic models
│               │
│               ├── analysis/         # Audio analysis module
│               │   ├── __init__.py
│               │   ├── tempo.py      # BPM detection
│               │   ├── key.py        # Key/scale detection (K-K algorithm)
│               │   ├── loudness.py   # LUFS, dynamic range
│               │   ├── spectral.py   # Spectral features
│               │   ├── onsets.py     # Transient detection
│               │   └── models.py     # AnalysisResult model
│               │
│               ├── pipeline/         # Max-quality Colab orchestration + QA
│               │   ├── __init__.py
│               │   ├── models.py     # PipelineConfig, QAConfig, CandidatePlan
│               │   ├── interfaces.py # Backend protocols (separator/transcriber/etc.)
│               │   ├── candidates.py # Strategy generation + excerpt plans
│               │   ├── qa.py         # Separation + MIDI QA metrics/scoring
│               │   ├── postprocess.py# Stem/MIDI post-processing utilities
│               │   └── cache.py      # Run IDs, cache paths, checkpoints
│               │
│               ├── voice/            # Voice generation module
│               │   ├── __init__.py
│               │   ├── tts.py        # XTTS-v2 wrapper
│               │   ├── svc.py        # RVC wrapper
│               │   └── models.py     # VoiceConfig, VoiceResult
│               │
│               ├── io/               # I/O utilities
│               │   ├── __init__.py
│               │   ├── audio_io.py   # Load/save with format detection
│               │   ├── midi_io.py    # MIDI read/write
│               │   └── export.py     # Batch export, zip creation
│               │
│               └── utils/            # Shared utilities
│                   ├── __init__.py
│                   ├── gpu.py        # GPU detection, memory management
│                   ├── logging.py    # Loguru configuration
│                   ├── retry.py      # Tenacity decorators
│                   └── progress.py   # Progress callbacks
│
├── notebooks/
│   ├── soundlab_studio.ipynb         # Main Colab notebook
│   └── examples/
│       ├── stem_separation.ipynb
│       ├── midi_transcription.ipynb
│       └── voice_conversion.ipynb
│
├── tests/
│   ├── conftest.py                   # Fixtures, test audio files
│   ├── unit/
│   │   ├── test_separation.py
│   │   ├── test_transcription.py
│   │   ├── test_effects.py
│   │   ├── test_analysis.py
│   │   └── test_voice.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_colab_compat.py
│   └── fixtures/
│       └── audio/                    # Test audio files (< 10s each)
│
├── docs/
│   ├── api/                          # Auto-generated API docs
│   ├── guides/
│   │   ├── quickstart.md
│   │   ├── colab-usage.md
│   │   └── extending.md
│   └── mkdocs.yml
│
└── scripts/
    ├── download_models.py            # Pre-download large models
    └── benchmark.py                  # Performance benchmarking
```

---

## 3. uv Workspace Configuration

### Root `pyproject.toml`

```toml
[project]
name = "soundlab-workspace"
version = "0.1.0"
description = "SoundLab workspace root"
requires-python = ">=3.12"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "Wyatt Walsh" }]

# Workspace root has no direct dependencies
# All deps are in the soundlab package
dependencies = []

[tool.uv]
dev-dependencies = [
    # Development tools (shared across workspace)
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.23",
    "hypothesis>=6.100",
    "ruff>=0.4",
    "ty>=0.3",
    "pre-commit>=3.7",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.25",
]

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
soundlab = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# === Tool Configuration ===

[tool.ruff]
target-version = "py312"
line-length = 100
src = ["packages/soundlab/src"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "PTH",    # flake8-use-pathlib
    "RUF",    # Ruff-specific rules
]
ignore = ["E501"]  # Line length handled by formatter

[tool.ruff.lint.isort]
known-first-party = ["soundlab"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
    "-v",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests requiring GPU",
    "integration: marks integration tests",
]

[tool.coverage.run]
source = ["packages/soundlab/src"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

### Package `packages/soundlab/pyproject.toml`

```toml
[project]
name = "soundlab"
version = "0.1.0"
description = "Production-ready music processing library for stem separation, transcription, effects, and voice generation"
requires-python = ">=3.12"
readme = "../../README.md"
license = { text = "MIT" }
authors = [{ name = "Wyatt Walsh" }]
keywords = [
    "audio",
    "music",
    "stem-separation",
    "demucs",
    "midi",
    "transcription",
    "effects",
    "voice-conversion",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Sound/Audio",
    "Topic :: Multimedia :: Sound/Audio :: Analysis",
    "Topic :: Multimedia :: Sound/Audio :: Conversion",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]

dependencies = [
    # Core audio processing
    "demucs>=4.0.1,<5",
    "basic-pitch>=0.3,<0.4",
    "pedalboard>=0.9,<1",
    "librosa>=0.10,<0.11",
    
    # Audio I/O
    "soundfile>=0.12,<0.13",
    "pydub>=0.25,<0.26",
    
    # Data models & validation
    "pydantic>=2.7,<3",
    "numpy>=1.26,<3",
    
    # Infrastructure
    "tenacity>=8.3,<9",
    "loguru>=0.7,<0.8",
    "tqdm>=4.66,<5",
    "httpx>=0.27,<0.28",
    
    # ML framework (torch is a demucs dependency, but pin for GPU)
    "torch>=2.2",
]

[project.optional-dependencies]
# Voice generation (heavy, optional)
voice = [
    "coqui-tts>=0.22",      # TTS with XTTS-v2
    # RVC requires manual setup due to complexity
]

# Gradio interface for notebooks
notebook = [
    "gradio>=4.26,<5",
]

# Full installation
all = [
    "soundlab[voice,notebook]",
]

[project.urls]
Homepage = "https://github.com/wyattowalsh/soundlab"
Documentation = "https://wyattowalsh.github.io/soundlab"
Repository = "https://github.com/wyattowalsh/soundlab"
Changelog = "https://github.com/wyattowalsh/soundlab/blob/main/CHANGELOG.md"

[project.scripts]
soundlab = "soundlab.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.sdist]
include = ["/src"]

[tool.hatch.build.targets.wheel]
packages = ["src/soundlab"]
```

---

## 4. Core Package Architecture

### 4.1 Exception Hierarchy

```python
# packages/soundlab/src/soundlab/core/exceptions.py
"""Custom exception hierarchy for SoundLab."""

from __future__ import annotations


class SoundLabError(Exception):
    """Base exception for all SoundLab errors."""
    pass


class AudioLoadError(SoundLabError):
    """Failed to load audio file."""
    pass


class AudioFormatError(SoundLabError):
    """Unsupported or invalid audio format."""
    pass


class ModelNotFoundError(SoundLabError):
    """Required model not available."""
    pass


class GPUMemoryError(SoundLabError):
    """Insufficient GPU memory for operation."""
    pass


class ProcessingError(SoundLabError):
    """Error during audio processing."""
    pass


class ConfigurationError(SoundLabError):
    """Invalid configuration."""
    pass


class VoiceConversionError(SoundLabError):
    """Error in voice conversion pipeline."""
    pass
```

### 4.2 Core Data Models

```python
# packages/soundlab/src/soundlab/core/audio.py
"""Core audio data models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AudioFormat(StrEnum):
    """Supported audio formats."""
    
    WAV  = "wav"
    MP3  = "mp3"
    FLAC = "flac"
    OGG  = "ogg"
    AIFF = "aiff"
    M4A  = "m4a"


class SampleRate(StrEnum):
    """Common sample rates."""
    
    SR_22050 = "22050"
    SR_44100 = "44100"
    SR_48000 = "48000"
    SR_96000 = "96000"
    
    @property
    def hz(self) -> int:
        return int(self.value)


class BitDepth(StrEnum):
    """Audio bit depths."""
    
    INT16 = "16"
    INT24 = "24"
    FLOAT32 = "32"


class AudioMetadata(BaseModel):
    """Metadata for an audio file."""
    
    model_config = ConfigDict(frozen=True)
    
    duration_seconds: Annotated[float, Field(ge=0)]
    sample_rate: int
    channels: Annotated[int, Field(ge=1, le=8)]
    bit_depth: BitDepth | None = None
    format: AudioFormat | None = None
    
    @property
    def duration_str(self) -> str:
        """Human-readable duration (MM:SS.ms)."""
        mins, secs = divmod(self.duration_seconds, 60)
        return f"{int(mins):02d}:{secs:05.2f}"
    
    @property
    def is_stereo(self) -> bool:
        return self.channels == 2
    
    @property
    def is_mono(self) -> bool:
        return self.channels == 1


class AudioSegment(BaseModel):
    """In-memory audio representation."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    samples: NDArray[np.float32]
    sample_rate: int
    source_path: Path | None = None
    metadata: AudioMetadata | None = None
    
    @field_validator("samples", mode="before")
    @classmethod
    def ensure_float32(cls, v: NDArray) -> NDArray[np.float32]:
        if v.dtype != np.float32:
            return v.astype(np.float32)
        return v
    
    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate
    
    @property
    def channels(self) -> int:
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[0]
    
    def to_mono(self) -> "AudioSegment":
        """Convert to mono by averaging channels."""
        if self.channels == 1:
            return self
        mono_samples = np.mean(self.samples, axis=0)
        return AudioSegment(
            samples=mono_samples,
            sample_rate=self.sample_rate,
            source_path=self.source_path,
        )
```

### 4.3 Stem Separation Module

```python
# packages/soundlab/src/soundlab/separation/models.py
"""Stem separation configuration and result models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class DemucsModel(StrEnum):
    """Available Demucs models."""
    
    HTDEMUCS      = "htdemucs"       # Hybrid Transformer, 4 stems
    HTDEMUCS_FT   = "htdemucs_ft"    # Fine-tuned, best quality
    HTDEMUCS_6S   = "htdemucs_6s"    # 6 stems (piano unreliable)
    MDX_EXTRA     = "mdx_extra"       # MDX architecture
    MDX_EXTRA_Q   = "mdx_extra_q"     # Quantized MDX
    
    @property
    def stem_count(self) -> int:
        return 6 if self == DemucsModel.HTDEMUCS_6S else 4
    
    @property
    def stems(self) -> list[str]:
        base = ["vocals", "drums", "bass", "other"]
        if self == DemucsModel.HTDEMUCS_6S:
            return base + ["piano", "guitar"]
        return base


class SeparationConfig(BaseModel):
    """Configuration for stem separation."""
    
    model_config = ConfigDict(frozen=True)
    
    model: DemucsModel = DemucsModel.HTDEMUCS_FT
    
    # Processing parameters
    segment_length: Annotated[float, Field(ge=1.0, le=30.0)] = 7.8
    overlap: Annotated[float, Field(ge=0.1, le=0.9)] = 0.25
    shifts: Annotated[int, Field(ge=0, le=5)] = 1
    
    # Output options
    two_stems: str | None = None  # "vocals", "drums", "bass", "other"
    float32: bool = False
    int24: bool = True
    mp3_bitrate: Annotated[int, Field(ge=128, le=320)] = 320
    
    # Resource management
    device: str = "auto"  # "auto", "cuda", "cpu"
    split: bool = True    # Enable segment-based processing for long audio


class StemResult(BaseModel):
    """Result from stem separation."""
    
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    
    stems: dict[str, Path]           # stem_name -> file_path
    source_path: Path
    config: SeparationConfig
    processing_time_seconds: float
    
    @property
    def vocals(self) -> Path | None:
        return self.stems.get("vocals")
    
    @property
    def instrumental(self) -> Path | None:
        """Combined non-vocal stems (computed on demand)."""
        # Implementation would mix drums + bass + other
        return None
```

```python
# packages/soundlab/src/soundlab/separation/demucs.py
"""Demucs stem separation wrapper."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from soundlab.core.exceptions import GPUMemoryError, ModelNotFoundError, ProcessingError
from soundlab.separation.models import DemucsModel, SeparationConfig, StemResult
from soundlab.utils.gpu import get_device, get_free_vram_gb
from soundlab.utils.progress import ProgressCallback

if TYPE_CHECKING:
    from soundlab.core.audio import AudioSegment


class StemSeparator:
    """High-level interface for stem separation using Demucs."""
    
    def __init__(self, config: SeparationConfig | None = None) -> None:
        """
        Initialize the stem separator.
        
        Parameters
        ----------
        config
            Separation configuration. Uses defaults if not provided.
        """
        self.config = config or SeparationConfig()
        self._model = None
        self._device = None
    
    def _load_model(self) -> None:
        """Lazy-load the Demucs model."""
        if self._model is not None:
            return
        
        from demucs.pretrained import get_model
        
        self._device = get_device(self.config.device)
        logger.info(f"Loading Demucs model: {self.config.model} on {self._device}")
        
        try:
            self._model = get_model(self.config.model.value)
            self._model.to(self._device)
            self._model.eval()
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load {self.config.model}: {e}") from e
    
    def _check_memory(self, duration_seconds: float) -> None:
        """Verify sufficient GPU memory for processing."""
        if self._device == "cpu":
            return
        
        # Rough estimate: ~2GB base + 0.5GB per minute
        estimated_gb = 2.0 + (duration_seconds / 60) * 0.5
        free_gb = get_free_vram_gb()
        
        if free_gb < estimated_gb:
            logger.warning(
                f"Low VRAM: {free_gb:.1f}GB free, ~{estimated_gb:.1f}GB needed. "
                "Consider enabling segment processing or using CPU."
            )
            if free_gb < 2.0:
                raise GPUMemoryError(
                    f"Insufficient VRAM ({free_gb:.1f}GB). "
                    "Use config.split=True or device='cpu'."
                )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def separate(
        self,
        audio_path: Path | str,
        output_dir: Path | str,
        progress_callback: ProgressCallback | None = None,
    ) -> StemResult:
        """
        Separate audio into stems.
        
        Parameters
        ----------
        audio_path
            Path to input audio file.
        output_dir
            Directory to save separated stems.
        progress_callback
            Optional callback for progress updates.
        
        Returns
        -------
        StemResult
            Paths to separated stem files and metadata.
        
        Raises
        ------
        AudioLoadError
            If the input file cannot be loaded.
        GPUMemoryError
            If insufficient GPU memory is available.
        ProcessingError
            If separation fails.
        
        Examples
        --------
        >>> separator = StemSeparator()
        >>> result = separator.separate("song.mp3", "output/")
        >>> print(result.stems)
        {'vocals': Path('output/vocals.wav'), 'drums': Path('output/drums.wav'), ...}
        """
        from demucs.apply import apply_model
        from demucs.audio import AudioFile
        
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_model()
        
        start_time = time.perf_counter()
        
        # Load audio
        logger.info(f"Loading audio: {audio_path}")
        wav = AudioFile(audio_path).read(
            streams=0,
            samplerate=self._model.samplerate,
            channels=self._model.audio_channels,
        )
        wav = wav.to(self._device)
        
        self._check_memory(wav.shape[-1] / self._model.samplerate)
        
        # Apply model
        logger.info(f"Separating with {self.config.model}...")
        
        with torch.no_grad():
            sources = apply_model(
                self._model,
                wav[None],
                segment=self.config.segment_length if self.config.split else None,
                overlap=self.config.overlap,
                shifts=self.config.shifts,
                progress=progress_callback is not None,
            )[0]
        
        # Save stems
        stems = {}
        stem_names = self.config.model.stems
        
        if self.config.two_stems:
            stem_names = [self.config.two_stems, "no_" + self.config.two_stems]
        
        for i, stem_name in enumerate(stem_names):
            stem_path = output_dir / f"{stem_name}.wav"
            self._save_stem(sources[i], stem_path)
            stems[stem_name] = stem_path
            logger.debug(f"Saved: {stem_path}")
        
        processing_time = time.perf_counter() - start_time
        logger.info(f"Separation complete in {processing_time:.1f}s")
        
        return StemResult(
            stems=stems,
            source_path=audio_path,
            config=self.config,
            processing_time_seconds=processing_time,
        )
    
    def _save_stem(self, tensor: torch.Tensor, path: Path) -> None:
        """Save a stem tensor to file."""
        import soundfile as sf
        
        audio = tensor.cpu().numpy()
        if audio.ndim == 2:
            audio = audio.T  # (channels, samples) -> (samples, channels)
        
        subtype = "PCM_24" if self.config.int24 else "FLOAT" if self.config.float32 else "PCM_16"
        sf.write(path, audio, self._model.samplerate, subtype=subtype)
```

### 4.4 Analysis Module (Key Detection)

```python
# packages/soundlab/src/soundlab/analysis/key.py
"""Musical key detection using Krumhansl-Schmuckler algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import librosa
import numpy as np
from numpy.typing import NDArray


class MusicalKey(StrEnum):
    """Musical key names."""
    
    C  = "C"
    Cs = "C#"
    D  = "D"
    Ds = "D#"
    E  = "E"
    F  = "F"
    Fs = "F#"
    G  = "G"
    Gs = "G#"
    A  = "A"
    As = "A#"
    B  = "B"


class Mode(StrEnum):
    """Musical mode."""
    
    MAJOR = "major"
    MINOR = "minor"


@dataclass(frozen=True)
class KeyDetectionResult:
    """Result of key detection analysis."""
    
    key: MusicalKey
    mode: Mode
    confidence: float
    all_correlations: dict[str, float]
    
    @property
    def name(self) -> str:
        """Full key name (e.g., 'A minor')."""
        return f"{self.key.value} {self.mode.value}"
    
    @property
    def camelot(self) -> str:
        """Camelot notation for DJ mixing."""
        camelot_map = {
            ("C", "major"): "8B", ("A", "minor"): "8A",
            ("G", "major"): "9B", ("E", "minor"): "9A",
            ("D", "major"): "10B", ("B", "minor"): "10A",
            ("A", "major"): "11B", ("F#", "minor"): "11A",
            ("E", "major"): "12B", ("C#", "minor"): "12A",
            ("B", "major"): "1B", ("G#", "minor"): "1A",
            ("F#", "major"): "2B", ("D#", "minor"): "2A",
            ("C#", "major"): "3B", ("A#", "minor"): "3A",
            ("G#", "major"): "4B", ("F", "minor"): "4A",
            ("D#", "major"): "5B", ("C", "minor"): "5A",
            ("A#", "major"): "6B", ("G", "minor"): "6A",
            ("F", "major"): "7B", ("D", "minor"): "7A",
        }
        return camelot_map.get((self.key.value, self.mode.value), "?")


# Krumhansl-Schmuckler key profiles (normalized)
_MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
])
_MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])


def detect_key(
    y: NDArray[np.float32],
    sr: int,
    *,
    hop_length: int = 512,
) -> KeyDetectionResult:
    """
    Detect the musical key using the Krumhansl-Schmuckler algorithm.
    
    Parameters
    ----------
    y
        Audio time series (mono).
    sr
        Sample rate.
    hop_length
        Hop length for chroma computation.
    
    Returns
    -------
    KeyDetectionResult
        Detected key, mode, and confidence score.
    
    Examples
    --------
    >>> y, sr = librosa.load("song.mp3", sr=22050, mono=True)
    >>> result = detect_key(y, sr)
    >>> print(result.name)
    'A minor'
    >>> print(result.camelot)
    '8A'
    """
    # Compute chroma features using CQT for better frequency resolution
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    # Average across time to get pitch class distribution
    chroma_avg = np.mean(chroma, axis=1)
    
    # Normalize
    chroma_avg = chroma_avg / (np.linalg.norm(chroma_avg) + 1e-8)
    major_norm = _MAJOR_PROFILE / np.linalg.norm(_MAJOR_PROFILE)
    minor_norm = _MINOR_PROFILE / np.linalg.norm(_MINOR_PROFILE)
    
    keys = list(MusicalKey)
    all_correlations = {}
    best_corr = -1.0
    best_key = MusicalKey.C
    best_mode = Mode.MAJOR
    
    for i, key in enumerate(keys):
        # Roll chroma to align with current key
        rolled = np.roll(chroma_avg, -i)
        
        # Correlate with both profiles
        maj_corr = float(np.corrcoef(rolled, major_norm)[0, 1])
        min_corr = float(np.corrcoef(rolled, minor_norm)[0, 1])
        
        all_correlations[f"{key.value} major"] = maj_corr
        all_correlations[f"{key.value} minor"] = min_corr
        
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = key
            best_mode = Mode.MAJOR
        
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = key
            best_mode = Mode.MINOR
    
    # Convert correlation to confidence (0-1 scale)
    confidence = (best_corr + 1) / 2  # Map [-1, 1] to [0, 1]
    
    return KeyDetectionResult(
        key=best_key,
        mode=best_mode,
        confidence=confidence,
        all_correlations=all_correlations,
    )
```

### 4.5 Effects Chain

```python
# packages/soundlab/src/soundlab/effects/chain.py
"""Effects chain orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pedalboard import Pedalboard
from pydantic import BaseModel

from soundlab.effects.models import EffectConfig


class AudioEffect(Protocol):
    """Protocol for audio effects."""
    
    def process(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        """Apply the effect to audio samples."""
        ...


class EffectsChain:
    """
    Chainable audio effects processor.
    
    Examples
    --------
    >>> from soundlab.effects import EffectsChain
    >>> from soundlab.effects.dynamics import CompressorConfig
    >>> from soundlab.effects.time_based import ReverbConfig
    >>> 
    >>> chain = EffectsChain()
    >>> chain.add(CompressorConfig(threshold_db=-20, ratio=4.0))
    >>> chain.add(ReverbConfig(room_size=0.5, wet=0.3))
    >>> 
    >>> result = chain.process("input.wav", "output.wav")
    """
    
    def __init__(self) -> None:
        self._effects: list[EffectConfig] = []
        self._board: Pedalboard | None = None
    
    def add(self, effect: EffectConfig) -> "EffectsChain":
        """Add an effect to the chain (fluent API)."""
        self._effects.append(effect)
        self._board = None  # Invalidate cached board
        return self
    
    def clear(self) -> "EffectsChain":
        """Clear all effects from the chain."""
        self._effects.clear()
        self._board = None
        return self
    
    @property
    def effects(self) -> Sequence[EffectConfig]:
        """Current effects in the chain."""
        return tuple(self._effects)
    
    def _build_board(self) -> Pedalboard:
        """Build the Pedalboard from effect configs."""
        if self._board is not None:
            return self._board
        
        plugins = [effect.to_plugin() for effect in self._effects]
        self._board = Pedalboard(plugins)
        return self._board
    
    def process_array(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> NDArray[np.float32]:
        """
        Process audio samples through the effects chain.
        
        Parameters
        ----------
        audio
            Audio samples as numpy array.
        sample_rate
            Sample rate of the audio.
        
        Returns
        -------
        NDArray[np.float32]
            Processed audio samples.
        """
        if not self._effects:
            return audio
        
        board = self._build_board()
        return board(audio, sample_rate)
    
    def process(
        self,
        input_path: Path | str,
        output_path: Path | str,
    ) -> Path:
        """
        Process an audio file through the effects chain.
        
        Parameters
        ----------
        input_path
            Path to input audio file.
        output_path
            Path to save processed audio.
        
        Returns
        -------
        Path
            Path to the processed audio file.
        """
        import soundfile as sf
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Load audio
        audio, sr = sf.read(input_path, dtype="float32")
        
        # Ensure correct shape (samples, channels) -> (channels, samples) for pedalboard
        if audio.ndim == 2:
            audio = audio.T
        
        # Process
        processed = self.process_array(audio, sr)
        
        # Save
        if processed.ndim == 2:
            processed = processed.T
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, processed, sr)
        
        return output_path
```

### 4.6 Max-Quality Pipeline + QA (Colab Orchestration)

To keep the notebook thin while delivering **best-possible stems + MIDI**, the package includes a
pipeline layer that coordinates staged separation, multi-candidate selection, QA scoring, and
alignment-safe post-processing. The notebook only wires UI inputs to these abstractions.

**Core interfaces (protocols)**

```python
# packages/soundlab/src/soundlab/pipeline/interfaces.py
class SeparatorBackend(Protocol):
    def separate(self, mix_wav: Path, output_dir: Path) -> dict[str, Path]: ...

class TranscriberBackend(Protocol):
    def transcribe(self, stem_wav: Path, output_dir: Path) -> Path: ...

class StemPostProcessor(Protocol):
    def process(self, stems: dict[str, Path], output_dir: Path) -> dict[str, Path]: ...

class MidiPostProcessor(Protocol):
    def process(self, midi_path: Path, output_dir: Path) -> Path: ...

class QAEvaluator(Protocol):
    def score(self, mix_wav: Path, stems: dict[str, Path]) -> dict[str, float]: ...
```

**Pipeline guarantees**

- **Staged separation default**: vocals-first extraction -> 4-stem on instrumental -> optional
  refinement pass for problematic stems.
- **Excerpt-based candidate selection**: try 2-4 separation strategies on a 30s excerpt, score,
  then run the best strategy on the full track.
- **Stem-specific transcription routing**: drums/vocals/bass/other choose different backends and
  parameter presets; supports two-candidate transcription and confidence-weighted merge.
- **Resumable checkpoints**: `run_id = hash(audio) + hash(config)`, with `runs/`, `cache/`,
  `artifacts/`, `reports/` and stage-level skip logic.
- **Alignment-safe post-processing**: denoise/filter/trim only on AMT copies; originals remain
  sample-aligned for reconstruction and residual QA.

**QA heuristics (no ground truth required)**

- **Separation QA**: reconstruction error (mix vs sum of stems), residual spectral flatness,
  clipping/intersample peak checks, stereo coherence, and stem leakage proxies (e.g., vocals
  voiced-ness vs broadband energy; drums onset density vs tonal energy).
- **Transcription QA**: notes/sec, max polyphony, pitch range sanity per stem, onset alignment vs
  audio onsets, confidence thresholds, and empty-MIDI detection with fallback.

**Fallback matrix**

- If a separator fails or scores below threshold -> switch to next strategy or safer chunk size.
- If transcription fails or QA is bad -> fallback per-stem to simpler models/presets.

---

## 5. Notebook Architecture

### 5.1 Design Philosophy

The notebook serves as a **thin orchestration layer** that:

1. **Imports** the `soundlab` package for all processing logic
2. **Provides UI** via Colab Forms (configuration) + Gradio (audio I/O)
3. **Orchestrates** the pipeline by connecting user inputs to package functions
4. **Displays** results, progress, and visualizations

The notebook should **never** contain:
- Complex processing algorithms (use `soundlab` package)
- Retry logic or error handling details (use `soundlab.utils.retry`)
- Model loading code (use `soundlab` module functions)
- File format handling (use `soundlab.io`)

### 5.2 Notebook Structure

```
notebooks/soundlab_studio.ipynb
├── Cell 1: Header & Instructions (Markdown)
├── Cell 2: Environment Setup (Code + Form)
├── Cell 3: Package Installation (Code)
├── Cell 4: File Upload Interface (Gradio)
├── Cell 5: Stem Separation Config (Form) + Execution (Code)
├── Cell 6: MIDI Transcription Config (Form) + Execution (Code)
├── Cell 7: Audio Analysis Config (Form) + Execution (Code)
├── Cell 8: Effects Chain Config (Form) + Execution (Code)
├── Cell 9: Voice Generation Config (Form) + Execution (Code)
├── Cell 10: Export & Download Interface (Gradio)
└── Cell 11: Cleanup (Code)
```

### 5.2b Max-Quality Notebook Structure (Expanded)

The max-quality notebook follows the same core flow but adds explicit cells for candidate
selection, QA, post-processing, and resumability:

```
notebooks/soundlab_studio.ipynb (max-quality mode)
├── Cell 0: Resume From Checkpoint (Code)
├── Cell 1: Header & Run Overview (Markdown)
├── Cell 2: Runtime Introspection + Determinism Flags (Code + Form)
├── Cell 3: Package Installation (Code)
├── Cell 4: Drive Mount + Cache Roots (Code + Form)
├── Cell 5: Upload + Canonical Decode + Hash/Metadata (Code + Form)
├── Cell 6: Excerpt Selection (Code + Form)
├── Cell 7: Separation Candidate Plan (Code + Form)
├── Cell 8: Separation (Excerpt -> QA -> Full Track) (Code)
├── Cell 9: Stem Post-Processing (Alignment-Safe) (Code + Form)
├── Cell 10: Transcription Routing + Execution (Code + Form)
├── Cell 11: MIDI Cleanup + Tempo/Quantize + Program Map (Code + Form)
├── Cell 12: Preview + QA Dashboard + Rerun Controls (Gradio)
├── Cell 13: Export + Reports + Zip (Code + Form)
└── Cell 14: Cleanup (Code)
```

These cells delegate all heavy logic to `soundlab.pipeline` and only manage UI, configuration,
and artifact display. Optional voice generation can be inserted before export when the voice
extras are installed.

### 5.3 Example Notebook Cells

The examples below show minimal patterns; max-quality mode extends these with QA scoring,
candidate selection, and post-processing hooks.

#### Cell 2: Environment Setup

```python
# @title ⚙️ Environment Setup
# @markdown Configure the processing environment

gpu_mode = "auto"  # @param ["auto", "force_gpu", "force_cpu"]
log_level = "INFO"  # @param ["DEBUG", "INFO", "WARNING", "ERROR"]
output_base = "/content/soundlab_outputs"  # @param {type:"string"}

# === Execution ===
import os
from pathlib import Path

# Configure environment
os.environ["SOUNDLAB_LOG_LEVEL"] = log_level
os.environ["SOUNDLAB_GPU_MODE"] = gpu_mode

OUTPUT_DIR = Path(output_base)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories
STEMS_DIR = OUTPUT_DIR / "stems"
MIDI_DIR = OUTPUT_DIR / "midi"
EFFECTS_DIR = OUTPUT_DIR / "effects"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
VOICE_DIR = OUTPUT_DIR / "voice"
EXPORTS_DIR = OUTPUT_DIR / "exports"

for d in [STEMS_DIR, MIDI_DIR, EFFECTS_DIR, ANALYSIS_DIR, VOICE_DIR, EXPORTS_DIR]:
    d.mkdir(exist_ok=True)

print(f"✅ Output directory: {OUTPUT_DIR}")
```

#### Cell 3: Package Installation

```python
# @title 📦 Install SoundLab
# @markdown Install the soundlab package and dependencies

install_voice = False  # @param {type:"boolean"}
install_from = "pypi"  # @param ["pypi", "github_main", "github_dev"]

# === Execution ===
import subprocess
import sys

def install_package():
    """Install soundlab with appropriate extras."""
    extras = "[notebook"
    if install_voice:
        extras += ",voice"
    extras += "]"
    
    if install_from == "pypi":
        pkg = f"soundlab{extras}"
    elif install_from == "github_main":
        pkg = f"soundlab{extras} @ git+https://github.com/wyattowalsh/soundlab.git"
    else:  # github_dev
        pkg = f"soundlab{extras} @ git+https://github.com/wyattowalsh/soundlab.git@dev"
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print(f"✅ Installed: {pkg}")

install_package()

# Verify installation
import soundlab
print(f"✅ SoundLab version: {soundlab.__version__}")

# Configure logging
from soundlab.utils.logging import configure_logging
configure_logging(level=log_level)
```

#### Cell 4: File Upload Interface

```python
# @title 🎵 Upload Audio
# @markdown Upload your audio file for processing

# === Execution ===
import gradio as gr
from soundlab.io import load_audio, get_audio_metadata
from soundlab.core.audio import AudioSegment

# Global state
CURRENT_AUDIO: AudioSegment | None = None

def handle_upload(audio_path: str | None) -> tuple[str, str]:
    """Handle audio file upload."""
    global CURRENT_AUDIO
    
    if audio_path is None:
        return "No file uploaded", ""
    
    try:
        CURRENT_AUDIO = load_audio(audio_path)
        meta = CURRENT_AUDIO.metadata
        
        info = f"""
        **File:** {Path(audio_path).name}
        **Duration:** {meta.duration_str}
        **Sample Rate:** {meta.sample_rate} Hz
        **Channels:** {meta.channels} ({'Stereo' if meta.is_stereo else 'Mono'})
        """
        return info.strip(), "✅ Audio loaded successfully!"
    except Exception as e:
        return f"Error: {e}", "❌ Failed to load audio"

with gr.Blocks(theme=gr.themes.Soft()) as upload_interface:
    gr.Markdown("## 🎵 Upload Audio File")
    gr.Markdown("Supported formats: WAV, MP3, FLAC, OGG, AIFF, M4A")
    
    with gr.Row():
        with gr.Column(scale=2):
            audio_input = gr.Audio(
                label="Input Audio",
                type="filepath",
                sources=["upload"],
            )
        with gr.Column(scale=1):
            info_output = gr.Markdown(label="Audio Info")
            status_output = gr.Markdown()
    
    audio_input.change(
        fn=handle_upload,
        inputs=[audio_input],
        outputs=[info_output, status_output],
    )

upload_interface.launch(height=400)
```

#### Cell 5: Stem Separation

```python
# @title 🎚️ Stem Separation
# @markdown Separate audio into individual stems (vocals, drums, bass, other)

# === Configuration ===
model = "htdemucs_ft"  # @param ["htdemucs", "htdemucs_ft", "htdemucs_6s"]
two_stems_mode = "None"  # @param ["None", "vocals", "drums", "bass", "other"]
segment_length = 7.8  # @param {type:"slider", min:1.0, max:30.0, step:0.1}
overlap = 0.25  # @param {type:"slider", min:0.1, max:0.5, step:0.05}
use_int24 = True  # @param {type:"boolean"}

# === Execution ===
from soundlab.separation import StemSeparator, SeparationConfig, DemucsModel

if CURRENT_AUDIO is None:
    print("⚠️ Please upload an audio file first!")
else:
    config = SeparationConfig(
        model=DemucsModel(model),
        segment_length=segment_length,
        overlap=overlap,
        int24=use_int24,
        two_stems=two_stems_mode if two_stems_mode != "None" else None,
    )
    
    separator = StemSeparator(config)
    
    print(f"🎵 Separating with {model}...")
    result = separator.separate(
        CURRENT_AUDIO.source_path,
        STEMS_DIR,
    )
    
    print(f"\n✅ Separation complete in {result.processing_time_seconds:.1f}s")
    print("\n📁 Output files:")
    for stem_name, stem_path in result.stems.items():
        print(f"  • {stem_name}: {stem_path}")
    
    # Store for downstream processing
    STEM_RESULT = result
```

#### Cell 10: Export & Download

```python
# @title 💾 Export & Download
# @markdown Download processed files

# === Execution ===
import zipfile
from google.colab import files as colab_files

def create_export_zip() -> Path:
    """Create a zip file of all outputs."""
    zip_path = EXPORTS_DIR / "soundlab_export.zip"
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for subdir in [STEMS_DIR, MIDI_DIR, EFFECTS_DIR, ANALYSIS_DIR, VOICE_DIR]:
            for file in subdir.glob("*"):
                if file.is_file():
                    zf.write(file, f"{subdir.name}/{file.name}")
    
    return zip_path

def download_interface():
    """Gradio interface for downloading files."""
    with gr.Blocks() as interface:
        gr.Markdown("## 💾 Download Processed Files")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Individual Files")
                file_dropdown = gr.Dropdown(
                    label="Select file",
                    choices=[str(f) for d in [STEMS_DIR, MIDI_DIR, EFFECTS_DIR] for f in d.glob("*")],
                )
                single_download = gr.File(label="Download")
            
            with gr.Column():
                gr.Markdown("### Download All")
                zip_btn = gr.Button("📦 Create ZIP", variant="primary")
                zip_download = gr.File(label="Download ZIP")
        
        file_dropdown.change(
            fn=lambda x: x,
            inputs=[file_dropdown],
            outputs=[single_download],
        )
        
        zip_btn.click(
            fn=create_export_zip,
            outputs=[zip_download],
        )
        
        return interface

download_interface().launch(height=400)
```

### 5.4 Quality Gates, Preview, and Rerun Controls

The notebook must surface quality indicators so users can trust results and re-run when needed:

- **QA report table**: separation score per candidate, residual RMS, spectral flatness, clipping
  flags, and transcription sanity stats (notes/sec, polyphony, pitch range).
- **Preview panel**: stem audio players, residual audio player, MIDI-rendered previews (per stem
  and combined), and mix reconstruction for A/B comparison.
- **Rerun controls**: buttons to select next-best separation or transcription candidate without
  clearing previous outputs.
- **Export bundle**: stems + MIDI + config snapshot + metrics CSV + env report for reproducibility.

---

## 6. Technology Stack

### 6.1 Core Processing Libraries

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| demucs | 4.0.1 | Stem separation | htdemucs_ft recommended |
| basic-pitch | 0.3.x | Audio-to-MIDI | Uses `onset_thresh`/`frame_thresh` |
| pedalboard | 0.9.x | Audio effects | Spotify's DSP library |
| librosa | 0.10.x | Audio analysis | BPM, spectral features |
| coqui-tts | 0.22.x | Text-to-speech | XTTS-v2, idiap fork |

### 6.2 Infrastructure Libraries

| Library | Purpose | Usage Pattern |
|---------|---------|---------------|
| pydantic | Data validation | All config/result models |
| tenacity | Retry logic | `@retry` decorator on I/O ops |
| loguru | Logging | Structured logging with levels |
| tqdm | Progress bars | Long-running operations |
| httpx | HTTP client | Model downloads |
| soundfile | Audio I/O | High-quality read/write |

### 6.3 Development Tools

| Tool | Purpose |
|------|---------|
| uv | Package/project management |
| ruff | Linting + formatting |
| ty | Type checking |
| pytest | Testing |
| hypothesis | Property-based testing |
| pre-commit | Git hooks |
| mkdocs-material | Documentation |

---

## 7. Module Specifications

### 7.1 Stem Separation

**Models:**

| Model | Stems | Quality | Speed (3min/T4) | Recommended For |
|-------|-------|---------|-----------------|-----------------|
| htdemucs | 4 | Good | ~45s | Quick previews |
| htdemucs_ft | 4 | Best | ~3min | Production use |
| htdemucs_6s | 6 | Variable | ~4min | Piano extraction (unreliable) |

**Memory Management:**

- T4 GPU: 16GB VRAM
- Max duration without segmentation: ~5 minutes
- Segment-based processing: Required for >5min with htdemucs_ft
- Recommendation: Always use `config.split=True`

### 7.2 Audio-to-MIDI Transcription

**Parameters:**

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| onset_thresh | 0.1-0.9 | 0.5 | Note onset sensitivity |
| frame_thresh | 0.1-0.9 | 0.3 | Frame activation threshold |
| minimum_note_length | 10-200ms | 58ms | Minimum note duration |
| minimum_frequency | 20-500Hz | 32.7Hz | Lowest detected pitch (C1) |
| maximum_frequency | 1000-8000Hz | 2093Hz | Highest detected pitch (C7) |

**Output Formats:**
- MIDI file (.mid)
- Piano roll visualization (PNG)
- Note list JSON with timestamps

### 7.3 Audio Analysis

**Features:**

| Feature | Algorithm | Output |
|---------|-----------|--------|
| BPM | librosa.beat.beat_track | Float + confidence |
| Key | Krumhansl-Schmuckler | Key, mode, Camelot code |
| Loudness | pyloudnorm | LUFS, dynamic range |
| Spectral | librosa.feature | Centroid, bandwidth, rolloff |
| Onsets | librosa.onset.onset_detect | Timestamps, count |

### 7.4 Effects Chain

**Available Effects (Pedalboard 0.9.x):**

| Category | Effects |
|----------|---------|
| Dynamics | Compressor, Limiter, NoiseGate, Gain |
| EQ | HighpassFilter, LowpassFilter, HighShelfFilter, LowShelfFilter, PeakFilter |
| Time-based | Reverb, Delay, Chorus, Phaser |
| Creative | Distortion, Clipping, Resample, Invert |
| Utility | Mix, GSMFullRateCompressor |

**Note:** Pedalboard lacks native Bitcrush; use `Resample` for downsampling effects.

### 7.5 Voice Generation

#### Text-to-Speech (XTTS-v2)

**Supported Languages:** en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi

**Voice Cloning Requirements:**
- 6-30 seconds of clean audio
- Single speaker, minimal background noise
- Clear speech, no music

**Note:** Coqui company shut down Dec 2023. Use `idiap/coqui-ai-TTS` fork.

#### Singing Voice Conversion (RVC)

**Parameters:**

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| pitch_shift_semitones | -12 to +12 | 0 | Output pitch transpose |
| f0_method | rmvpe/crepe/harvest/pm | rmvpe | Pitch detection algorithm |
| index_rate | 0.0-1.0 | 0.75 | Target voice style preservation |
| protect_rate | 0.0-0.5 | 0.33 | Consonant/breath protection |

---

## 8. CI/CD Pipeline

### 8.1 GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

env:
  UV_CACHE_DIR: /tmp/.uv-cache

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      - run: uv sync --frozen --dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      - run: uv sync --frozen --dev
      - run: uv run ty check packages/soundlab/src

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      - run: uv python install ${{ matrix.python-version }}
      - run: uv sync --frozen --dev
      - run: uv run pytest tests/ -v --cov=soundlab --cov-report=xml -m "not slow and not gpu"
      - uses: codecov/codecov-action@v4
        with:
          files: coverage.xml

  test-gpu:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --frozen --dev
      - run: uv run pytest tests/ -v -m "gpu" --timeout=600
    # Note: Would need self-hosted runner with GPU

  colab-compat:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: |
          # Verify notebook syntax
          uv run python -m py_compile notebooks/soundlab_studio.ipynb || true
          # Check imports resolve
          uv sync --frozen
          uv run python -c "import soundlab; print(soundlab.__version__)"
```

### 8.2 Release Workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - "v*"

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv build --package soundlab
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

---

## 9. Performance Benchmarks

### 9.1 Target Performance (T4 GPU, 3-minute song)

| Operation | Target Time | Notes |
|-----------|-------------|-------|
| htdemucs separation | ~45s | 4 stems |
| htdemucs_ft separation | ~3min | Fine-tuned, best quality |
| Basic Pitch transcription | ~30s | Full polyphonic |
| Audio analysis (all features) | ~10s | BPM + key + loudness + spectral |
| Effects chain (10 effects) | ~5s | Full processing chain |
| **Full pipeline** | **~5min** | All modules |

### 9.2 Memory Constraints

| Resource | Limit | Mitigation |
|----------|-------|------------|
| T4 VRAM | 16GB | Segment-based processing |
| Colab RAM | 12.7GB | Stream large files |
| Disk | 100GB | Clean up intermediates |
| Runtime | 12h max | Checkpoint long operations |

---

## 10. Error Handling Strategy

### 10.1 Exception Hierarchy

```
SoundLabError (base)
├── AudioLoadError          # File I/O failures
├── AudioFormatError        # Unsupported formats
├── ModelNotFoundError      # Missing ML models
├── GPUMemoryError          # CUDA OOM
├── ProcessingError         # Algorithm failures
├── ConfigurationError      # Invalid parameters
└── VoiceConversionError    # Voice pipeline failures
```

### 10.2 Retry Configuration

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Standard retry for I/O operations
io_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((IOError, ConnectionError)),
)

# GPU retry with memory clearing
gpu_retry = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type(torch.cuda.OutOfMemoryError),
    before_sleep=lambda _: torch.cuda.empty_cache(),
)
```

---

## 11. Future Roadmap

### Phase 1: Core (v0.1.0)
- [x] Stem separation (Demucs)
- [x] Audio-to-MIDI (Basic Pitch)
- [x] Effects chain (Pedalboard)
- [x] Audio analysis (librosa)
- [x] Colab notebook

### Phase 2: Voice (v0.2.0)
- [ ] Text-to-speech (XTTS-v2)
- [ ] Singing voice conversion (RVC)
- [ ] Voice cloning interface

### Phase 3: Advanced (v0.3.0)
- [ ] Chord recognition
- [ ] Lyrics transcription (Whisper)
- [ ] Drum pattern extraction
- [ ] Batch processing

### Phase 4: Production (v0.4.0)
- [ ] Reference track matching
- [ ] Auto-tune / pitch correction
- [ ] Stereo width control
- [ ] Transient shaping

### Phase 5: Genre-Specific (v0.5.0)
- [ ] Hardstyle kick analysis
- [ ] Reverse bass generator
- [ ] 150 BPM grid quantization

---

## 12. Appendices

### A. Colab Forms Quick Reference

```python
# String dropdown
model = "htdemucs"  # @param ["htdemucs", "htdemucs_ft", "htdemucs_6s"]

# Boolean checkbox
enable_gpu = True  # @param {type:"boolean"}

# Numeric slider
threshold = 0.5  # @param {type:"slider", min:0, max:1, step:0.05}

# Integer slider
semitones = 0  # @param {type:"slider", min:-12, max:12, step:1}

# Free text
folder_name = "outputs"  # @param {type:"string"}

# Raw (no widget)
advanced_config = {}  # @param {type:"raw"}
```

### B. Gradio Audio Pattern

```python
import gradio as gr

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎵 Audio Processor")
    
    with gr.Row():
        audio_in = gr.Audio(type="filepath", sources=["upload"])
        audio_out = gr.Audio(type="filepath")
    
    process_btn = gr.Button("Process", variant="primary")
    process_btn.click(fn=process_audio, inputs=[audio_in], outputs=[audio_out])

demo.launch(share=True, debug=True)
```

### C. Development Commands

```bash
# Initial setup
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=soundlab --cov-report=html

# Type checking
uv run ty check packages/soundlab/src

# Linting
uv run ruff check .
uv run ruff format .

# Build package
uv build --package soundlab

# Update lockfile
uv lock

# Add dependency
uv add --package soundlab <package>

# Run notebook locally (for testing)
uv run jupyter notebook notebooks/
```

### D. Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SOUNDLAB_LOG_LEVEL` | Logging verbosity | `INFO` |
| `SOUNDLAB_GPU_MODE` | GPU usage mode | `auto` |
| `SOUNDLAB_CACHE_DIR` | Model cache directory | `~/.cache/soundlab` |
| `SOUNDLAB_OUTPUT_DIR` | Default output directory | `./outputs` |

---

*End of PRD*
