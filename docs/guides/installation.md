# Installation Guide

This guide covers all installation methods for SoundLab, including package manager installation, development setup, and GPU configuration.

## System Requirements

### Minimum Requirements

- **Python**: 3.12 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: 4 GB minimum (8 GB recommended)
- **Storage**: 5 GB for models and dependencies

### Recommended Requirements

- **Python**: 3.12+
- **RAM**: 16 GB for large audio files
- **GPU**: CUDA-capable GPU with 8+ GB VRAM
- **Storage**: 10 GB+ for models and cache

!!! info "Python Version"
    SoundLab requires Python 3.12 or higher for optimal type checking and performance. Check your version:
    ```bash
    python --version
    ```

## Installation Methods

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is the recommended package manager for SoundLab, offering fast, reliable dependency management.

#### Install uv

=== "Linux/macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "With pip"

    ```bash
    pip install uv
    ```

#### Install SoundLab from Source

```bash
# Clone the repository
git clone https://github.com/wyattwalsh/soundlab.git
cd soundlab

# Install with uv (basic features)
uv sync

# Install with optional features
uv sync --extra voice         # Add voice generation
uv sync --extra notebook      # Add Gradio notebook
uv sync --extra all           # Install everything
```

#### Verify Installation

```bash
# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Test installation
python -c "import soundlab; print(soundlab.__version__)"
```

### Option 2: Using pip

Install directly from PyPI for production use:

#### Basic Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install SoundLab
pip install soundlab
```

#### With Optional Features

=== "Voice Generation"

    ```bash
    # Includes XTTS-v2 for text-to-speech and voice cloning
    pip install soundlab[voice]
    ```

=== "Notebook Interface"

    ```bash
    # Includes Gradio for interactive notebooks
    pip install soundlab[notebook]
    ```

=== "Complete Installation"

    ```bash
    # Install all features
    pip install soundlab[all]
    ```

#### Verify Installation

```python
import soundlab
print(f"SoundLab version: {soundlab.__version__}")

# Check available features
from soundlab.core import get_config
config = get_config()
print(f"GPU available: {config.use_gpu}")
```

### Option 3: Development Installation

For contributing to SoundLab or modifying the source:

#### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/wyattwalsh/soundlab.git
cd soundlab

# Install with development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

#### Development Tools

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=soundlab --cov-report=html

# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Type checking
uv run mypy packages/soundlab/src
```

!!! tip "IDE Setup"
    For VS Code, install the Python extension and point it to `.venv/bin/python` for IntelliSense and type checking.

## GPU Setup

SoundLab automatically detects and uses CUDA GPUs when available. For optimal performance, install the GPU-accelerated version of PyTorch.

### CUDA Installation

#### Check CUDA Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

#### Install CUDA-enabled PyTorch

=== "CUDA 12.x"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

=== "CUDA 11.x"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

=== "CPU Only"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

!!! warning "CUDA Version"
    Make sure your PyTorch CUDA version matches your system CUDA version. Check with:
    ```bash
    nvidia-smi
    ```

### GPU Memory Management

SoundLab intelligently manages GPU memory, but you can optimize for your hardware:

```python
from soundlab.separation import SeparationConfig, DemucsModel

# For GPUs with limited VRAM (< 8 GB)
config = SeparationConfig(
    model=DemucsModel.HTDEMUCS,  # Smaller model
    segment_length=5.0,          # Smaller segments
)

# For high-end GPUs (>= 16 GB)
config = SeparationConfig(
    model=DemucsModel.HTDEMUCS_FT,  # Best quality
    segment_length=10.0,             # Larger segments
    shifts=2,                        # More shifts for quality
)
```

### CPU Fallback

If no GPU is available, SoundLab automatically falls back to CPU:

```python
# Force CPU usage
config = SeparationConfig(device="cpu")
```

!!! info "Performance Impact"
    CPU processing is 10-20x slower than GPU but produces identical results.

## Platform-Specific Notes

### Linux

Most features work out of the box. Install FFmpeg for audio format support:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg libsndfile1

# Fedora/RHEL
sudo dnf install ffmpeg libsndfile

# Arch Linux
sudo pacman -S ffmpeg libsndfile
```

### macOS

Install dependencies via Homebrew:

```bash
brew install ffmpeg libsndfile

# For M1/M2 Macs, use MPS acceleration (automatic)
# PyTorch will use Metal Performance Shaders
```

!!! tip "Apple Silicon"
    On M1/M2/M3 Macs, SoundLab automatically uses MPS (Metal Performance Shaders) for GPU acceleration.

### Windows

Install FFmpeg and add to PATH:

1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\Program Files\ffmpeg`
3. Add `C:\Program Files\ffmpeg\bin` to system PATH

Alternatively, use Chocolatey:

```powershell
choco install ffmpeg
```

!!! warning "Windows Long Paths"
    Enable long path support in Windows 10/11:
    ```powershell
    # Run as Administrator
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
      -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
    ```

## Docker Installation

Run SoundLab in a container with GPU support:

### Build Docker Image

```bash
# Clone repository
git clone https://github.com/wyattwalsh/soundlab.git
cd soundlab

# Build image
docker build -t soundlab:latest .
```

### Run Container

=== "With GPU"

    ```bash
    docker run --gpus all -it \
      -v $(pwd)/data:/data \
      soundlab:latest
    ```

=== "CPU Only"

    ```bash
    docker run -it \
      -v $(pwd)/data:/data \
      soundlab:latest
    ```

### Docker Compose

```yaml
version: '3.8'

services:
  soundlab:
    image: soundlab:latest
    volumes:
      - ./data:/data
      - ./output:/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Google Colab

Try SoundLab without local installation:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wyattwalsh/soundlab/blob/main/notebooks/soundlab_studio.ipynb)

### Colab Installation

```python
# Install SoundLab
!pip install soundlab[all]

# Upload audio files
from google.colab import files
uploaded = files.upload()

# Process audio
from soundlab.separation import StemSeparator
separator = StemSeparator()
result = separator.separate("song.mp3", "output/")

# Download results
from google.colab import files
files.download("output/vocals.wav")
```

!!! tip "Free GPU"
    Google Colab provides free GPU access (T4) with limitations. For production use, consider Colab Pro.

## Model Downloads

SoundLab automatically downloads required models on first use:

### Model Storage

Models are cached in:

- **Linux/macOS**: `~/.cache/soundlab/`
- **Windows**: `%LOCALAPPDATA%\soundlab\cache\`

### Manual Model Download

Pre-download models to avoid delays:

```python
from soundlab.separation import StemSeparator
from soundlab.transcription import MIDITranscriber
from soundlab.voice import TextToSpeech

# Download separation models
separator = StemSeparator()

# Download transcription models
transcriber = MIDITranscriber()

# Download voice models (optional)
tts = TextToSpeech()
```

### Model Sizes

| Model | Size | Purpose |
|-------|------|---------|
| Demucs htdemucs | ~320 MB | Stem separation |
| Demucs htdemucs_ft | ~320 MB | Stem separation (fine-tuned) |
| Basic Pitch | ~40 MB | MIDI transcription |
| XTTS-v2 | ~1.8 GB | Text-to-speech (optional) |

## Troubleshooting

### Common Installation Issues

#### ModuleNotFoundError

```python
# Error: No module named 'soundlab'
# Solution: Ensure soundlab is installed
pip install soundlab

# Or activate virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

#### CUDA Errors

```python
# Error: CUDA out of memory
# Solution: Reduce segment length
config = SeparationConfig(segment_length=5.0)

# Or force CPU
config = SeparationConfig(device="cpu")
```

#### FFmpeg Not Found

```bash
# Linux/macOS
which ffmpeg  # Should show path

# Windows
where ffmpeg  # Should show path

# If not found, install per platform instructions above
```

#### Permission Errors

```bash
# Linux/macOS: Use virtual environment
python -m venv venv
source venv/bin/activate
pip install soundlab

# Or install with --user flag
pip install --user soundlab
```

### Getting Help

If you encounter issues:

1. Check the [FAQ](https://github.com/wyattwalsh/soundlab#faq)
2. Search [GitHub Issues](https://github.com/wyattwalsh/soundlab/issues)
3. Ask in [GitHub Discussions](https://github.com/wyattwalsh/soundlab/discussions)
4. Include:
    - Python version (`python --version`)
    - SoundLab version (`python -c "import soundlab; print(soundlab.__version__)"`)
    - Operating system
    - Error message and traceback

## Upgrade Instructions

### Upgrade SoundLab

=== "uv"

    ```bash
    uv sync --upgrade
    ```

=== "pip"

    ```bash
    pip install --upgrade soundlab
    ```

### Upgrade with New Features

```bash
# Add voice generation to existing installation
pip install --upgrade soundlab[voice]

# Or upgrade everything
pip install --upgrade soundlab[all]
```

### Check Current Version

```python
import soundlab
print(soundlab.__version__)

# Check for updates
pip list --outdated | grep soundlab
```

## Uninstallation

### Remove Package

```bash
# With pip
pip uninstall soundlab

# Remove cache and models
rm -rf ~/.cache/soundlab/  # Linux/macOS
# or
rmdir /s %LOCALAPPDATA%\soundlab\cache\  # Windows
```

### Clean Development Environment

```bash
# Remove virtual environment
rm -rf .venv/

# Remove pre-commit hooks
pre-commit uninstall
```

## Next Steps

Now that SoundLab is installed, continue with:

- **[Quick Start Guide](quickstart.md)** - Basic usage examples
- **[Stem Separation Guide](separation.md)** - Isolate instruments
- **[API Reference](../api/index.md)** - Detailed API documentation

!!! success "Installation Complete!"
    You're ready to start processing audio with SoundLab. Head to the [Quick Start guide](quickstart.md) to begin!
