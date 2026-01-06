# Contributing to SoundLab

Thank you for your interest in contributing to SoundLab! We're excited to have you join our community of developers, researchers, and audio enthusiasts working together to build a production-ready music processing platform.

SoundLab is a modular audio processing workspace that provides stem separation, audio-to-MIDI transcription, effects processing, audio analysis, and voice generation capabilities. Whether you're fixing a bug, adding a feature, improving documentation, or helping with tests, your contributions are valuable and appreciated.

This guide will help you get started with contributing to the project. If you have any questions, don't hesitate to open an issue or start a discussion on GitHub.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting Guidelines](#issue-reporting-guidelines)
- [Code of Conduct](#code-of-conduct)
- [License](#license)

---

## Development Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12 or higher** - SoundLab requires Python 3.12+
  - Check your version: `python --version`
  - Download from: [python.org/downloads](https://www.python.org/downloads/)

- **uv** - Fast, reliable Python package manager
  - Recommended for dependency management
  - Installation instructions below

- **Git** - Version control system
  - Check your version: `git --version`
  - Download from: [git-scm.com](https://git-scm.com/)

### Installing uv

If you don't have `uv` installed, install it using one of these methods:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

For more installation options, see the [uv documentation](https://github.com/astral-sh/uv).

### Cloning the Repository

1. **Fork the repository** on GitHub by clicking the "Fork" button at the top right of the repository page.

2. **Clone your fork** to your local machine:

   ```bash
   git clone https://github.com/YOUR-USERNAME/soundlab.git
   cd soundlab
   ```

3. **Add the upstream remote** to keep your fork in sync:

   ```bash
   git remote add upstream https://github.com/wyattwalsh/soundlab.git
   ```

### Installing Dependencies with uv

Install all project dependencies including development tools:

```bash
uv sync --dev
```

This command will:
- Create a virtual environment in `.venv/`
- Install the `soundlab` package in editable mode
- Install all development dependencies (pytest, ruff, ty, pre-commit, etc.)
- Use the lockfile (`uv.lock`) to ensure reproducible builds

To install with optional features:

```bash
# Install with voice generation support
uv sync --dev --extra voice

# Install with notebook interface
uv sync --dev --extra notebook

# Install everything
uv sync --dev --all-extras
```

### Running Tests

Verify your installation by running the test suite:

```bash
# Run all tests with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov

# Run specific test file
uv run pytest packages/soundlab/tests/test_separation.py -v
```

All tests should pass. If you encounter any issues, please open an issue on GitHub.

### Setting Up Pre-commit Hooks

We use pre-commit hooks to automatically check your code before commits:

```bash
uv run pre-commit install
```

The pre-commit hooks will automatically:
- Format code with `ruff format`
- Check for linting issues with `ruff check`
- Run type checking with `ty`
- Validate commit message format
- Check for common issues (trailing whitespace, merge conflicts, large files, etc.)

You can manually run the hooks at any time:

```bash
# Run hooks on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff --all-files
```

---

## Code Style Guidelines

We maintain high code quality standards to ensure the codebase is readable, maintainable, and robust.

### Ruff for Linting and Formatting

SoundLab uses **Ruff** for both linting and formatting. Ruff is configured in `pyproject.toml` with the following settings:

- **Line length**: 100 characters maximum
- **Target version**: Python 3.12
- **Import sorting**: Automatic with isort-compatible ordering

#### Running Ruff

```bash
# Check for linting issues
uv run ruff check .

# Auto-fix linting issues where possible
uv run ruff check . --fix

# Format code
uv run ruff format .

# Check formatting without making changes
uv run ruff format . --check
```

#### Enabled Linting Rules

Our Ruff configuration enables the following rule sets:

- `E`, `W` - pycodestyle errors and warnings
- `F` - Pyflakes
- `I` - isort (import sorting)
- `B` - flake8-bugbear (common bugs)
- `C4` - flake8-comprehensions
- `UP` - pyupgrade (modern Python syntax)
- `ARG` - flake8-unused-arguments
- `SIM` - flake8-simplify
- `TCH` - flake8-type-checking
- `PTH` - flake8-use-pathlib (prefer pathlib over os.path)
- `RUF` - Ruff-specific rules

### Type Hints Required

**All public functions, methods, and classes must have type hints.**

- Use built-in types for Python 3.12+ (`list`, `dict`, `tuple` instead of `List`, `Dict`, `Tuple`)
- Use `typing` module for advanced types (`Optional`, `Union`, `Protocol`, etc.)
- Type hint return values, including `None`
- Type hint class attributes and instance variables

#### Type Checking

We use **ty** for static type checking:

```bash
# Check types in the main package
uv run ty check packages/soundlab/src

# Check specific file
uv run ty check packages/soundlab/src/soundlab/separation/separator.py
```

### Docstring Format (Google Style)

**All public modules, classes, functions, and methods must have docstrings in Google style format.**

#### Google Style Docstring Template

```python
def function_name(param1: int, param2: str, param3: Optional[float] = None) -> bool:
    """Brief one-line description of the function.

    More detailed description if needed. Explain what the function does,
    any important behavior, and when to use it.

    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter
        param3: Description of optional parameter. Defaults to None.

    Returns:
        Description of the return value

    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not a string

    Example:
        >>> result = function_name(42, "hello")
        >>> print(result)
        True

    Note:
        Any additional notes or important information
    """
    pass
```

#### Complete Example

Here's a complete example demonstrating our code style standards:

```python
"""Module for audio processing utilities.

This module provides utility functions for loading, processing, and saving
audio files with various formats and configurations.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    """Configuration for audio processing operations.

    Attributes:
        sample_rate: Audio sample rate in Hz (e.g., 44100, 48000)
        channels: Number of audio channels (1=mono, 2=stereo)
        bit_depth: Bit depth for audio samples (16, 24, or 32)
        normalize: Whether to normalize audio to [-1.0, 1.0] range
    """

    sample_rate: int = Field(default=44100, ge=8000, le=192000)
    channels: int = Field(default=2, ge=1, le=8)
    bit_depth: int = Field(default=16, ge=16, le=32)
    normalize: bool = Field(default=True)


def process_audio(
    audio: np.ndarray,
    config: AudioConfig,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """Process audio with the given configuration.

    Applies normalization and resampling according to the provided configuration.
    Optionally saves the processed audio to disk.

    Args:
        audio: Input audio array with shape (samples, channels) or (samples,)
            for mono audio
        config: Processing configuration including sample rate and normalization
        output_path: Optional path to save processed audio. If None, audio is
            not saved. Defaults to None.

    Returns:
        Processed audio array with shape (samples, channels)

    Raises:
        ValueError: If audio shape is invalid or empty
        IOError: If output_path directory doesn't exist

    Example:
        >>> audio = np.random.randn(44100, 2)
        >>> config = AudioConfig(sample_rate=44100, normalize=True)
        >>> processed = process_audio(audio, config)
        >>> processed.shape
        (44100, 2)

    Note:
        For large audio files, consider processing in chunks to reduce
        memory usage.
    """
    if audio.size == 0:
        raise ValueError("Audio array is empty")

    if audio.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D audio array, got {audio.ndim}D")

    # Ensure 2D array (samples, channels)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    # Apply normalization
    if config.normalize:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

    # Save if path provided
    if output_path is not None:
        if not output_path.parent.exists():
            raise IOError(f"Output directory does not exist: {output_path.parent}")
        save_audio(audio, output_path, config)

    return audio


def save_audio(audio: np.ndarray, path: Path, config: AudioConfig) -> None:
    """Save audio array to file.

    Args:
        audio: Audio array with shape (samples, channels)
        path: Output file path
        config: Audio configuration for encoding

    Raises:
        IOError: If file cannot be written
    """
    # Implementation details...
    pass
```

### Additional Style Guidelines

- **Imports**: Organize imports in three groups (standard library, third-party, first-party)
- **Naming conventions**:
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private attributes: `_leading_underscore`
- **Line breaks**: Use line breaks to improve readability for long expressions
- **Comments**: Write clear, concise comments explaining "why", not "what"

---

## Testing Guidelines

We maintain comprehensive test coverage to ensure code quality and prevent regressions.

### Unit Tests with pytest

SoundLab uses **pytest** as the testing framework with several useful plugins:

- `pytest-cov` - Code coverage reporting
- `pytest-asyncio` - Async test support
- `hypothesis` - Property-based testing

#### Running Tests

```bash
# Run all tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest packages/soundlab/tests/test_separation.py -v

# Run specific test function
uv run pytest packages/soundlab/tests/test_separation.py::test_stem_separator_init -v

# Run tests matching a pattern
uv run pytest -v -k "test_separation"

# Run tests with specific markers
uv run pytest -v -m "not slow"

# Run with output capture disabled (see print statements)
uv run pytest -v -s
```

#### Writing Unit Tests

Place unit tests in the appropriate test directory:

```
packages/soundlab/tests/
├── test_separation.py      # Tests for stem separation
├── test_transcription.py   # Tests for MIDI transcription
├── test_effects.py         # Tests for audio effects
├── test_analysis.py        # Tests for audio analysis
└── test_voice.py           # Tests for voice generation
```

**Unit test example:**

```python
"""Tests for audio processing utilities."""

import numpy as np
import pytest
from soundlab.io.audio import load_audio, save_audio, AudioFormat


class TestAudioLoading:
    """Tests for audio loading functionality."""

    def test_load_wav_file(self, tmp_path):
        """Test loading a WAV file."""
        # Arrange: Create test audio file
        audio_path = tmp_path / "test.wav"
        sample_rate = 44100
        audio = np.random.randn(sample_rate, 2).astype(np.float32)
        save_audio(audio, audio_path, sample_rate)

        # Act: Load the audio
        loaded_audio, loaded_sr = load_audio(audio_path)

        # Assert: Check loaded audio matches original
        assert loaded_audio.shape == audio.shape
        assert loaded_sr == sample_rate
        np.testing.assert_array_almost_equal(loaded_audio, audio, decimal=5)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_audio("nonexistent.wav")

    def test_load_invalid_format(self, tmp_path):
        """Test loading an invalid format raises ValueError."""
        invalid_file = tmp_path / "invalid.xyz"
        invalid_file.write_text("not audio data")

        with pytest.raises(ValueError, match="Unsupported audio format"):
            load_audio(invalid_file)

    @pytest.mark.parametrize("sample_rate", [22050, 44100, 48000])
    def test_load_various_sample_rates(self, tmp_path, sample_rate):
        """Test loading audio files with various sample rates."""
        audio_path = tmp_path / f"test_{sample_rate}.wav"
        audio = np.random.randn(sample_rate, 2).astype(np.float32)
        save_audio(audio, audio_path, sample_rate)

        loaded_audio, loaded_sr = load_audio(audio_path)
        assert loaded_sr == sample_rate
```

### Integration Tests

Integration tests verify that multiple components work together correctly.

**Mark integration tests:**

```python
import pytest


@pytest.mark.integration
def test_full_stem_separation_pipeline(audio_file, output_dir):
    """Test complete stem separation pipeline from input to output."""
    from soundlab.separation import StemSeparator, SeparationConfig

    config = SeparationConfig(model="htdemucs")
    separator = StemSeparator(config)

    result = separator.separate(audio_file, output_dir)

    # Verify all stems are created
    assert result.vocals.exists()
    assert result.stems["drums"].exists()
    assert result.stems["bass"].exists()
    assert result.stems["other"].exists()

    # Verify audio quality
    assert result.processing_time_seconds > 0
    assert result.model_name == "htdemucs"
```

**Run only integration tests:**

```bash
uv run pytest -v -m integration
```

**Skip integration tests:**

```bash
uv run pytest -v -m "not integration"
```

### Test Markers

We use pytest markers to categorize tests:

- `@pytest.mark.slow` - Tests that take more than 5 seconds
- `@pytest.mark.gpu` - Tests that require GPU
- `@pytest.mark.integration` - Integration tests

**Available markers in pyproject.toml:**

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')",
    "integration: marks tests as integration tests",
]
```

### Coverage Requirements

We aim for **high test coverage** (>80%) for all new code.

#### Running Coverage Reports

```bash
# Run tests with coverage report
uv run pytest --cov

# Generate HTML coverage report
uv run pytest --cov --cov-report=html
# Open htmlcov/index.html in browser

# Show missing lines
uv run pytest --cov --cov-report=term-missing

# Check coverage for specific package
uv run pytest --cov=soundlab.separation
```

#### Coverage Configuration

Coverage is configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["packages/soundlab/src"]
branch = true
parallel = true
omit = [
    "*/tests/*",
    "*/test_*.py",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
```

#### Coverage Requirements for PRs

- **New features**: Must include tests with >80% coverage
- **Bug fixes**: Must include regression tests
- **Refactoring**: Must maintain or improve existing coverage
- **Critical paths**: Aim for 100% coverage (e.g., error handling, validation)

### Testing Best Practices

1. **Arrange-Act-Assert pattern**: Structure tests clearly
2. **One assertion per test**: Focus each test on a single behavior
3. **Use fixtures**: Share common setup between tests
4. **Test edge cases**: Empty inputs, None values, boundary conditions
5. **Test error cases**: Verify proper exception handling
6. **Use parametrize**: Test multiple inputs with one test function
7. **Mock external dependencies**: Use `pytest-mock` or `unittest.mock`
8. **Clear test names**: Use descriptive names that explain what is tested

---

## Pull Request Process

### Branch Naming Conventions

Create a new branch for your changes following these conventions:

- `feature/short-description` - New features
- `fix/short-description` - Bug fixes
- `docs/short-description` - Documentation updates
- `refactor/short-description` - Code refactoring
- `test/short-description` - Test additions/improvements
- `chore/short-description` - Maintenance tasks

**Examples:**

```bash
git checkout -b feature/add-pitch-shift-effect
git checkout -b fix/stem-separator-memory-leak
git checkout -b docs/update-transcription-examples
```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/) for clear, semantic commit messages.

#### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Types

- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, whitespace)
- `refactor` - Code refactoring (no functional changes)
- `perf` - Performance improvements
- `test` - Adding or updating tests
- `build` - Build system or dependency changes
- `ci` - CI/CD pipeline changes
- `chore` - Maintenance tasks

#### Scope (Optional)

The scope specifies what part of the codebase is affected:

- `separation` - Stem separation
- `transcription` - MIDI transcription
- `effects` - Audio effects
- `analysis` - Audio analysis
- `voice` - Voice generation
- `io` - Input/output utilities
- `core` - Core functionality

#### Examples

```bash
# Simple feature
feat: add pitch shifting effect

# Bug fix with scope
fix(separation): handle empty audio files correctly

# Breaking change
feat(transcription)!: redesign MIDI transcriber API

BREAKING CHANGE: MIDITranscriber.transcribe() now returns
TranscriptionResult instead of raw MIDI data

# Documentation
docs: add examples for audio analysis

# Multiple paragraphs
refactor(effects): simplify effects chain processing

Refactored the effects chain to use a more functional approach.
This makes the code easier to test and reason about.

Closes #123
```

#### Commit Message Best Practices

- Use imperative mood: "add" not "added" or "adds"
- Keep subject line under 72 characters
- Capitalize the subject line
- Don't end subject with a period
- Separate subject from body with blank line
- Wrap body at 72 characters
- Use body to explain "why" not "what"
- Reference issues and PRs: "Fixes #123", "Closes #456", "Refs #789"

### PR Template Checklist

When you create a pull request, include the following information:

#### Title

Use a clear, descriptive title following commit conventions:

```
feat(separation): add support for 6-stem separation model
```

#### Description

Provide a comprehensive description including:

```markdown
## Summary

Brief description of what this PR does.

## Changes

- Bullet point list of changes made
- Each bullet should be specific and clear
- Group related changes together

## Motivation

Why are these changes needed? What problem do they solve?

## Testing

How were these changes tested?
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] All tests pass locally

## Documentation

- [ ] Docstrings added/updated
- [ ] README updated (if needed)
- [ ] CHANGELOG updated
- [ ] Examples added/updated (if needed)

## Screenshots (if applicable)

Include screenshots or recordings for UI changes.

## Breaking Changes

List any breaking changes and migration steps.

## Related Issues

Fixes #123
Closes #456
Refs #789
```

#### Before Submitting

Ensure your PR meets these requirements:

- [ ] Code follows the style guidelines
- [ ] Type hints added to all new functions
- [ ] Docstrings added (Google style)
- [ ] Tests added/updated with >80% coverage
- [ ] All tests pass: `uv run pytest`
- [ ] Linting passes: `uv run ruff check .`
- [ ] Formatting passes: `uv run ruff format . --check`
- [ ] Type checking passes: `uv run ty check packages/soundlab/src`
- [ ] Pre-commit hooks installed and passing
- [ ] CHANGELOG.md updated
- [ ] Commits follow conventional commit format
- [ ] Branch is up to date with main

### PR Review Process

1. **Automated checks**: CI will run tests, linting, type checking, and coverage
2. **Code review**: Maintainers will review your code and provide feedback
3. **Address feedback**: Make requested changes in new commits
4. **Approval**: Once approved, your PR will be merged
5. **Post-merge**: Your branch will be deleted automatically

#### Review Guidelines

- Be responsive to feedback
- Don't force push after review (unless requested)
- Engage in constructive discussion
- Be patient - reviews may take a few days
- Ask questions if anything is unclear

---

## Issue Reporting Guidelines

We welcome bug reports, feature requests, and questions through GitHub Issues.

### Before Opening an Issue

1. **Search existing issues** - Check if someone has already reported it
2. **Check documentation** - Review README, docstrings, and examples
3. **Update to latest version** - Ensure you're using the latest release
4. **Minimal reproduction** - Create the smallest example that demonstrates the issue

### Bug Reports

When reporting bugs, include:

#### Environment

```
- OS: [e.g., Ubuntu 22.04, macOS 14, Windows 11]
- Python version: [e.g., 3.12.1]
- SoundLab version: [e.g., 0.1.0]
- Installation method: [e.g., uv, pip]
- GPU: [e.g., NVIDIA RTX 4090, CPU only]
```

#### Description

Clear and concise description of the bug.

#### Steps to Reproduce

```python
# Minimal code example that reproduces the issue
from soundlab.separation import StemSeparator

separator = StemSeparator()
result = separator.separate("test.mp3")  # Bug occurs here
```

#### Expected Behavior

What you expected to happen.

#### Actual Behavior

What actually happened.

#### Error Messages

```
Full stack trace if applicable:

Traceback (most recent call last):
  File "test.py", line 4, in <module>
    result = separator.separate("test.mp3")
  ...
ValueError: Invalid audio format
```

#### Additional Context

- Screenshots
- Sample audio files (if relevant and small)
- Related issues or PRs
- Possible solutions you've tried

### Feature Requests

When suggesting features, include:

#### Use Case

Describe the problem you're trying to solve.

#### Proposed Solution

How you envision the feature working.

#### Example Usage

```python
# How would you use this feature?
from soundlab.effects import NewEffect

effect = NewEffect(param1=value1)
result = effect.process(audio)
```

#### Alternatives Considered

Other ways to solve the problem.

#### Additional Context

- Links to relevant research papers
- Examples from other libraries
- Implementation complexity considerations

### Questions

For questions:

1. **Be specific**: Include code examples and context
2. **Show what you've tried**: Demonstrate your attempts to solve it
3. **Check discussions**: Consider using GitHub Discussions for open-ended questions

### Issue Labels

We use labels to categorize issues:

- `bug` - Something isn't working
- `feature` - New feature request
- `documentation` - Documentation improvements
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `question` - Further information requested
- `enhancement` - Improvement to existing feature
- `performance` - Performance optimization
- `testing` - Test-related issues

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

**Positive behavior includes:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**

- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Our Responsibilities

Project maintainers are responsible for clarifying standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

### Scope

This Code of Conduct applies within all project spaces, including GitHub repositories, discussions, issues, pull requests, and any other forums created by the project team.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by opening an issue or contacting the project maintainers directly. All complaints will be reviewed and investigated promptly and fairly.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org/), version 2.1.

---

## License

By contributing to SoundLab, you agree that your contributions will be licensed under the **MIT License**.

The MIT License is a permissive open-source license that allows anyone to:

- Use the software for any purpose
- Modify the software
- Distribute the software
- Sublicense the software
- Use the software commercially

**With the following conditions:**

- The original copyright notice and license must be included in any substantial portions of the software
- The software is provided "as is" without warranty of any kind

**Full license text:** See [LICENSE](LICENSE) file in the repository root.

### Copyright

Copyright (c) 2026 Wyatt Walsh

---

## Additional Resources

### Useful Commands

```bash
# Development workflow
uv sync --dev                          # Install dependencies
uv run pytest -v                       # Run tests
uv run pytest --cov                    # Run tests with coverage
uv run ruff check .                    # Check linting
uv run ruff check . --fix              # Fix linting issues
uv run ruff format .                   # Format code
uv run ty check packages/soundlab/src  # Type check

# Pre-commit
uv run pre-commit install              # Install hooks
uv run pre-commit run --all-files      # Run all hooks

# Git workflow
git checkout -b feature/my-feature     # Create feature branch
git add .                              # Stage changes
git commit -m "feat: add new feature"  # Commit with message
git push origin feature/my-feature     # Push to remote
```

### Project Structure

```
soundlab/
├── packages/
│   └── soundlab/
│       ├── src/soundlab/
│       │   ├── core/           # Core data models and exceptions
│       │   ├── separation/     # Stem separation (Demucs)
│       │   ├── transcription/  # Audio-to-MIDI (Basic Pitch)
│       │   ├── effects/        # Audio effects (Pedalboard)
│       │   ├── analysis/       # Audio analysis (librosa)
│       │   ├── voice/          # Voice generation (XTTS-v2, RVC)
│       │   ├── io/             # Audio/MIDI I/O utilities
│       │   └── utils/          # GPU management, logging, retry
│       └── tests/              # Test files
├── notebooks/                  # Jupyter notebooks
├── pyproject.toml              # Project configuration
├── uv.lock                     # Dependency lockfile
├── README.md                   # Project documentation
├── CONTRIBUTING.md             # This file
├── CHANGELOG.md                # Version history
└── LICENSE                     # MIT License
```

### Links

- **Repository**: [github.com/wyattwalsh/soundlab](https://github.com/wyattwalsh/soundlab)
- **Issues**: [github.com/wyattwalsh/soundlab/issues](https://github.com/wyattwalsh/soundlab/issues)
- **Discussions**: [github.com/wyattwalsh/soundlab/discussions](https://github.com/wyattwalsh/soundlab/discussions)
- **Documentation**: [github.com/wyattwalsh/soundlab#readme](https://github.com/wyattwalsh/soundlab#readme)
- **PyPI**: [pypi.org/project/soundlab/](https://pypi.org/project/soundlab/)

### Getting Help

If you need help:

1. Check the [README](README.md) and existing documentation
2. Search [existing issues](https://github.com/wyattwalsh/soundlab/issues)
3. Ask in [GitHub Discussions](https://github.com/wyattwalsh/soundlab/discussions)
4. Open a new issue with the `question` label

---

Thank you for contributing to SoundLab! Your efforts help make this project better for everyone. We appreciate your time, expertise, and collaboration in building a world-class audio processing platform.

Happy coding! 🎵
