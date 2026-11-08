<div align="center">

# SoundLab

**One Colab workspace for understanding and transforming music.**

[![CI](https://github.com/wyattowalsh/soundlab/actions/workflows/ci.yml/badge.svg)](https://github.com/wyattowalsh/soundlab/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wyattowalsh/soundlab/blob/main/notebooks/soundlab_studio.ipynb)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Separate · Analyze · Transcribe · Process · Export

</div>

> [!IMPORTANT]
> The PyPI distribution named `soundlab` belongs to a different project. Do **not** use `pip install soundlab` for this repository. Install from this Git repository until a distinct distribution name is selected and released.

## Start in Colab

Open [`notebooks/soundlab_studio.ipynb`](notebooks/soundlab_studio.ipynb), or use the badge above.

The repository currently also contains the older root [`soundlab.ipynb`](soundlab.ipynb). The project is being consolidated toward one generated canonical notebook; both files remain until feature parity and migration checks are complete.

The safe first run should not mount Drive, publish a Gradio link, upload audio to a hosted service, or start paid processing without explicit selection.

## Current capabilities

- Stem separation and vocal isolation
- Audio and drum transcription to MIDI
- Tempo, key, loudness, spectral, and onset analysis
- Effects processing
- Optional voice-generation integrations
- Checkpointing, candidate selection, QA, previews, and export
- Colab and Gradio interfaces backed by the same Python package

Capability availability depends on the selected backend, Python version, model terms, checkpoint access, and runtime resources. A passing CPU test suite does not establish GPU fit, model quality, or perceptual correctness.

## Install for development

```bash
git clone https://github.com/wyattowalsh/soundlab.git
cd soundlab
uv sync --all-packages --extra notebook
```

The repository does not currently ship a reviewed lockfile, so CI and local development intentionally avoid `uv sync --frozen` until one is generated and validated.

## Project model

The notebook is the user-facing product. Reusable behavior belongs in the typed package under `packages/soundlab/`.

```text
notebooks/soundlab_studio.ipynb   # current interactive entrypoint
soundlab.ipynb                    # legacy entrypoint pending consolidation
packages/soundlab/src/soundlab/   # canonical reusable Python behavior
tests/                            # unit and integration evidence
```

See [`docs/canonicalization.md`](docs/canonicalization.md) for the migration plan.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check packages/soundlab/src
uv run pytest tests/ -m "not slow and not gpu"
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [AGENTS.md](AGENTS.md).

## Evidence and release status

SoundLab is an actively consolidated beta project. Static checks, local tests, managed-Colab execution, frontend rendering, external services, and perceptual review are separate evidence classes. Release notes must state exactly which checks ran on which artifact and environment.

No PyPI publication is currently enabled. Package builds are retained for installation testing only.

## License

SoundLab code is available under the [MIT License](LICENSE). Third-party model weights and services retain their own terms.
