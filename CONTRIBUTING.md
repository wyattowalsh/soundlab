# Contributing to SoundLab

Thanks for your interest in contributing. This guide covers local setup, workflow, and validation.

SoundLab is a beta workspace. `soundlab.__version__` is `0.1.0`. Package metadata in `packages/soundlab/pyproject.toml` is `1.0.0`. PyPI publication is disabled. The PyPI project named `soundlab` is a different project. The repository policy page is [README.md](README.md).

## Development setup

1. Install Python 3.12 or newer. The workspace root requires it. The library package itself allows Python 3.10+.
2. Install `uv` (https://github.com/astral-sh/uv).
3. Sync the workspace.

The root project is `soundlab-workspace`. It does not depend on the library, so `uv sync` without `--all-packages` does not install SoundLab.

```bash
uv sync --all-packages
```

The repository README installs the notebook extra as well:

```bash
uv sync --all-packages --extra notebook
```

Extras are declared on `packages/soundlab`, not on the workspace root. A local editable install of the notebook extra, which does not select the `all` extra, is:

```bash
uv pip install -e "packages/soundlab[notebook]"
```

A git install needs the package subdirectory:

```bash
uv pip install "soundlab[notebook] @ git+https://github.com/wyattowalsh/soundlab.git#subdirectory=packages/soundlab"
```

There is no `uv.lock` in this repository.

## Workflow

- Create a focused branch per change.
- Keep changes small and scoped to a single concern when possible.
- Update docs and tests alongside code changes.
- Open a PR with a clear description of the intent and testing performed.
- Use Conventional Commits (https://www.conventionalcommits.org/).

Examples:

- `feat: add new analysis model`
- `fix: handle empty audio input`
- `docs: update README quick start`
- `test: add separation unit coverage`

## Validation

Run these commands before requesting review:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check packages/soundlab/src
uv run pytest tests/ -m "not slow and not gpu"
```

A passing CPU run is local runtime evidence. It does not establish GPU fit, managed-Colab execution, or perceptual quality. If a command fails, record the failure in the PR with enough context to reproduce it.

Notebook edits need parity checks against `packages/soundlab` and the other entrypoint. See [AGENTS.md](AGENTS.md).
