# Contributing to SoundLab

Thanks for your interest in contributing! This guide covers local setup,
workflow expectations, and validation requirements.

## Development setup

1. Install Python 3.12+.
2. Install `uv` (https://github.com/astral-sh/uv).
3. Sync the environment:

```bash
uv sync
```

## Workflow

- Create a focused branch per change.
- Keep changes small and scoped to a single concern when possible.
- Update docs and tests alongside code changes.
- Open a PR with a clear description of the intent and testing performed.

## Commit conventions

Use Conventional Commits (https://www.conventionalcommits.org/):

- `feat: add new analysis model`
- `fix: handle empty audio input`
- `docs: update README quick start`
- `test: add separation unit coverage`

## Validation requirements

Run these commands before requesting review:

```bash
uv run ruff format .
uv run ruff check .
uv run ty check packages/soundlab/src
uv run pytest tests/ -v -x --tb=short
```

If a command fails, note the failure in your PR description with any context
that would help reproduce or debug the issue.
