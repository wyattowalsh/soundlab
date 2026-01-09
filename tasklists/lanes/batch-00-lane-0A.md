# SoundLab Task Lane: Batch 0 Lane 0A - Core configuration

**Batch:** 0
**Lane:** 0A
**Title:** Core configuration
**Dependencies:** None (batch has zero dependencies)
**Barrier:** All B0 tasks must complete before B1

## Tasks
- `B0.01` Create root pyproject.toml. Output: `pyproject.toml`. Spec: Workspace config with `[tool.uv.workspace]`, dev-dependencies, ruff/pytest/coverage config per PRD §3
- `B0.02` Create package pyproject.toml. Output: `packages/soundlab/pyproject.toml`. Spec: Package metadata, dependencies, optional-dependencies `[voice,notebook,all]`, build-system per PRD §3
- `B0.03` Create .python-version. Output: `.python-version`. Spec: Content: `3.12`

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
