# SoundLab Task Lane: Batch 0 Lane 0C - Repo structure and hygiene

**Batch:** 0
**Lane:** 0C
**Title:** Repo structure and hygiene
**Dependencies:** None (batch has zero dependencies)
**Barrier:** All B0 tasks must complete before B1

## Tasks
- `B0.08` Create .gitignore. Output: `.gitignore`. Spec: Python, uv, Jupyter, IDE, OS patterns; include `.venv/`, `__pycache__/`, `*.egg-info/`, `.coverage`, `dist/`
- `B0.09` Create directory structure. Output: All `__init__.py` files. Spec: Create empty `__init__.py` in: `packages/soundlab/src/soundlab/`, and all submodules: `core/`, `separation/`, `transcription/`, `effects/`, `analysis/`, `voice/`, `io/`, `utils/`
- `B0.10` Create py.typed markers. Output: `packages/soundlab/py.typed`, `packages/soundlab/src/soundlab/py.typed`. Spec: Empty PEP 561 marker files
- `B0.11` Create test directory structure. Output: `tests/conftest.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`. Spec: Empty conftest with TODO comment, empty init files
- `B0.12` Create notebooks directory. Output: `notebooks/.gitkeep`, `notebooks/examples/.gitkeep`. Spec: Placeholder files for notebook directories

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
