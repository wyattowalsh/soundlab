# SoundLab Task Lane: Batch 11 Lane 11A - Sequential validation (single lane)

**Batch:** 11
**Lane:** 11A
**Title:** Sequential validation (single lane)
**Dependencies:** B10 complete

## Tasks
- `B11.01` Run full test suite. Output: none. Spec: `uv run pytest tests/ -v --cov=soundlab`; ensure >80% coverage
- `B11.02` Build and test package. Output: none. Spec: `uv build --package soundlab`; test install in fresh venv
- `B11.03` Validate Colab notebook. Output: none. Spec: Open in Colab, run all cells, verify no errors

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
