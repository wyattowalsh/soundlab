# SoundLab Task Lane: Batch 4 Lane 4A - Separation implementations

**Batch:** 4
**Lane:** 4A
**Title:** Separation implementations
**Dependencies:** B3 complete (all models available)
**Barrier:** All B4 tasks must complete before B5

## Tasks
- `B4.01` Create Demucs wrapper. Output: `packages/soundlab/src/soundlab/separation/demucs.py`. Spec: `StemSeparator` class per PRD §4.3 with lazy model loading, memory checking, retry logic, segment processing
- `B4.02` Create separation __init__. Output: `packages/soundlab/src/soundlab/separation/__init__.py`. Spec: Public exports: `StemSeparator`, `SeparationConfig`, `DemucsModel`, `StemResult`

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
