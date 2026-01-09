# SoundLab Task Lane: Batch 4 Lane 4C - Analysis implementations

**Batch:** 4
**Lane:** 4C
**Title:** Analysis implementations
**Dependencies:** B3 complete (all models available)
**Barrier:** All B4 tasks must complete before B5

## Tasks
- `B4.05` Create key detection. Output: `packages/soundlab/src/soundlab/analysis/key.py`. Spec: Full K-K algorithm implementation per PRD §4.4 with `MusicalKey`, `Mode` enums, `detect_key()` function
- `B4.06` Create onset detection. Output: `packages/soundlab/src/soundlab/analysis/onsets.py`. Spec: `detect_onsets(y, sr) -> OnsetResult` with timestamps, count, strength
- `B4.07` Create analysis __init__. Output: `packages/soundlab/src/soundlab/analysis/__init__.py`. Spec: Public exports + `analyze_audio(path) -> AnalysisResult` convenience function that runs all analyzers

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
