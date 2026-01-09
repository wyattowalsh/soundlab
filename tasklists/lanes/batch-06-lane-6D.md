# SoundLab Task Lane: Batch 6 Lane 6D - Effects + analysis tests

**Batch:** 6
**Lane:** 6D
**Title:** Effects + analysis tests
**Dependencies:** B5 complete (package fully implemented)
**Barrier:** All B6 tasks must complete before B7

## Tasks
- `B6.10` Test effects models. Output: `tests/unit/test_effects_models.py`. Spec: Test all effect configs validate parameters, `to_plugin()` returns correct types
- `B6.11` Test analysis key. Output: `tests/unit/test_key_detection.py`. Spec: Test K-K algorithm with known key audio samples, Camelot conversion
- `B6.12` Test effects chain. Output: `tests/unit/test_effects_chain.py`. Spec: Test `EffectsChain` fluent API, empty chain passthrough, multi-effect processing

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
