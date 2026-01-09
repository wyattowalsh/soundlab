# SoundLab Task Lane: Batch 4 Lane 4D - Effects implementations

**Batch:** 4
**Lane:** 4D
**Title:** Effects implementations
**Dependencies:** B3 complete (all models available)
**Barrier:** All B4 tasks must complete before B5

## Tasks
- `B4.08` Create effects chain. Output: `packages/soundlab/src/soundlab/effects/chain.py`. Spec: `EffectsChain` class per PRD §4.5 with fluent API, process_array, process file
- `B4.09` Create effects implementations. Output: `packages/soundlab/src/soundlab/effects/dynamics.py`, `effects/eq.py`, `effects/time_based.py`, `effects/creative.py`. Spec: Implement `to_plugin()` for all effect configs; map to Pedalboard plugins
- `B4.10` Create effects __init__. Output: `packages/soundlab/src/soundlab/effects/__init__.py`. Spec: Public exports: `EffectsChain`, all config classes

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
