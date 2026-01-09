# SoundLab Task Lane: Batch 4 Lane 4E - Pipeline implementations

**Batch:** 4
**Lane:** 4E
**Title:** Pipeline implementations
**Dependencies:** B3 complete (all models available)
**Barrier:** All B4 tasks must complete before B5

## Tasks
- `B4.11` Create pipeline interfaces. Output: `packages/soundlab/src/soundlab/pipeline/interfaces.py`. Spec: Protocols: `SeparatorBackend`, `TranscriberBackend`, `StemPostProcessor`, `MidiPostProcessor`, `QAEvaluator` per PRD §4.6
- `B4.12` Create pipeline candidates. Output: `packages/soundlab/src/soundlab/pipeline/candidates.py`. Spec: Candidate strategy generation for excerpt trials and full runs; supports staged separation plan
- `B4.13` Create pipeline QA. Output: `packages/soundlab/src/soundlab/pipeline/qa.py`. Spec: Separation QA metrics (reconstruction, spectral flatness, leakage proxies) + MIDI sanity metrics
- `B4.14` Create pipeline post-processing. Output: `packages/soundlab/src/soundlab/pipeline/postprocess.py`. Spec: Alignment-safe stem cleaning, mono AMT exports, MIDI cleanup helpers
- `B4.15` Create pipeline cache. Output: `packages/soundlab/src/soundlab/pipeline/cache.py`. Spec: Run ID hashing, cache paths, stage checkpoints, resume helpers
- `B4.16` Create pipeline __init__. Output: `packages/soundlab/src/soundlab/pipeline/__init__.py`. Spec: Public exports: config, interfaces, QA, candidates, postprocess, cache

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
