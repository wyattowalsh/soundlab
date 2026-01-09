# SoundLab Task Lane: Batch 2 Lane 2A - Utils progress and exports

**Batch:** 2
**Lane:** 2A
**Title:** Utils progress and exports
**Dependencies:** B1 complete (core types, exceptions available)
**Barrier:** All B2 tasks must complete before B3

## Tasks
- `B2.01` Create progress utilities. Output: `packages/soundlab/src/soundlab/utils/progress.py`. Spec: `ProgressCallback` protocol impl, `TqdmProgressCallback`, `GradioProgressCallback` adapters
- `B2.02` Create utils __init__. Output: `packages/soundlab/src/soundlab/utils/__init__.py`. Spec: Public exports: `get_device`, `configure_logging`, `io_retry`, `gpu_retry`, `TqdmProgressCallback`

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
