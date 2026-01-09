# SoundLab Task Lane: Batch 6 Lane 6E - Pipeline tests

**Batch:** 6
**Lane:** 6E
**Title:** Pipeline tests
**Dependencies:** B5 complete (package fully implemented)
**Barrier:** All B6 tasks must complete before B7

## Tasks
- `B6.13` Test pipeline models. Output: `tests/unit/test_pipeline_models.py`. Spec: Validate `PipelineConfig` defaults, QA thresholds, candidate plan schema
- `B6.14` Test pipeline QA metrics. Output: `tests/unit/test_pipeline_qa.py`. Spec: Synthetic signals verify reconstruction error, spectral flatness, leakage proxies
- `B6.15` Test pipeline post-process. Output: `tests/unit/test_pipeline_postprocess.py`. Spec: Ensure alignment-safe trimming and mono AMT exports preserve sample length

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
