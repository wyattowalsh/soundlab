# SoundLab Task Lane: Batch 6 Lane 6B - Core + utils tests

**Batch:** 6
**Lane:** 6B
**Title:** Core + utils tests
**Dependencies:** B5 complete (package fully implemented)
**Barrier:** All B6 tasks must complete before B7

## Tasks
- `B6.03` Test core exceptions. Output: `tests/unit/test_exceptions.py`. Spec: Test exception hierarchy, inheritance, string representation
- `B6.04` Test core audio models. Output: `tests/unit/test_audio_models.py`. Spec: Test `AudioSegment`, `AudioMetadata` validation, properties, conversions
- `B6.05` Test utils gpu. Output: `tests/unit/test_gpu.py`. Spec: Test `get_device()` with mocked torch, `is_cuda_available()`
- `B6.06` Test utils logging. Output: `tests/unit/test_logging.py`. Spec: Test `configure_logging()` sets correct levels, handlers

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
