# SoundLab Task Lane: Batch 6 Lane 6A - Fixtures and shared test data

**Batch:** 6
**Lane:** 6A
**Title:** Fixtures and shared test data
**Dependencies:** B5 complete (package fully implemented)
**Barrier:** All B6 tasks must complete before B7

## Tasks
- `B6.01` Create test fixtures. Output: `tests/conftest.py`. Spec: pytest fixtures: `sample_audio_path`, `sample_mono_audio`, `sample_stereo_audio`, `temp_output_dir`, `mock_gpu_available`
- `B6.02` Create test audio files. Output: `tests/fixtures/audio/sine_440hz_3s.wav`, `tests/fixtures/audio/silence_1s.wav`. Spec: Generate programmatically in conftest or include small test files

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
