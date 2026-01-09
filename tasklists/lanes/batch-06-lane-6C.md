# SoundLab Task Lane: Batch 6 Lane 6C - IO + separation + transcription tests

**Batch:** 6
**Lane:** 6C
**Title:** IO + separation + transcription tests
**Dependencies:** B5 complete (package fully implemented)
**Barrier:** All B6 tasks must complete before B7

## Tasks
- `B6.07` Test io audio. Output: `tests/unit/test_audio_io.py`. Spec: Test `load_audio()`, `save_audio()` roundtrip, format detection, error handling
- `B6.08` Test separation models. Output: `tests/unit/test_separation_models.py`. Spec: Test `DemucsModel` enum properties, `SeparationConfig` validation, defaults
- `B6.09` Test transcription models. Output: `tests/unit/test_transcription_models.py`. Spec: Test `TranscriptionConfig` bounds validation, `NoteEvent` ordering

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
