# SoundLab Task Lane: Batch 7 Lane 7A - Separation + transcription integration

**Batch:** 7
**Lane:** 7A
**Title:** Separation + transcription integration
**Dependencies:** B6 complete (unit tests validate components)
**Barrier:** All B7 tasks must complete before B8

## Tasks
- `B7.01` Test separation integration. Output: `tests/integration/test_separation_integration.py`. Spec: End-to-end test: load audio -> separate -> verify stems exist; mark `@pytest.mark.slow`
- `B7.02` Test transcription integration. Output: `tests/integration/test_transcription_integration.py`. Spec: End-to-end: load audio -> transcribe -> verify MIDI output; mark `@pytest.mark.slow`

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
