# SoundLab Task Lane: Batch 2 Lane 2B - Audio I/O

**Batch:** 2
**Lane:** 2B
**Title:** Audio I/O
**Dependencies:** B1 complete (core types, exceptions available)
**Barrier:** All B2 tasks must complete before B3

## Tasks
- `B2.03` Create audio I/O. Output: `packages/soundlab/src/soundlab/io/audio_io.py`. Spec: `load_audio(path) -> AudioSegment`, `save_audio(segment, path, format)`, `get_audio_metadata(path) -> AudioMetadata`; use soundfile + pydub fallback

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
