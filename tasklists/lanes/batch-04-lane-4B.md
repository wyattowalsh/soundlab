# SoundLab Task Lane: Batch 4 Lane 4B - Transcription implementations

**Batch:** 4
**Lane:** 4B
**Title:** Transcription implementations
**Dependencies:** B3 complete (all models available)
**Barrier:** All B4 tasks must complete before B5

## Tasks
- `B4.03` Create Basic Pitch wrapper. Output: `packages/soundlab/src/soundlab/transcription/basic_pitch.py`. Spec: `MIDITranscriber` class with `transcribe(audio_path, output_dir) -> MIDIResult`; use correct API: `onset_thresh`, `frame_thresh`
- `B4.04` Create transcription __init__. Output: `packages/soundlab/src/soundlab/transcription/__init__.py`. Spec: Public exports: `MIDITranscriber`, `TranscriptionConfig`, `MIDIResult`, `render_piano_roll`

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
