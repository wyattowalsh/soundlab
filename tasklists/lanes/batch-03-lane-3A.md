# SoundLab Task Lane: Batch 3 Lane 3A - Separation + transcription scaffolding

**Batch:** 3
**Lane:** 3A
**Title:** Separation + transcription scaffolding
**Dependencies:** B2 complete (core + utils + io available)
**Strategy:** Create all Pydantic models first, then implementations
**Barrier:** All B3 tasks must complete before B4

## Tasks
- `B3.01` Create separation models. Output: `packages/soundlab/src/soundlab/separation/models.py`. Spec: `DemucsModel` enum, `SeparationConfig`, `StemResult` per PRD §4.3
- `B3.02` Create transcription models. Output: `packages/soundlab/src/soundlab/transcription/models.py`. Spec: `TranscriptionConfig(onset_thresh, frame_thresh, min_note_length, min_freq, max_freq)`, `NoteEvent(start, end, pitch, velocity)`, `MIDIResult(notes, path, config, processing_time)`
- `B3.06` Create separation utils. Output: `packages/soundlab/src/soundlab/separation/utils.py`. Spec: `calculate_segments(duration, segment_length, overlap) -> list[tuple[float, float]]`, `overlap_add(segments, overlap) -> AudioArray`
- `B3.07` Create transcription viz. Output: `packages/soundlab/src/soundlab/transcription/visualization.py`. Spec: `render_piano_roll(notes, output_path, figsize, colormap)` using matplotlib

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
