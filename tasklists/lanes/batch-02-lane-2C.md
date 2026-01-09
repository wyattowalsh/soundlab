# SoundLab Task Lane: Batch 2 Lane 2C - MIDI I/O and export

**Batch:** 2
**Lane:** 2C
**Title:** MIDI I/O and export
**Dependencies:** B1 complete (core types, exceptions available)
**Barrier:** All B2 tasks must complete before B3

## Tasks
- `B2.04` Create MIDI I/O. Output: `packages/soundlab/src/soundlab/io/midi_io.py`. Spec: `load_midi(path) -> MIDIData`, `save_midi(data, path)`, `MIDIData` model with notes, tempo, time_signature
- `B2.05` Create export utilities. Output: `packages/soundlab/src/soundlab/io/export.py`. Spec: `export_audio(segment, path, format, normalize_lufs)`, `create_zip(files, output_path)`, `batch_export(segments, output_dir)`
- `B2.06` Create io __init__. Output: `packages/soundlab/src/soundlab/io/__init__.py`. Spec: Public exports: `load_audio`, `save_audio`, `get_audio_metadata`, `load_midi`, `save_midi`, `export_audio`

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
