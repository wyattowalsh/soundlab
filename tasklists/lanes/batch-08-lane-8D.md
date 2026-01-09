# SoundLab Task Lane: Batch 8 Lane 8D - Post-processing + transcription + MIDI cleanup

**Batch:** 8
**Lane:** 8D
**Title:** Post-processing + transcription + MIDI cleanup
**Dependencies:** B7 complete (package tested and working)
**Barrier:** All B8 tasks must complete before B9

## Tasks
- `B8.10` Add stem post-processing cell. Output: `notebooks/soundlab_studio.ipynb` (cell 9). Spec: Alignment-safe filtering, mono AMT exports, clipping checks
- `B8.11` Add transcription routing cell. Output: `notebooks/soundlab_studio.ipynb` (cell 10). Spec: Per-stem backend routing + fallback matrix + confidence collection
- `B8.12` Add MIDI cleanup cell. Output: `notebooks/soundlab_studio.ipynb` (cell 11). Spec: Note cleanup, tempo detection, soft quantization, program mapping

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
