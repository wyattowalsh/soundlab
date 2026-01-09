# SoundLab Task Lane: Batch 8 Lane 8B - Input ingestion + canonical decode

**Batch:** 8
**Lane:** 8B
**Title:** Input ingestion + canonical decode
**Dependencies:** B7 complete (package tested and working)
**Barrier:** All B8 tasks must complete before B9

## Tasks
- `B8.02` Create upload interface cell. Output: `notebooks/soundlab_studio.ipynb` (cell 4). Spec: Gradio upload interface with `gr.Audio`, metadata display per PRD §5.3
- `B8.08` Add canonical decode + excerpt. Output: `notebooks/soundlab_studio.ipynb` (cells 5-6). Spec: Decode to 44.1kHz stereo, hash input, optional excerpt selector

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
