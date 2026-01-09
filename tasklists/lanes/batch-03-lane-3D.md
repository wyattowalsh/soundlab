# SoundLab Task Lane: Batch 3 Lane 3D - Voice + pipeline models

**Batch:** 3
**Lane:** 3D
**Title:** Voice + pipeline models
**Dependencies:** B2 complete (core + utils + io available)
**Strategy:** Create all Pydantic models first, then implementations
**Barrier:** All B3 tasks must complete before B4

## Tasks
- `B3.05` Create voice models. Output: `packages/soundlab/src/soundlab/voice/models.py`. Spec: `TTSConfig(text, language, speaker_wav, temperature, speed)`, `TTSResult(audio_path, processing_time)`, `SVCConfig(pitch_shift, f0_method, index_rate, protect_rate)`, `SVCResult(audio_path, processing_time)`
- `B3.11` Create pipeline models. Output: `packages/soundlab/src/soundlab/pipeline/models.py`. Spec: `PipelineConfig`, `QAConfig`, `CandidatePlan`, `CandidateScore`, `StageCheckpoint`, `StemQAResult`, `MIDIQAResult`, `RunArtifacts` per PRD §4.6

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
