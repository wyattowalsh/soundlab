# SoundLab Task Lane: Batch 3 Lane 3C - Analysis models and feature extractors

**Batch:** 3
**Lane:** 3C
**Title:** Analysis models and feature extractors
**Dependencies:** B2 complete (core + utils + io available)
**Strategy:** Create all Pydantic models first, then implementations
**Barrier:** All B3 tasks must complete before B4

## Tasks
- `B3.04` Create analysis models. Output: `packages/soundlab/src/soundlab/analysis/models.py`. Spec: `TempoResult(bpm, confidence, beats)`, `KeyDetectionResult` per PRD §4.4, `LoudnessResult(lufs, dynamic_range, peak)`, `SpectralResult(centroid, bandwidth, rolloff)`, `AnalysisResult` composite
- `B3.08` Create analysis tempo. Output: `packages/soundlab/src/soundlab/analysis/tempo.py`. Spec: `detect_tempo(y, sr) -> TempoResult` using librosa.beat.beat_track
- `B3.09` Create analysis loudness. Output: `packages/soundlab/src/soundlab/analysis/loudness.py`. Spec: `measure_loudness(y, sr) -> LoudnessResult` using pyloudnorm
- `B3.10` Create analysis spectral. Output: `packages/soundlab/src/soundlab/analysis/spectral.py`. Spec: `analyze_spectral(y, sr) -> SpectralResult` using librosa spectral features

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
