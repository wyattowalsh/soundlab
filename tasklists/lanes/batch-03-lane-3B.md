# SoundLab Task Lane: Batch 3 Lane 3B - Effects models

**Batch:** 3
**Lane:** 3B
**Title:** Effects models
**Dependencies:** B2 complete (core + utils + io available)
**Strategy:** Create all Pydantic models first, then implementations
**Barrier:** All B3 tasks must complete before B4

## Tasks
- `B3.03` Create effects models. Output: `packages/soundlab/src/soundlab/effects/models.py`. Spec: Base `EffectConfig` with `to_plugin() -> Plugin` abstract; `CompressorConfig`, `LimiterConfig`, `GateConfig`, `ReverbConfig`, `DelayConfig`, `ChorusConfig`, `DistortionConfig`, `PhaserConfig`, `HighpassConfig`, `LowpassConfig`, `GainConfig`

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
