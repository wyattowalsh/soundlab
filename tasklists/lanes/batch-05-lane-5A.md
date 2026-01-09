# SoundLab Task Lane: Batch 5 Lane 5A - Voice module

**Batch:** 5
**Lane:** 5A
**Title:** Voice module
**Dependencies:** B4 complete (all feature modules available)
**Barrier:** All B5 tasks must complete before B6

## Tasks
- `B5.01` Create TTS wrapper. Output: `packages/soundlab/src/soundlab/voice/tts.py`. Spec: `TTSGenerator` class wrapping coqui-tts XTTS-v2; `generate(config) -> TTSResult`; handle model download
- `B5.02` Create SVC wrapper. Output: `packages/soundlab/src/soundlab/voice/svc.py`. Spec: `VoiceConverter` class for RVC; `convert(audio_path, model_path, config) -> SVCResult`; document manual setup requirements
- `B5.03` Create voice __init__. Output: `packages/soundlab/src/soundlab/voice/__init__.py`. Spec: Public exports: `TTSGenerator`, `TTSConfig`, `VoiceConverter`, `SVCConfig`

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
