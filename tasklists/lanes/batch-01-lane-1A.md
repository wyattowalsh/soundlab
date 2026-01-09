# SoundLab Task Lane: Batch 1 Lane 1A - Core models and config

**Batch:** 1
**Lane:** 1A
**Title:** Core models and config
**Dependencies:** B0 complete (directory structure exists)
**Barrier:** All B1 tasks must complete before B2

## Tasks
- `B1.01` Create exceptions module. Output: `packages/soundlab/src/soundlab/core/exceptions.py`. Spec: Full exception hierarchy per PRD §4.1: `SoundLabError`, `AudioLoadError`, `AudioFormatError`, `ModelNotFoundError`, `GPUMemoryError`, `ProcessingError`, `ConfigurationError`, `VoiceConversionError`
- `B1.02` Create types module. Output: `packages/soundlab/src/soundlab/core/types.py`. Spec: Type aliases: `AudioArray = NDArray[np.float32]`, `SampleRate = int`, `PathLike = str | Path`; Protocols: `ProgressCallback`, `AudioProcessor`
- `B1.03` Create audio models. Output: `packages/soundlab/src/soundlab/core/audio.py`. Spec: `AudioFormat`, `SampleRate`, `BitDepth` enums; `AudioMetadata`, `AudioSegment` Pydantic models per PRD §4.2
- `B1.04` Create config module. Output: `packages/soundlab/src/soundlab/core/config.py`. Spec: `SoundLabConfig` singleton with env var loading: `SOUNDLAB_LOG_LEVEL`, `SOUNDLAB_GPU_MODE`, `SOUNDLAB_CACHE_DIR`, `SOUNDLAB_OUTPUT_DIR`
- `B1.05` Create core __init__. Output: `packages/soundlab/src/soundlab/core/__init__.py`. Spec: Public exports: all exceptions, `AudioSegment`, `AudioMetadata`, `AudioFormat`, `SoundLabConfig`

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
