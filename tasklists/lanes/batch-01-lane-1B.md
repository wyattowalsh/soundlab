# SoundLab Task Lane: Batch 1 Lane 1B - Utils (GPU, logging, retry)

**Batch:** 1
**Lane:** 1B
**Title:** Utils (GPU, logging, retry)
**Dependencies:** B0 complete (directory structure exists)
**Barrier:** All B1 tasks must complete before B2

## Tasks
- `B1.06` Create GPU utilities. Output: `packages/soundlab/src/soundlab/utils/gpu.py`. Spec: `get_device(mode: str) -> str`, `get_free_vram_gb() -> float`, `clear_gpu_cache()`, `is_cuda_available() -> bool`
- `B1.07` Create logging utilities. Output: `packages/soundlab/src/soundlab/utils/logging.py`. Spec: `configure_logging(level: str, log_file: Path | None)` using loguru; format with timestamp, level, module
- `B1.08` Create retry utilities. Output: `packages/soundlab/src/soundlab/utils/retry.py`. Spec: Tenacity decorators: `io_retry`, `gpu_retry`, `network_retry` with configs per PRD §10.2

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
