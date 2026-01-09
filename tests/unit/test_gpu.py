"""Tests for GPU utilities."""

import soundlab.utils.gpu as gpu


class _FakeCuda:
    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.cleared = False

    def is_available(self) -> bool:
        return self._available

    def mem_get_info(self) -> tuple[int, int]:
        return (3 * 1024**3, 6 * 1024**3)

    def empty_cache(self) -> None:
        self.cleared = True


class _FakeTorch:
    def __init__(self, available: bool = True) -> None:
        self.cuda = _FakeCuda(available)


def test_get_device_auto_uses_cuda_when_available(monkeypatch: object) -> None:
    monkeypatch.setattr(gpu, "torch", _FakeTorch(True))
    monkeypatch.delenv("SOUNDLAB_GPU_MODE", raising=False)

    assert gpu.get_device("auto") == "cuda"


def test_get_device_respects_env_override(monkeypatch: object) -> None:
    monkeypatch.setattr(gpu, "torch", _FakeTorch(True))
    monkeypatch.setenv("SOUNDLAB_GPU_MODE", "force_cpu")

    assert gpu.get_device("auto") == "cpu"


def test_get_device_falls_back_when_unavailable(monkeypatch: object) -> None:
    monkeypatch.setattr(gpu, "torch", _FakeTorch(False))
    monkeypatch.delenv("SOUNDLAB_GPU_MODE", raising=False)

    assert gpu.is_cuda_available() is False
    assert gpu.get_device("force_gpu") == "cpu"


def test_get_free_vram_and_clear_cache(monkeypatch: object) -> None:
    fake_torch = _FakeTorch(True)
    monkeypatch.setattr(gpu, "torch", fake_torch)

    free_gb = gpu.get_free_vram_gb()
    assert free_gb > 0

    gpu.clear_gpu_cache()
    assert fake_torch.cuda.cleared is True

    monkeypatch.setattr(gpu, "torch", None)
    assert gpu.get_free_vram_gb() == 0.0
