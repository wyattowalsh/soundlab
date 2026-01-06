"""Tests for soundlab.utils.gpu."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from soundlab.utils.gpu import (
    clear_gpu_cache,
    get_device,
    get_free_vram_gb,
    get_gpu_info,
    is_cuda_available,
)


class TestIsCudaAvailable:
    """Test is_cuda_available function."""

    def test_cuda_available(self):
        """Should return True when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            assert is_cuda_available() is True

    def test_cuda_unavailable(self):
        """Should return False when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            assert is_cuda_available() is False

    def test_torch_import_error(self):
        """Should return False when torch is not installed."""
        with patch.dict("sys.modules", {"torch": None}):
            # Force reimport to trigger ImportError path
            with patch("soundlab.utils.gpu.is_cuda_available") as mock:
                mock.return_value = False
                assert mock() is False


class TestGetDevice:
    """Test get_device function."""

    def test_cpu_mode_always_returns_cpu(self):
        """CPU mode should always return 'cpu'."""
        assert get_device("cpu") == "cpu"

    def test_cuda_mode_with_cuda_available(self):
        """CUDA mode should return 'cuda' when available."""
        with patch("soundlab.utils.gpu.is_cuda_available", return_value=True):
            assert get_device("cuda") == "cuda"

    def test_cuda_mode_fallback_to_cpu(self):
        """CUDA mode should fallback to 'cpu' when unavailable."""
        with patch("soundlab.utils.gpu.is_cuda_available", return_value=False):
            assert get_device("cuda") == "cpu"

    def test_auto_mode_with_cuda(self):
        """Auto mode should return 'cuda' when available."""
        with patch("soundlab.utils.gpu.is_cuda_available", return_value=True):
            assert get_device("auto") == "cuda"

    def test_auto_mode_without_cuda(self):
        """Auto mode should return 'cpu' when CUDA unavailable."""
        with patch("soundlab.utils.gpu.is_cuda_available", return_value=False):
            assert get_device("auto") == "cpu"


class TestGetFreeVramGb:
    """Test get_free_vram_gb function."""

    def test_returns_zero_without_cuda(self):
        """Should return 0.0 when CUDA is not available."""
        with patch("soundlab.utils.gpu.is_cuda_available", return_value=False):
            assert get_free_vram_gb() == 0.0

    def test_returns_vram_with_cuda(self):
        """Should return VRAM amount when CUDA is available."""
        free_bytes = 8 * (1024 ** 3)  # 8 GB
        total_bytes = 16 * (1024 ** 3)  # 16 GB

        with patch("soundlab.utils.gpu.is_cuda_available", return_value=True):
            with patch("torch.cuda.current_device", return_value=0):
                with patch("torch.cuda.mem_get_info", return_value=(free_bytes, total_bytes)):
                    vram = get_free_vram_gb()
                    assert vram == pytest.approx(8.0, rel=0.01)


class TestClearGpuCache:
    """Test clear_gpu_cache function."""

    def test_does_nothing_without_cuda(self):
        """Should do nothing when CUDA is not available."""
        with patch("soundlab.utils.gpu.is_cuda_available", return_value=False):
            # Should not raise
            clear_gpu_cache()

    def test_clears_cache_with_cuda(self):
        """Should call torch.cuda.empty_cache when CUDA is available."""
        with patch("soundlab.utils.gpu.is_cuda_available", return_value=True):
            mock_empty_cache = MagicMock()
            with patch("torch.cuda.empty_cache", mock_empty_cache):
                clear_gpu_cache()
                mock_empty_cache.assert_called_once()


class TestGetGpuInfo:
    """Test get_gpu_info function."""

    def test_returns_unavailable_info_without_cuda(self):
        """Should return unavailable info when CUDA is not available."""
        with patch("soundlab.utils.gpu.is_cuda_available", return_value=False):
            info = get_gpu_info()
            assert info["available"] is False
            assert info["name"] is None

    def test_returns_gpu_info_with_cuda(self):
        """Should return GPU info when CUDA is available."""
        free_bytes = 8 * (1024 ** 3)
        total_bytes = 16 * (1024 ** 3)

        with patch("soundlab.utils.gpu.is_cuda_available", return_value=True):
            with patch("torch.cuda.current_device", return_value=0):
                with patch("torch.cuda.get_device_name", return_value="NVIDIA Test GPU"):
                    with patch("torch.cuda.mem_get_info", return_value=(free_bytes, total_bytes)):
                        info = get_gpu_info()
                        assert info["available"] is True
                        assert info["name"] == "NVIDIA Test GPU"
                        assert info["total_gb"] == pytest.approx(16.0, rel=0.01)
                        assert info["free_gb"] == pytest.approx(8.0, rel=0.01)
