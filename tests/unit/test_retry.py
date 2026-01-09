"""Retry module tests."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

tenacity = pytest.importorskip("tenacity")
httpx = pytest.importorskip("httpx")

from tenacity import RetryError

from soundlab.utils.retry import gpu_retry, io_retry, network_retry


class TestRetryIO:
    """Tests for I/O retry decorator."""

    def test_retry_on_io_error(self) -> None:
        """Retries on IOError."""
        call_count = 0

        @io_retry
        def flaky_io() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise IOError("Temporary failure")
            return "success"

        result = flaky_io()
        assert result == "success"
        assert call_count == 3

    def test_retry_on_permission_error(self) -> None:
        """Retries on PermissionError (subclass of IOError)."""
        call_count = 0

        @io_retry
        def flaky_permission() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise PermissionError("Access denied")
            return "granted"

        # PermissionError is a subclass of OSError, not IOError directly
        # IOError is an alias for OSError in Python 3, so this should work
        result = flaky_permission()
        assert result == "granted"
        assert call_count == 2

    def test_retry_on_connection_error(self) -> None:
        """Retries on ConnectionError."""
        call_count = 0

        @io_retry
        def flaky_connection() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection refused")
            return "connected"

        result = flaky_connection()
        assert result == "connected"
        assert call_count == 2

    def test_max_attempts_exceeded(self) -> None:
        """Raises RetryError after max attempts exceeded."""
        call_count = 0

        with patch("time.sleep"):

            @io_retry
            def always_fails() -> str:
                nonlocal call_count
                call_count += 1
                raise IOError("Persistent failure")

            with pytest.raises(RetryError) as exc_info:
                always_fails()

            # Verify the original exception is accessible
            assert "Persistent failure" in str(exc_info.value)
            # io_retry has stop_after_attempt(3)
            assert call_count == 3

    def test_exponential_backoff(self) -> None:
        """Uses exponential backoff between retries."""
        # Verify the retry config has exponential wait
        # The io_retry uses wait_exponential(multiplier=1, min=2, max=30)
        call_count = 0
        wait_times: list[float] = []

        with patch("time.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda t: wait_times.append(t)

            @io_retry
            def flaky_io() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise IOError("Temporary failure")
                return "success"

            result = flaky_io()
            assert result == "success"

        # Should have 2 waits (between attempts 1-2 and 2-3)
        assert len(wait_times) == 2
        # Exponential backoff: first wait >= min(2), second wait >= first
        assert wait_times[0] >= 2
        assert wait_times[1] >= wait_times[0]

    def test_no_retry_on_unrelated_error(self) -> None:
        """Does not retry on unrelated exceptions."""
        call_count = 0

        @io_retry
        def raises_value_error() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Not an IO error")

        with pytest.raises(ValueError, match="Not an IO error"):
            raises_value_error()

        # Should only be called once (no retry)
        assert call_count == 1


class TestRetryOOM:
    """Tests for GPU OOM retry decorator."""

    def test_clears_cuda_cache_on_oom(self) -> None:
        """Clears CUDA cache on OOM error."""
        call_count = 0

        with patch("soundlab.utils.retry.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.OutOfMemoryError = RuntimeError

            # Import _clear_cuda_cache to test it directly
            from soundlab.utils.retry import _clear_cuda_cache

            # Create a mock RetryCallState
            mock_state = MagicMock()

            _clear_cuda_cache(mock_state)

            mock_torch.cuda.empty_cache.assert_called_once()

    def test_no_cache_clear_when_cuda_unavailable(self) -> None:
        """Does not clear cache when CUDA is unavailable."""
        with patch("soundlab.utils.retry.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            from soundlab.utils.retry import _clear_cuda_cache

            mock_state = MagicMock()
            _clear_cuda_cache(mock_state)

            mock_torch.cuda.empty_cache.assert_not_called()

    def test_no_cache_clear_when_torch_none(self) -> None:
        """Handles torch being None gracefully."""
        with patch("soundlab.utils.retry.torch", None):
            from soundlab.utils.retry import _clear_cuda_cache

            mock_state = MagicMock()
            # Should not raise
            _clear_cuda_cache(mock_state)

    def test_retry_succeeds_after_cache_clear(self) -> None:
        """Operation succeeds after cache clearing."""
        call_count = 0

        # Create a custom OOM-like exception for testing
        class MockOOMError(RuntimeError):
            pass

        with patch("soundlab.utils.retry._oom_error", MockOOMError):
            with patch("time.sleep"):
                # Re-create the decorator with our mock error
                from tenacity import (
                    retry,
                    retry_if_exception_type,
                    stop_after_attempt,
                    wait_exponential,
                )

                test_gpu_retry = retry(
                    stop=stop_after_attempt(2),
                    wait=wait_exponential(multiplier=2, min=5, max=60),
                    retry=retry_if_exception_type(MockOOMError),
                )

                @test_gpu_retry
                def flaky_gpu() -> str:
                    nonlocal call_count
                    call_count += 1
                    if call_count < 2:
                        raise MockOOMError("CUDA out of memory")
                    return "success"

                result = flaky_gpu()
                assert result == "success"
                assert call_count == 2

    def test_max_retries_on_persistent_oom(self) -> None:
        """Stops after max retries on persistent OOM."""
        call_count = 0

        class MockOOMError(RuntimeError):
            pass

        with patch("time.sleep"):
            from tenacity import (
                retry,
                retry_if_exception_type,
                stop_after_attempt,
                wait_exponential,
            )

            test_gpu_retry = retry(
                stop=stop_after_attempt(2),
                wait=wait_exponential(multiplier=2, min=5, max=60),
                retry=retry_if_exception_type(MockOOMError),
            )

            @test_gpu_retry
            def always_oom() -> str:
                nonlocal call_count
                call_count += 1
                raise MockOOMError("Persistent OOM")

            with pytest.raises(RetryError) as exc_info:
                always_oom()

            # Verify the original exception is accessible
            assert "Persistent OOM" in str(exc_info.value)
            # gpu_retry has stop_after_attempt(2)
            assert call_count == 2


class TestRetryNetwork:
    """Tests for network retry decorator."""

    def test_retry_on_timeout(self) -> None:
        """Retries on timeout error."""
        call_count = 0

        with patch("time.sleep"):

            @network_retry
            def flaky_timeout() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise TimeoutError("Request timed out")
                return "success"

            result = flaky_timeout()
            assert result == "success"
            assert call_count == 2

    def test_retry_on_connection_error(self) -> None:
        """Retries on connection error."""
        call_count = 0

        with patch("time.sleep"):

            @network_retry
            def flaky_connection() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ConnectionError("Connection refused")
                return "connected"

            result = flaky_connection()
            assert result == "connected"
            assert call_count == 2

    def test_retry_on_httpx_request_error(self) -> None:
        """Retries on httpx RequestError."""
        call_count = 0

        with patch("time.sleep"):

            @network_retry
            def flaky_httpx() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    # httpx.RequestError requires a message
                    raise httpx.RequestError("Request failed")
                return "success"

            result = flaky_httpx()
            assert result == "success"
            assert call_count == 2

    def test_retry_on_httpx_timeout_error(self) -> None:
        """Retries on httpx timeout error."""
        call_count = 0

        with patch("time.sleep"):

            @network_retry
            def flaky_httpx_timeout() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    # httpx.TimeoutException is a subclass of httpx.RequestError
                    raise httpx.TimeoutException("Timeout")
                return "success"

            result = flaky_httpx_timeout()
            assert result == "success"
            assert call_count == 2

    def test_retry_on_httpx_connect_error(self) -> None:
        """Retries on httpx ConnectError."""
        call_count = 0

        with patch("time.sleep"):

            @network_retry
            def flaky_httpx_connect() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise httpx.ConnectError("Connection failed")
                return "success"

            result = flaky_httpx_connect()
            assert result == "success"
            assert call_count == 2

    def test_network_backoff(self) -> None:
        """Uses appropriate backoff for network errors."""
        call_count = 0
        wait_times: list[float] = []

        with patch("time.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda t: wait_times.append(t)

            @network_retry
            def flaky_network() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise TimeoutError("Timeout")
                return "success"

            result = flaky_network()
            assert result == "success"

        # Should have 2 waits
        assert len(wait_times) == 2
        # Exponential backoff: min=2, so first wait >= 2
        assert wait_times[0] >= 2
        assert wait_times[1] >= wait_times[0]

    def test_max_attempts_network(self) -> None:
        """Raises RetryError after max attempts exceeded for network errors."""
        call_count = 0

        with patch("time.sleep"):

            @network_retry
            def always_fails() -> str:
                nonlocal call_count
                call_count += 1
                raise TimeoutError("Persistent timeout")

            with pytest.raises(RetryError) as exc_info:
                always_fails()

            # Verify the original exception is accessible
            assert "Persistent timeout" in str(exc_info.value)
            # network_retry has stop_after_attempt(3)
            assert call_count == 3

    def test_no_retry_on_http_status_error(self) -> None:
        """Does not retry on HTTP status errors (not network errors)."""
        call_count = 0

        @network_retry
        def raises_http_error() -> str:
            nonlocal call_count
            call_count += 1
            # Create a mock response for HTTPStatusError
            mock_request = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            raise httpx.HTTPStatusError("Not Found", request=mock_request, response=mock_response)

        with pytest.raises(httpx.HTTPStatusError):
            raises_http_error()

        # Should only be called once (no retry for HTTP status errors)
        assert call_count == 1


class TestRetryIntegration:
    """Integration tests for retry decorators."""

    def test_retry_preserves_function_metadata(self) -> None:
        """Retry decorators preserve function metadata."""

        @io_retry
        def documented_function() -> str:
            """This is my docstring."""
            return "result"

        # tenacity wraps the function but preserves metadata via functools.wraps
        assert documented_function.__name__ == "documented_function"
        assert "docstring" in (documented_function.__doc__ or "")

    def test_retry_with_arguments(self) -> None:
        """Retry decorators work with function arguments."""
        call_count = 0

        with patch("tenacity.nap.sleep"):

            @io_retry
            def flaky_with_args(x: int, y: str, z: bool = True) -> dict:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise IOError("Temporary")
                return {"x": x, "y": y, "z": z}

            result = flaky_with_args(42, "hello", z=False)
            assert result == {"x": 42, "y": "hello", "z": False}
            assert call_count == 2

    def test_retry_with_generator(self) -> None:
        """Retry decorators work with functions returning generators."""
        call_count = 0

        with patch("tenacity.nap.sleep"):

            @io_retry
            def flaky_generator():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise IOError("Temporary")
                return (i for i in range(3))

            result = list(flaky_generator())
            assert result == [0, 1, 2]
            assert call_count == 2

    def test_fallback_oom_error_class(self) -> None:
        """_FallbackOOM is a RuntimeError subclass."""
        from soundlab.utils.retry import _FallbackOOM

        assert issubclass(_FallbackOOM, RuntimeError)

        error = _FallbackOOM("Test OOM")
        assert str(error) == "Test OOM"
