"""Tests for soundlab.core.exceptions."""

from __future__ import annotations

import pytest

from soundlab.core.exceptions import (
    AudioFormatError,
    AudioLoadError,
    ConfigurationError,
    GPUMemoryError,
    ModelNotFoundError,
    ProcessingError,
    SoundLabError,
    VoiceConversionError,
)


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_soundlab_error_is_base(self):
        """SoundLabError should be the base exception."""
        assert issubclass(SoundLabError, Exception)

    def test_audio_load_error_inherits(self):
        """AudioLoadError should inherit from SoundLabError."""
        assert issubclass(AudioLoadError, SoundLabError)

    def test_audio_format_error_inherits(self):
        """AudioFormatError should inherit from SoundLabError."""
        assert issubclass(AudioFormatError, SoundLabError)

    def test_model_not_found_error_inherits(self):
        """ModelNotFoundError should inherit from SoundLabError."""
        assert issubclass(ModelNotFoundError, SoundLabError)

    def test_gpu_memory_error_inherits(self):
        """GPUMemoryError should inherit from SoundLabError."""
        assert issubclass(GPUMemoryError, SoundLabError)

    def test_processing_error_inherits(self):
        """ProcessingError should inherit from SoundLabError."""
        assert issubclass(ProcessingError, SoundLabError)

    def test_configuration_error_inherits(self):
        """ConfigurationError should inherit from SoundLabError."""
        assert issubclass(ConfigurationError, SoundLabError)

    def test_voice_conversion_error_inherits(self):
        """VoiceConversionError should inherit from SoundLabError."""
        assert issubclass(VoiceConversionError, SoundLabError)


class TestExceptionRaising:
    """Test that exceptions can be raised and caught properly."""

    def test_raise_soundlab_error(self):
        """Should be able to raise and catch SoundLabError."""
        with pytest.raises(SoundLabError):
            raise SoundLabError("Test error")

    def test_raise_audio_load_error(self):
        """Should be able to raise AudioLoadError with message."""
        msg = "Failed to load audio file"
        with pytest.raises(AudioLoadError, match=msg):
            raise AudioLoadError(msg)

    def test_raise_gpu_memory_error(self):
        """Should be able to raise GPUMemoryError."""
        with pytest.raises(GPUMemoryError):
            raise GPUMemoryError("Insufficient VRAM")

    def test_catch_derived_as_base(self):
        """Derived exceptions should be catchable as SoundLabError."""
        with pytest.raises(SoundLabError):
            raise AudioLoadError("File not found")

    def test_exception_message(self):
        """Exception should preserve its message."""
        msg = "Test message"
        error = AudioFormatError(msg)
        assert str(error) == msg

    def test_exception_args(self):
        """Exception should preserve its args."""
        error = ProcessingError("Error", "details")
        assert error.args == ("Error", "details")


class TestExceptionUseCases:
    """Test exceptions in realistic use cases."""

    def test_audio_load_error_for_missing_file(self):
        """AudioLoadError should be appropriate for missing files."""
        def load_audio(path: str):
            if not path:
                raise AudioLoadError(f"File not found: {path}")

        with pytest.raises(AudioLoadError):
            load_audio("")

    def test_configuration_error_for_invalid_config(self):
        """ConfigurationError should be appropriate for invalid configs."""
        def validate_config(threshold: float):
            if not 0 <= threshold <= 1:
                raise ConfigurationError(f"Invalid threshold: {threshold}")

        with pytest.raises(ConfigurationError):
            validate_config(1.5)

    def test_chained_exception(self):
        """Should support exception chaining."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ProcessingError("Processing failed") from e
        except ProcessingError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
