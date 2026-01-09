"""Tests for core exception hierarchy."""

from soundlab.core import exceptions


def test_exception_hierarchy_and_messages() -> None:
    subclasses = [
        exceptions.AudioLoadError,
        exceptions.AudioFormatError,
        exceptions.ModelNotFoundError,
        exceptions.GPUMemoryError,
        exceptions.ProcessingError,
        exceptions.ConfigurationError,
        exceptions.VoiceConversionError,
    ]

    for exc_type in subclasses:
        assert issubclass(exc_type, exceptions.SoundLabError)
        err = exc_type("boom")
        assert isinstance(err, exceptions.SoundLabError)
        assert str(err) == "boom"
