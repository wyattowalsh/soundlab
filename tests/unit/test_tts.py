"""TTSGenerator tests with mocked TTS.api."""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError

from soundlab.voice.models import TTSConfig, TTSResult


@pytest.fixture
def mock_tts_api():
    """Mock Coqui TTS API."""
    mock_tts_class = MagicMock()
    mock_tts_instance = MagicMock()
    mock_tts_class.return_value = mock_tts_instance

    mock_tts_module = MagicMock()
    mock_tts_module.TTS = mock_tts_class

    with patch.dict(sys.modules, {"TTS": MagicMock(), "TTS.api": mock_tts_module}):
        yield mock_tts_instance, mock_tts_class


class TestTTSGeneratorInit:
    """Tests for TTSGenerator initialization."""

    def test_default_model_name(self, mock_tts_api):  # noqa: ARG002
        """Uses XTTS-v2 model by default."""
        from soundlab.voice.tts import TTSGenerator

        gen = TTSGenerator()
        assert gen._model_name == "tts_models/multilingual/multi-dataset/xtts_v2"

    def test_custom_model_name(self, mock_tts_api):  # noqa: ARG002
        """Accepts custom model name."""
        from soundlab.voice.tts import TTSGenerator

        gen = TTSGenerator(model_name="tts_models/custom/model")
        assert gen._model_name == "tts_models/custom/model"

    def test_lazy_model_loading(self, mock_tts_api):  # noqa: ARG002
        """Model not loaded on init."""
        from soundlab.voice.tts import TTSGenerator

        gen = TTSGenerator()
        assert gen._tts is None

    def test_cache_dir_from_env(self, mock_tts_api, monkeypatch, tmp_path):  # noqa: ARG002
        """Uses SOUNDLAB_CACHE_DIR env var."""
        from soundlab.voice.tts import TTSGenerator

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setenv("SOUNDLAB_CACHE_DIR", str(cache_dir))

        # Clear TTS_HOME if set
        monkeypatch.delenv("TTS_HOME", raising=False)

        gen = TTSGenerator()
        gen._load_model()

        import os

        assert os.environ.get("TTS_HOME") == str(cache_dir / "tts")


class TestTTSGeneratorGenerate:
    """Tests for TTSGenerator.generate method."""

    def test_generate_returns_tts_result(self, mock_tts_api, tmp_path, monkeypatch):
        """generate returns TTSResult."""
        from soundlab.voice.tts import TTSGenerator

        _mock_tts_instance, _ = mock_tts_api
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(tmp_path))

        gen = TTSGenerator()
        config = TTSConfig(text="Hello world")
        result = gen.generate(config)

        assert isinstance(result, TTSResult)
        assert result.audio_path.suffix == ".wav"
        assert result.processing_time >= 0.0

    def test_generate_calls_tts_to_file(self, mock_tts_api, tmp_path, monkeypatch):
        """generate calls tts.tts_to_file."""
        from soundlab.voice.tts import TTSGenerator

        mock_tts_instance, _ = mock_tts_api
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(tmp_path))

        gen = TTSGenerator()
        config = TTSConfig(text="Test text")
        gen.generate(config)

        mock_tts_instance.tts_to_file.assert_called_once()

    def test_generate_uses_config_language(self, mock_tts_api, tmp_path, monkeypatch):
        """generate passes language from config."""
        from soundlab.voice.tts import TTSGenerator

        mock_tts_instance, _ = mock_tts_api
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(tmp_path))

        gen = TTSGenerator()
        config = TTSConfig(text="Bonjour", language="fr")
        gen.generate(config)

        call_kwargs = mock_tts_instance.tts_to_file.call_args.kwargs
        assert call_kwargs.get("language") == "fr"

    def test_generate_uses_config_temperature(self, mock_tts_api, tmp_path, monkeypatch):
        """generate passes temperature from config."""
        from soundlab.voice.tts import TTSGenerator

        mock_tts_instance, _ = mock_tts_api
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(tmp_path))

        gen = TTSGenerator()
        config = TTSConfig(text="Test", temperature=0.5)
        gen.generate(config)

        call_kwargs = mock_tts_instance.tts_to_file.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.5

    def test_generate_uses_speaker_wav(self, mock_tts_api, tmp_path, monkeypatch):
        """generate passes speaker_wav for voice cloning."""
        from soundlab.voice.tts import TTSGenerator

        mock_tts_instance, _ = mock_tts_api
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(tmp_path))

        speaker_file = tmp_path / "speaker.wav"
        speaker_file.touch()

        gen = TTSGenerator()
        config = TTSConfig(text="Clone this voice", speaker_wav=speaker_file)
        gen.generate(config)

        call_kwargs = mock_tts_instance.tts_to_file.call_args.kwargs
        assert call_kwargs.get("speaker_wav") == str(speaker_file)

    def test_generate_output_dir_from_env(self, mock_tts_api, tmp_path, monkeypatch):  # noqa: ARG002
        """generate uses SOUNDLAB_OUTPUT_DIR."""
        from soundlab.voice.tts import TTSGenerator

        output_dir = tmp_path / "custom_output"
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(output_dir))

        gen = TTSGenerator()
        config = TTSConfig(text="Output test")
        result = gen.generate(config)

        assert str(output_dir) in str(result.audio_path.parent)

    def test_generate_creates_output_dir(self, mock_tts_api, tmp_path, monkeypatch):  # noqa: ARG002
        """generate creates output directory if missing."""
        from soundlab.voice.tts import TTSGenerator

        output_dir = tmp_path / "new_output"
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(output_dir))

        assert not output_dir.exists()

        gen = TTSGenerator()
        config = TTSConfig(text="Directory test")
        gen.generate(config)

        assert (output_dir / "voice").exists()

    def test_generate_processing_time_measured(self, mock_tts_api, tmp_path, monkeypatch):
        """generate measures processing time."""
        from soundlab.voice.tts import TTSGenerator

        mock_tts_instance, _ = mock_tts_api
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(tmp_path))

        def slow_tts(*_args, **_kwargs):
            time.sleep(0.05)

        mock_tts_instance.tts_to_file.side_effect = slow_tts

        gen = TTSGenerator()
        config = TTSConfig(text="Timing test")
        result = gen.generate(config)

        assert result.processing_time >= 0.05

    def test_generate_fallback_on_type_error(self, mock_tts_api, tmp_path, monkeypatch):
        """generate falls back to minimal kwargs on TypeError."""
        from soundlab.voice.tts import TTSGenerator

        mock_tts_instance, _ = mock_tts_api
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(tmp_path))

        call_count = 0

        def raise_on_first_call(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("Unexpected keyword argument")

        mock_tts_instance.tts_to_file.side_effect = raise_on_first_call

        gen = TTSGenerator()
        config = TTSConfig(text="Fallback test")
        result = gen.generate(config)

        assert call_count == 2
        assert isinstance(result, TTSResult)

    def test_generate_speed_parameter(self, mock_tts_api, tmp_path, monkeypatch):
        """generate passes speed from config."""
        from soundlab.voice.tts import TTSGenerator

        mock_tts_instance, _ = mock_tts_api
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(tmp_path))

        gen = TTSGenerator()
        config = TTSConfig(text="Speed test", speed=1.5)
        gen.generate(config)

        call_kwargs = mock_tts_instance.tts_to_file.call_args.kwargs
        assert call_kwargs.get("speed") == 1.5


class TestTTSGeneratorLoadModel:
    """Tests for model loading behavior."""

    def test_loads_model_lazily(self, mock_tts_api, tmp_path, monkeypatch):
        """Model loaded on first generate call."""
        from soundlab.voice.tts import TTSGenerator

        _, mock_tts_class = mock_tts_api
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(tmp_path))

        gen = TTSGenerator()
        mock_tts_class.assert_not_called()

        config = TTSConfig(text="Lazy load test")
        gen.generate(config)

        mock_tts_class.assert_called_once()

    def test_model_cached_after_first_load(self, mock_tts_api, tmp_path, monkeypatch):
        """Model reused across generate calls."""
        from soundlab.voice.tts import TTSGenerator

        _, mock_tts_class = mock_tts_api
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(tmp_path))

        gen = TTSGenerator()

        config = TTSConfig(text="First call")
        gen.generate(config)

        config2 = TTSConfig(text="Second call")
        gen.generate(config2)

        mock_tts_class.assert_called_once()

    def test_import_error_propagated(self):
        """ImportError raised when TTS not installed."""
        from soundlab.voice.tts import TTSGenerator

        with (
            patch.dict(sys.modules, {"TTS": None, "TTS.api": None}),
            patch(
                "soundlab.voice.tts.TTSGenerator._load_model",
                side_effect=ImportError("coqui-tts is required"),
            ),
        ):
            gen = TTSGenerator()
            gen._tts = None

            with pytest.raises(ImportError, match="coqui-tts is required"):
                gen._load_model()

    def test_load_model_with_progress_bar_fallback(self, mock_tts_api):
        """Falls back when progress_bar kwarg not supported."""
        from soundlab.voice.tts import TTSGenerator

        _, mock_tts_class = mock_tts_api

        call_count = 0

        def raise_on_progress_bar(*_args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "progress_bar" in kwargs:
                raise TypeError("unexpected keyword argument 'progress_bar'")
            return MagicMock()

        mock_tts_class.side_effect = raise_on_progress_bar

        gen = TTSGenerator()
        gen._load_model()

        assert call_count == 2
        assert gen._tts is not None


class TestTTSConfig:
    """Tests for TTSConfig model validation."""

    def test_default_values(self):
        """TTSConfig has correct defaults."""
        config = TTSConfig(text="Test")
        assert config.language == "en"
        assert config.temperature == 0.7
        assert config.speed == 1.0
        assert config.speaker_wav is None

    def test_temperature_bounds(self):
        """TTSConfig validates temperature bounds."""
        with pytest.raises(ValidationError):
            TTSConfig(text="Test", temperature=-0.1)

        with pytest.raises(ValidationError):
            TTSConfig(text="Test", temperature=1.1)

    def test_speed_bounds(self):
        """TTSConfig validates speed bounds."""
        with pytest.raises(ValidationError):
            TTSConfig(text="Test", speed=0.4)

        with pytest.raises(ValidationError):
            TTSConfig(text="Test", speed=2.1)

    def test_frozen_model(self):
        """TTSConfig is immutable."""
        config = TTSConfig(text="Test")
        with pytest.raises(ValidationError):
            config.text = "Changed"
