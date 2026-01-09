"""Voice module Pydantic model tests - TTS and SVC models."""

from __future__ import annotations

from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")


class TestTTSConfig:
    """Tests for TTSConfig model."""

    def test_default_values(self):
        """TTSConfig has sensible defaults."""
        from soundlab.voice.models import TTSConfig

        config = TTSConfig(text="Hello")
        assert config.language == "en"
        assert config.temperature == 0.7
        assert config.speed == 1.0

    def test_text_required(self):
        """TTSConfig requires text field."""
        from soundlab.voice.models import TTSConfig

        with pytest.raises(pydantic.ValidationError):
            TTSConfig()

    def test_temperature_min_bound(self):
        """temperature rejects values below 0.0."""
        from soundlab.voice.models import TTSConfig

        with pytest.raises(pydantic.ValidationError) as exc_info:
            TTSConfig(text="Hello", temperature=-0.1)
        assert "temperature" in str(exc_info.value)

    def test_temperature_max_bound(self):
        """temperature rejects values above 1.0."""
        from soundlab.voice.models import TTSConfig

        with pytest.raises(pydantic.ValidationError) as exc_info:
            TTSConfig(text="Hello", temperature=1.1)
        assert "temperature" in str(exc_info.value)

    def test_temperature_boundary_values(self):
        """temperature accepts boundary values 0.0 and 1.0."""
        from soundlab.voice.models import TTSConfig

        config_min = TTSConfig(text="Test", temperature=0.0)
        config_max = TTSConfig(text="Test", temperature=1.0)
        assert config_min.temperature == 0.0
        assert config_max.temperature == 1.0

    def test_speed_min_bound(self):
        """speed rejects values below 0.5."""
        from soundlab.voice.models import TTSConfig

        with pytest.raises(pydantic.ValidationError) as exc_info:
            TTSConfig(text="Hello", speed=0.4)
        assert "speed" in str(exc_info.value)

    def test_speed_max_bound(self):
        """speed rejects values above 2.0."""
        from soundlab.voice.models import TTSConfig

        with pytest.raises(pydantic.ValidationError) as exc_info:
            TTSConfig(text="Hello", speed=2.1)
        assert "speed" in str(exc_info.value)

    def test_speed_boundary_values(self):
        """speed accepts boundary values 0.5 and 2.0."""
        from soundlab.voice.models import TTSConfig

        config_min = TTSConfig(text="Test", speed=0.5)
        config_max = TTSConfig(text="Test", speed=2.0)
        assert config_min.speed == 0.5
        assert config_max.speed == 2.0

    def test_frozen_model(self):
        """TTSConfig is immutable (frozen=True)."""
        from soundlab.voice.models import TTSConfig

        config = TTSConfig(text="Hello")
        with pytest.raises(pydantic.ValidationError):
            config.text = "World"

    def test_speaker_wav_optional(self):
        """speaker_wav is optional (None by default)."""
        from soundlab.voice.models import TTSConfig

        config = TTSConfig(text="Hello")
        assert config.speaker_wav is None

    def test_speaker_wav_accepts_path(self):
        """speaker_wav accepts Path objects."""
        from soundlab.voice.models import TTSConfig

        speaker_path = Path("/path/to/speaker.wav")
        config = TTSConfig(text="Hello", speaker_wav=speaker_path)
        assert config.speaker_wav == speaker_path

    def test_json_serialization(self):
        """TTSConfig serializes to/from JSON."""
        from soundlab.voice.models import TTSConfig

        config = TTSConfig(
            text="Hello world",
            language="es",
            temperature=0.8,
            speed=1.2,
        )
        json_str = config.model_dump_json()
        restored = TTSConfig.model_validate_json(json_str)
        assert restored.text == config.text
        assert restored.language == config.language
        assert restored.temperature == config.temperature
        assert restored.speed == config.speed

    def test_json_serialization_with_speaker_wav(self):
        """TTSConfig with speaker_wav serializes correctly."""
        from soundlab.voice.models import TTSConfig

        config = TTSConfig(
            text="Test",
            speaker_wav=Path("/tmp/ref.wav"),
        )
        json_str = config.model_dump_json()
        restored = TTSConfig.model_validate_json(json_str)
        assert restored.speaker_wav == config.speaker_wav


class TestTTSResult:
    """Tests for TTSResult model."""

    def test_construction(self):
        """TTSResult can be constructed with required fields."""
        from soundlab.voice.models import TTSResult

        result = TTSResult(
            audio_path=Path("/output/speech.wav"),
            processing_time=2.5,
        )
        assert result.audio_path == Path("/output/speech.wav")
        assert result.processing_time == 2.5

    def test_processing_time_non_negative(self):
        """processing_time must be >= 0."""
        from soundlab.voice.models import TTSResult

        with pytest.raises(pydantic.ValidationError) as exc_info:
            TTSResult(
                audio_path=Path("/output/speech.wav"),
                processing_time=-1.0,
            )
        assert "processing_time" in str(exc_info.value)

    def test_processing_time_zero_allowed(self):
        """processing_time accepts 0.0."""
        from soundlab.voice.models import TTSResult

        result = TTSResult(
            audio_path=Path("/output/speech.wav"),
            processing_time=0.0,
        )
        assert result.processing_time == 0.0

    def test_audio_path_stored(self):
        """audio_path is stored correctly."""
        from soundlab.voice.models import TTSResult

        path = Path("/custom/path/to/audio.wav")
        result = TTSResult(audio_path=path, processing_time=1.0)
        assert result.audio_path == path

    def test_audio_path_required(self):
        """audio_path is a required field."""
        from soundlab.voice.models import TTSResult

        with pytest.raises(pydantic.ValidationError):
            TTSResult(processing_time=1.0)

    def test_frozen_model(self):
        """TTSResult is immutable."""
        from soundlab.voice.models import TTSResult

        result = TTSResult(
            audio_path=Path("/output/speech.wav"),
            processing_time=1.0,
        )
        with pytest.raises(pydantic.ValidationError):
            result.audio_path = Path("/other/path.wav")

    def test_json_serialization(self):
        """TTSResult serializes to/from JSON."""
        from soundlab.voice.models import TTSResult

        result = TTSResult(
            audio_path=Path("/output/speech.wav"),
            processing_time=3.14,
        )
        json_str = result.model_dump_json()
        restored = TTSResult.model_validate_json(json_str)
        assert restored.audio_path == result.audio_path
        assert restored.processing_time == result.processing_time


# ---------------------------------------------------------------------------
# SVCConfig Tests
# ---------------------------------------------------------------------------


class TestSVCConfig:
    """Tests for SVCConfig model."""

    def test_default_values(self):
        """SVCConfig has sensible defaults."""
        from soundlab.voice.models import SVCConfig

        config = SVCConfig()
        assert config.pitch_shift == 0.0
        assert config.f0_method == "rmvpe"
        assert config.index_rate == 0.0
        assert config.protect_rate == 0.5

    def test_pitch_shift_min_bound(self):
        """pitch_shift rejects values below -24."""
        from soundlab.voice.models import SVCConfig

        with pytest.raises(pydantic.ValidationError) as exc_info:
            SVCConfig(pitch_shift=-25.0)
        assert "pitch_shift" in str(exc_info.value)

    def test_pitch_shift_max_bound(self):
        """pitch_shift rejects values above 24."""
        from soundlab.voice.models import SVCConfig

        with pytest.raises(pydantic.ValidationError) as exc_info:
            SVCConfig(pitch_shift=25.0)
        assert "pitch_shift" in str(exc_info.value)

    def test_pitch_shift_boundary_values(self):
        """pitch_shift accepts boundary values -24 and 24."""
        from soundlab.voice.models import SVCConfig

        config_min = SVCConfig(pitch_shift=-24.0)
        config_max = SVCConfig(pitch_shift=24.0)
        assert config_min.pitch_shift == -24.0
        assert config_max.pitch_shift == 24.0

    def test_index_rate_min_bound(self):
        """index_rate rejects values below 0.0."""
        from soundlab.voice.models import SVCConfig

        with pytest.raises(pydantic.ValidationError) as exc_info:
            SVCConfig(index_rate=-0.1)
        assert "index_rate" in str(exc_info.value)

    def test_index_rate_max_bound(self):
        """index_rate rejects values above 1.0."""
        from soundlab.voice.models import SVCConfig

        with pytest.raises(pydantic.ValidationError) as exc_info:
            SVCConfig(index_rate=1.1)
        assert "index_rate" in str(exc_info.value)

    def test_index_rate_boundary_values(self):
        """index_rate accepts boundary values 0.0 and 1.0."""
        from soundlab.voice.models import SVCConfig

        config_min = SVCConfig(index_rate=0.0)
        config_max = SVCConfig(index_rate=1.0)
        assert config_min.index_rate == 0.0
        assert config_max.index_rate == 1.0

    def test_protect_rate_min_bound(self):
        """protect_rate rejects values below 0.0."""
        from soundlab.voice.models import SVCConfig

        with pytest.raises(pydantic.ValidationError) as exc_info:
            SVCConfig(protect_rate=-0.1)
        assert "protect_rate" in str(exc_info.value)

    def test_protect_rate_max_bound(self):
        """protect_rate rejects values above 1.0."""
        from soundlab.voice.models import SVCConfig

        with pytest.raises(pydantic.ValidationError) as exc_info:
            SVCConfig(protect_rate=1.1)
        assert "protect_rate" in str(exc_info.value)

    def test_protect_rate_boundary_values(self):
        """protect_rate accepts boundary values 0.0 and 1.0."""
        from soundlab.voice.models import SVCConfig

        config_min = SVCConfig(protect_rate=0.0)
        config_max = SVCConfig(protect_rate=1.0)
        assert config_min.protect_rate == 0.0
        assert config_max.protect_rate == 1.0

    def test_f0_method_valid_values(self):
        """f0_method accepts valid string values."""
        from soundlab.voice.models import SVCConfig

        # Valid f0 methods commonly used in RVC
        valid_methods = ["rmvpe", "harvest", "crepe", "dio", "pm"]
        for method in valid_methods:
            config = SVCConfig(f0_method=method)
            assert config.f0_method == method

    def test_f0_method_default(self):
        """f0_method defaults to rmvpe."""
        from soundlab.voice.models import SVCConfig

        config = SVCConfig()
        assert config.f0_method == "rmvpe"

    def test_frozen_model(self):
        """SVCConfig is immutable (frozen=True)."""
        from soundlab.voice.models import SVCConfig

        config = SVCConfig()
        with pytest.raises(pydantic.ValidationError):
            config.pitch_shift = 5.0

    def test_json_serialization(self):
        """SVCConfig serializes to/from JSON."""
        from soundlab.voice.models import SVCConfig

        config = SVCConfig(
            pitch_shift=12.0,
            f0_method="harvest",
            index_rate=0.75,
            protect_rate=0.25,
        )
        json_str = config.model_dump_json()
        restored = SVCConfig.model_validate_json(json_str)
        assert restored.pitch_shift == config.pitch_shift
        assert restored.f0_method == config.f0_method
        assert restored.index_rate == config.index_rate
        assert restored.protect_rate == config.protect_rate

    def test_all_params_custom(self):
        """SVCConfig accepts all custom parameters."""
        from soundlab.voice.models import SVCConfig

        config = SVCConfig(
            pitch_shift=-12.0,
            f0_method="crepe",
            index_rate=0.8,
            protect_rate=0.1,
        )
        assert config.pitch_shift == -12.0
        assert config.f0_method == "crepe"
        assert config.index_rate == 0.8
        assert config.protect_rate == 0.1


# ---------------------------------------------------------------------------
# SVCResult Tests
# ---------------------------------------------------------------------------


class TestSVCResult:
    """Tests for SVCResult model."""

    def test_construction(self):
        """SVCResult can be constructed with required fields."""
        from soundlab.voice.models import SVCResult

        result = SVCResult(
            audio_path=Path("/output/converted.wav"),
            processing_time=5.2,
        )
        assert result.audio_path == Path("/output/converted.wav")
        assert result.processing_time == 5.2

    def test_processing_time_non_negative(self):
        """processing_time must be >= 0."""
        from soundlab.voice.models import SVCResult

        with pytest.raises(pydantic.ValidationError) as exc_info:
            SVCResult(
                audio_path=Path("/output/converted.wav"),
                processing_time=-0.5,
            )
        assert "processing_time" in str(exc_info.value)

    def test_processing_time_zero_allowed(self):
        """processing_time accepts 0.0."""
        from soundlab.voice.models import SVCResult

        result = SVCResult(
            audio_path=Path("/output/converted.wav"),
            processing_time=0.0,
        )
        assert result.processing_time == 0.0

    def test_audio_path_stored(self):
        """audio_path is stored correctly."""
        from soundlab.voice.models import SVCResult

        path = Path("/custom/output/voice.wav")
        result = SVCResult(audio_path=path, processing_time=1.0)
        assert result.audio_path == path

    def test_audio_path_required(self):
        """audio_path is a required field."""
        from soundlab.voice.models import SVCResult

        with pytest.raises(pydantic.ValidationError):
            SVCResult(processing_time=1.0)

    def test_audio_path_from_string(self):
        """audio_path accepts string paths (coerced to Path)."""
        from soundlab.voice.models import SVCResult

        result = SVCResult(
            audio_path="/output/converted.wav",
            processing_time=1.0,
        )
        assert result.audio_path == Path("/output/converted.wav")

    def test_frozen_model(self):
        """SVCResult is immutable (frozen=True)."""
        from soundlab.voice.models import SVCResult

        result = SVCResult(
            audio_path=Path("/output/converted.wav"),
            processing_time=1.0,
        )
        with pytest.raises(pydantic.ValidationError):
            result.audio_path = Path("/other/path.wav")

    def test_json_serialization(self):
        """SVCResult serializes to/from JSON."""
        from soundlab.voice.models import SVCResult

        result = SVCResult(
            audio_path=Path("/output/voice.wav"),
            processing_time=7.89,
        )
        json_str = result.model_dump_json()
        restored = SVCResult.model_validate_json(json_str)
        assert restored.audio_path == result.audio_path
        assert restored.processing_time == result.processing_time

    def test_required_fields_missing(self):
        """SVCResult requires both audio_path and processing_time."""
        from soundlab.voice.models import SVCResult

        with pytest.raises(pydantic.ValidationError):
            SVCResult()

        with pytest.raises(pydantic.ValidationError):
            SVCResult(audio_path=Path("/output.wav"))


# ---------------------------------------------------------------------------
# Cross-Model Integration Tests
# ---------------------------------------------------------------------------


class TestVoiceModelsIntegration:
    """Integration tests for voice models."""

    def test_all_models_exported(self):
        """All voice models are exported from __all__."""
        from soundlab.voice import models

        assert "TTSConfig" in models.__all__
        assert "TTSResult" in models.__all__
        assert "SVCConfig" in models.__all__
        assert "SVCResult" in models.__all__

    def test_tts_workflow(self):
        """TTS config and result can be used together."""
        from soundlab.voice.models import TTSConfig, TTSResult

        config = TTSConfig(
            text="Hello, world!",
            language="en",
            temperature=0.6,
            speed=1.1,
        )
        result = TTSResult(
            audio_path=Path("/output/speech.wav"),
            processing_time=1.5,
        )

        assert config.text == "Hello, world!"
        assert result.audio_path.suffix == ".wav"

    def test_svc_workflow(self):
        """SVC config and result can be used together."""
        from soundlab.voice.models import SVCConfig, SVCResult

        config = SVCConfig(
            pitch_shift=7.0,
            f0_method="rmvpe",
            index_rate=0.6,
            protect_rate=0.4,
        )
        result = SVCResult(
            audio_path=Path("/output/converted.wav"),
            processing_time=3.2,
        )

        assert config.pitch_shift == 7.0
        assert result.processing_time > 0
