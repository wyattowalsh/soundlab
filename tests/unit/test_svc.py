"""VoiceConverter tests for singing voice conversion."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic")


class TestVoiceConverterInit:
    """Tests for VoiceConverter initialization."""

    def test_rvc_root_from_parameter(self, tmp_path: Path) -> None:
        """Uses rvc_root from parameter."""
        from soundlab.voice.svc import VoiceConverter

        converter = VoiceConverter(rvc_root=tmp_path)
        assert converter._rvc_root == tmp_path

    def test_rvc_root_from_str_parameter(self, tmp_path: Path) -> None:
        """Accepts string path for rvc_root."""
        from soundlab.voice.svc import VoiceConverter

        converter = VoiceConverter(rvc_root=str(tmp_path))
        assert converter._rvc_root == tmp_path

    def test_rvc_root_from_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uses SOUNDLAB_RVC_ROOT env var when parameter not provided."""
        monkeypatch.setenv("SOUNDLAB_RVC_ROOT", str(tmp_path))
        from soundlab.voice.svc import VoiceConverter

        converter = VoiceConverter()
        assert converter._rvc_root == tmp_path

    def test_rvc_root_parameter_overrides_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parameter takes precedence over env var."""
        from soundlab.voice.svc import VoiceConverter

        env_path = tmp_path / "env_root"
        param_path = tmp_path / "param_root"
        env_path.mkdir()
        param_path.mkdir()

        monkeypatch.setenv("SOUNDLAB_RVC_ROOT", str(env_path))
        converter = VoiceConverter(rvc_root=param_path)
        assert converter._rvc_root == param_path

    def test_rvc_root_none_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """None rvc_root is valid at init (errors on convert)."""
        from soundlab.voice.svc import VoiceConverter

        monkeypatch.delenv("SOUNDLAB_RVC_ROOT", raising=False)
        converter = VoiceConverter(rvc_root=None)
        assert converter._rvc_root is None

    def test_rvc_root_empty_string_treated_as_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty string rvc_root is treated as None."""
        from soundlab.voice.svc import VoiceConverter

        monkeypatch.delenv("SOUNDLAB_RVC_ROOT", raising=False)
        converter = VoiceConverter(rvc_root="")
        assert converter._rvc_root is None


class TestVoiceConverterConvert:
    """Tests for VoiceConverter.convert method."""

    def test_convert_raises_without_rvc_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raises VoiceConversionError when rvc_root is None."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice.models import SVCConfig
        from soundlab.voice.svc import VoiceConverter

        monkeypatch.delenv("SOUNDLAB_RVC_ROOT", raising=False)

        audio_path = tmp_path / "input.wav"
        audio_path.touch()
        model_path = tmp_path / "model.pth"
        model_path.touch()

        converter = VoiceConverter(rvc_root=None)
        config = SVCConfig()

        with pytest.raises(VoiceConversionError) as exc_info:
            converter.convert(audio_path, model_path, config)
        assert "RVC is not configured" in str(exc_info.value)

    def test_convert_raises_source_not_found(self, tmp_path: Path) -> None:
        """Raises error when source audio missing."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice.models import SVCConfig
        from soundlab.voice.svc import VoiceConverter

        rvc_root = tmp_path / "rvc"
        rvc_root.mkdir()
        model_path = tmp_path / "model.pth"
        model_path.touch()
        audio_path = tmp_path / "nonexistent.wav"

        converter = VoiceConverter(rvc_root=rvc_root)
        config = SVCConfig()

        with pytest.raises(VoiceConversionError) as exc_info:
            converter.convert(audio_path, model_path, config)
        assert "Input audio not found" in str(exc_info.value)
        assert "nonexistent.wav" in str(exc_info.value)

    def test_convert_raises_model_not_found(self, tmp_path: Path) -> None:
        """Raises error when model file missing."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice.models import SVCConfig
        from soundlab.voice.svc import VoiceConverter

        rvc_root = tmp_path / "rvc"
        rvc_root.mkdir()
        audio_path = tmp_path / "input.wav"
        audio_path.touch()
        model_path = tmp_path / "nonexistent.pth"

        converter = VoiceConverter(rvc_root=rvc_root)
        config = SVCConfig()

        with pytest.raises(VoiceConversionError) as exc_info:
            converter.convert(audio_path, model_path, config)
        assert "RVC model not found" in str(exc_info.value)
        assert "nonexistent.pth" in str(exc_info.value)

    def test_convert_raises_not_implemented(self, tmp_path: Path) -> None:
        """Raises not-implemented error (current behavior)."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice.models import SVCConfig
        from soundlab.voice.svc import VoiceConverter

        rvc_root = tmp_path / "rvc"
        rvc_root.mkdir()
        audio_path = tmp_path / "input.wav"
        audio_path.touch()
        model_path = tmp_path / "model.pth"
        model_path.touch()

        converter = VoiceConverter(rvc_root=rvc_root)
        config = SVCConfig()

        with pytest.raises(VoiceConversionError) as exc_info:
            converter.convert(audio_path, model_path, config)
        assert "requires manual setup" in str(exc_info.value)

    def test_convert_error_message_helpful(self, tmp_path: Path) -> None:
        """Error message includes setup instructions."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice.models import SVCConfig
        from soundlab.voice.svc import VoiceConverter

        rvc_root = tmp_path / "rvc"
        rvc_root.mkdir()
        audio_path = tmp_path / "input.wav"
        audio_path.touch()
        model_path = tmp_path / "model.pth"
        model_path.touch()

        converter = VoiceConverter(rvc_root=rvc_root)
        config = SVCConfig()

        with pytest.raises(VoiceConversionError) as exc_info:
            converter.convert(audio_path, model_path, config)
        error_msg = str(exc_info.value)
        assert "RVC" in error_msg
        assert "manual setup" in error_msg or "configuration" in error_msg

    def test_convert_accepts_path_objects(self, tmp_path: Path) -> None:
        """Accepts Path objects for audio and model."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice.models import SVCConfig
        from soundlab.voice.svc import VoiceConverter

        rvc_root = tmp_path / "rvc"
        rvc_root.mkdir()
        audio_path = Path(tmp_path / "input.wav")
        audio_path.touch()
        model_path = Path(tmp_path / "model.pth")
        model_path.touch()

        converter = VoiceConverter(rvc_root=rvc_root)
        config = SVCConfig()

        with pytest.raises(VoiceConversionError) as exc_info:
            converter.convert(audio_path, model_path, config)
        assert "requires manual setup" in str(exc_info.value)

    def test_convert_accepts_str_paths(self, tmp_path: Path) -> None:
        """Accepts string paths for audio and model."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice.models import SVCConfig
        from soundlab.voice.svc import VoiceConverter

        rvc_root = tmp_path / "rvc"
        rvc_root.mkdir()
        audio_path = tmp_path / "input.wav"
        audio_path.touch()
        model_path = tmp_path / "model.pth"
        model_path.touch()

        converter = VoiceConverter(rvc_root=rvc_root)
        config = SVCConfig()

        with pytest.raises(VoiceConversionError) as exc_info:
            converter.convert(str(audio_path), str(model_path), config)
        assert "requires manual setup" in str(exc_info.value)

    def test_convert_creates_output_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Creates output directory if it doesn't exist."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice.models import SVCConfig
        from soundlab.voice.svc import VoiceConverter

        output_dir = tmp_path / "custom_outputs"
        monkeypatch.setenv("SOUNDLAB_OUTPUT_DIR", str(output_dir))

        rvc_root = tmp_path / "rvc"
        rvc_root.mkdir()
        audio_path = tmp_path / "input.wav"
        audio_path.touch()
        model_path = tmp_path / "model.pth"
        model_path.touch()

        converter = VoiceConverter(rvc_root=rvc_root)
        config = SVCConfig()

        with pytest.raises(VoiceConversionError):
            converter.convert(audio_path, model_path, config)

        assert (output_dir / "voice").exists()


class TestVoiceConverterErrorOrder:
    """Tests for error checking order in VoiceConverter.convert."""

    def test_rvc_root_checked_before_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RVC root is checked before source file existence."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice.models import SVCConfig
        from soundlab.voice.svc import VoiceConverter

        monkeypatch.delenv("SOUNDLAB_RVC_ROOT", raising=False)

        audio_path = tmp_path / "nonexistent.wav"
        model_path = tmp_path / "nonexistent.pth"

        converter = VoiceConverter(rvc_root=None)
        config = SVCConfig()

        with pytest.raises(VoiceConversionError) as exc_info:
            converter.convert(audio_path, model_path, config)
        assert "RVC is not configured" in str(exc_info.value)

    def test_source_checked_before_model(self, tmp_path: Path) -> None:
        """Source file is checked before model file."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice.models import SVCConfig
        from soundlab.voice.svc import VoiceConverter

        rvc_root = tmp_path / "rvc"
        rvc_root.mkdir()
        audio_path = tmp_path / "nonexistent.wav"
        model_path = tmp_path / "nonexistent.pth"

        converter = VoiceConverter(rvc_root=rvc_root)
        config = SVCConfig()

        with pytest.raises(VoiceConversionError) as exc_info:
            converter.convert(audio_path, model_path, config)
        assert "Input audio not found" in str(exc_info.value)


class TestVoiceConverterWithSVCConfig:
    """Tests for VoiceConverter integration with SVCConfig."""

    def test_convert_accepts_default_config(self, tmp_path: Path) -> None:
        """Convert accepts SVCConfig with default values."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice.models import SVCConfig
        from soundlab.voice.svc import VoiceConverter

        rvc_root = tmp_path / "rvc"
        rvc_root.mkdir()
        audio_path = tmp_path / "input.wav"
        audio_path.touch()
        model_path = tmp_path / "model.pth"
        model_path.touch()

        converter = VoiceConverter(rvc_root=rvc_root)
        config = SVCConfig()

        assert config.pitch_shift == 0.0
        assert config.f0_method == "rmvpe"

        with pytest.raises(VoiceConversionError) as exc_info:
            converter.convert(audio_path, model_path, config)
        assert "requires manual setup" in str(exc_info.value)

    def test_convert_accepts_custom_config(self, tmp_path: Path) -> None:
        """Convert accepts SVCConfig with custom values."""
        from soundlab.core.exceptions import VoiceConversionError
        from soundlab.voice.models import SVCConfig
        from soundlab.voice.svc import VoiceConverter

        rvc_root = tmp_path / "rvc"
        rvc_root.mkdir()
        audio_path = tmp_path / "input.wav"
        audio_path.touch()
        model_path = tmp_path / "model.pth"
        model_path.touch()

        converter = VoiceConverter(rvc_root=rvc_root)
        config = SVCConfig(
            pitch_shift=12.0,
            f0_method="crepe",
            index_rate=0.5,
            protect_rate=0.33,
        )

        with pytest.raises(VoiceConversionError) as exc_info:
            converter.convert(audio_path, model_path, config)
        assert "requires manual setup" in str(exc_info.value)
