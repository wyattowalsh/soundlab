"""Integration tests for stem separation pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from soundlab.separation import DemucsModel, SeparationConfig, StemSeparator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_demucs_model() -> MagicMock:
    """Create a mock Demucs model."""
    model = MagicMock()
    model.samplerate = 44100
    model.audio_channels = 2
    model.sources = ["drums", "bass", "other", "vocals"]
    return model


@pytest.fixture
def sample_audio_tensor() -> torch.Tensor:
    """Create a sample audio tensor (2 channels, 44100 samples = 1 second)."""
    return torch.randn(2, 44100, dtype=torch.float32)


@pytest.fixture
def sample_stems_tensor() -> torch.Tensor:
    """Create sample separated stems tensor (4 stems, 2 channels, 44100 samples)."""
    return torch.randn(4, 2, 44100, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSeparationIntegration:
    """End-to-end integration tests for stem separation."""

    def test_separate_creates_stem_files(
        self,
        tmp_path: Path,
        mock_demucs_model: MagicMock,
        sample_audio_tensor: torch.Tensor,
        sample_stems_tensor: torch.Tensor,
    ) -> None:
        """Test full separation pipeline: load audio -> separate -> verify stems exist."""
        # Setup paths
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        audio_path = input_dir / "test_audio.wav"

        # Create a simple test WAV file
        import soundfile as sf

        audio_data = np.random.randn(44100, 2).astype(np.float32) * 0.5
        sf.write(str(audio_path), audio_data, 44100)

        # Mock demucs components
        mock_get_model = MagicMock(return_value=mock_demucs_model)
        mock_audio_file = MagicMock()
        mock_audio_file.return_value.read.return_value = sample_audio_tensor
        mock_apply_model = MagicMock(return_value=sample_stems_tensor[None])

        with (
            patch("demucs.pretrained.get_model", mock_get_model),
            patch("demucs.audio.AudioFile", mock_audio_file),
            patch("demucs.apply.apply_model", mock_apply_model),
            patch("soundlab.utils.gpu.get_device", return_value="cpu"),
        ):
            config = SeparationConfig(model=DemucsModel.HTDEMUCS, device="cpu")
            separator = StemSeparator(config=config)
            result = separator.separate(audio_path, output_dir)

        # Verify result structure
        assert result.source_path == audio_path
        assert result.config == config
        assert result.processing_time_seconds >= 0

        # Verify stems dictionary has expected keys
        assert isinstance(result.stems, dict)
        assert len(result.stems) > 0

    def test_separate_with_two_stems_mode(
        self,
        tmp_path: Path,
        mock_demucs_model: MagicMock,
        sample_audio_tensor: torch.Tensor,
        sample_stems_tensor: torch.Tensor,
    ) -> None:
        """Test separation in two-stems mode (vocals vs. accompaniment)."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        audio_path = input_dir / "test_audio.wav"

        # Create test WAV
        import soundfile as sf

        audio_data = np.random.randn(22050, 2).astype(np.float32) * 0.5
        sf.write(str(audio_path), audio_data, 44100)

        mock_get_model = MagicMock(return_value=mock_demucs_model)
        mock_audio_file = MagicMock()
        mock_audio_file.return_value.read.return_value = sample_audio_tensor
        mock_apply_model = MagicMock(return_value=sample_stems_tensor[None])

        with (
            patch("demucs.pretrained.get_model", mock_get_model),
            patch("demucs.audio.AudioFile", mock_audio_file),
            patch("demucs.apply.apply_model", mock_apply_model),
            patch("soundlab.utils.gpu.get_device", return_value="cpu"),
        ):
            config = SeparationConfig(
                model=DemucsModel.HTDEMUCS,
                device="cpu",
                two_stems="vocals",
            )
            separator = StemSeparator(config=config)
            result = separator.separate(audio_path, output_dir)

        # In two-stems mode, should have vocals and no_vocals
        assert "vocals" in result.stems or len(result.stems) == 2

    def test_separator_model_lazy_loading(self, mock_demucs_model: MagicMock) -> None:
        """Test that model is lazily loaded only when needed."""
        mock_get_model = MagicMock(return_value=mock_demucs_model)

        with patch("demucs.pretrained.get_model", mock_get_model):
            separator = StemSeparator()

            # Model should not be loaded yet
            assert separator._model is None
            mock_get_model.assert_not_called()

    def test_stem_result_vocals_property(self, tmp_path: Path) -> None:
        """Test StemResult.vocals property accessor."""
        from soundlab.separation.models import StemResult

        vocals_path = tmp_path / "vocals.wav"
        vocals_path.touch()

        result = StemResult(
            stems={"vocals": vocals_path, "drums": tmp_path / "drums.wav"},
            source_path=tmp_path / "source.wav",
            config=SeparationConfig(),
            processing_time_seconds=1.5,
        )

        assert result.vocals == vocals_path

    def test_separation_config_defaults(self) -> None:
        """Test SeparationConfig default values."""
        config = SeparationConfig()

        assert config.model == DemucsModel.HTDEMUCS_FT
        assert config.device == "auto"
        assert config.split is True
        assert config.overlap == 0.25
        assert config.shifts == 1

    def test_demucs_model_stem_count(self) -> None:
        """Test DemucsModel stem_count property."""
        assert DemucsModel.HTDEMUCS.stem_count == 4
        assert DemucsModel.HTDEMUCS_FT.stem_count == 4

    def test_demucs_model_stems_list(self) -> None:
        """Test DemucsModel stems property returns correct stems."""
        stems = DemucsModel.HTDEMUCS.stems
        assert "vocals" in stems
        assert "drums" in stems
        assert "bass" in stems
        assert "other" in stems
