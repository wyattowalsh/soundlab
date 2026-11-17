"""Integration tests for stem separation pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import soundfile as sf

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from typing import TYPE_CHECKING

from soundlab.core.exceptions import ProcessingError
from soundlab.separation import DemucsModel, SeparationConfig, StemResult, StemSeparator

if TYPE_CHECKING:
    from pathlib import Path

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


def _source_tensor(sources: list[str], frames: int = 32) -> torch.Tensor:
    """Build stems whose constant value identifies the model.sources index."""
    stems = torch.zeros(len(sources), 2, frames, dtype=torch.float32)
    for index in range(len(sources)):
        stems[index] = (index + 1) / 10
    return stems


def _separate(
    tmp_path: Path,
    sources: list[str] | None,
    stems: torch.Tensor,
    *,
    demucs_model: DemucsModel = DemucsModel.HTDEMUCS,
    two_stems: str | None = None,
) -> tuple[StemResult, Path]:
    """Run StemSeparator against a mock checkpoint."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    audio_path = input_dir / "test_audio.wav"
    sf.write(audio_path, np.zeros((16, 2), dtype=np.float32), 44100, subtype="FLOAT")

    mock_model = MagicMock()
    mock_model.samplerate = 44100
    mock_model.audio_channels = 2
    mock_model.sources = None if sources is None else list(sources)
    waveform = torch.zeros(2, int(stems.shape[-1]), dtype=torch.float32)
    with (
        patch("demucs.pretrained.get_model", return_value=mock_model),
        patch("demucs.audio.AudioFile") as audio_file,
        patch("demucs.apply.apply_model", return_value=stems[None]),
        patch("soundlab.utils.gpu.get_device", return_value="cpu"),
    ):
        audio_file.return_value.read.return_value = waveform
        config = SeparationConfig(
            model=demucs_model,
            device="cpu",
            two_stems=two_stems,
        )
        result = StemSeparator(config=config).separate(audio_path, output_dir)

    assert isinstance(result, StemResult)
    assert result.source_path == audio_path
    assert result.config == config
    assert result.processing_time_seconds >= 0
    return result, output_dir


def _assert_constant(path: Path, value: float) -> None:
    audio, sample_rate = sf.read(path, always_2d=True)
    assert sample_rate == 44100
    np.testing.assert_allclose(audio, value, atol=1e-5)


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
    ) -> None:
        """Saved stems follow model.sources order and carry that index's audio."""
        sources = list(mock_demucs_model.sources)
        result, _output_dir = _separate(tmp_path, sources, _source_tensor(sources))

        assert list(result.stems) == sources
        for index, name in enumerate(sources):
            assert result.stems[name].is_file()
            _assert_constant(result.stems[name], (index + 1) / 10)

    def test_separate_with_two_stems_mode(
        self,
        tmp_path: Path,
        mock_demucs_model: MagicMock,
    ) -> None:
        """Vocals-only indexes model.sources and keeps the complement as no_vocals."""
        sources = list(mock_demucs_model.sources)
        result, output_dir = _separate(
            tmp_path,
            sources,
            _source_tensor(sources),
            two_stems="vocals",
        )
        vocals_index = sources.index("vocals")
        complement = sum((index + 1) / 10 for index in range(len(sources)) if index != vocals_index)

        assert list(result.stems) == ["vocals", "no_vocals"]
        _assert_constant(result.stems["vocals"], (vocals_index + 1) / 10)
        _assert_constant(result.stems["no_vocals"], complement)
        assert result.instrumental == result.stems["no_vocals"]
        assert result.instrumental is not None
        assert result.instrumental.is_file()
        assert not (output_dir / "instrumental.wav").exists()

    def test_two_stems_uses_the_same_source_list(
        self,
        tmp_path: Path,
        mock_demucs_model: MagicMock,
    ) -> None:
        """A non-vocal two-stem target uses the same model.sources index."""
        sources = list(mock_demucs_model.sources)
        result, _output_dir = _separate(
            tmp_path,
            sources,
            _source_tensor(sources),
            two_stems="bass",
        )
        bass_index = sources.index("bass")
        complement = sum((index + 1) / 10 for index in range(len(sources)) if index != bass_index)

        assert list(result.stems) == ["bass", "no_bass"]
        _assert_constant(result.stems["bass"], (bass_index + 1) / 10)
        _assert_constant(result.stems["no_bass"], complement)

    def test_six_stem_names_follow_model_sources(self, tmp_path: Path) -> None:
        """htdemucs_6s output order is the checkpoint sources, not the enum catalog."""
        sources = ["guitar", "piano", "vocals", "other", "bass", "drums"]
        assert sources != list(DemucsModel.HTDEMUCS_6S.stems)

        result, _output_dir = _separate(
            tmp_path,
            sources,
            _source_tensor(sources),
            demucs_model=DemucsModel.HTDEMUCS_6S,
        )

        assert list(result.stems) == sources
        for index, name in enumerate(sources):
            _assert_constant(result.stems[name], (index + 1) / 10)

    def test_separate_rejects_source_count_mismatch(self, tmp_path: Path) -> None:
        """A tensor count that disagrees with model.sources raises ProcessingError."""
        with pytest.raises(ProcessingError, match=r"model\.sources"):
            _separate(
                tmp_path,
                ["drums", "bass", "other"],
                _source_tensor(["drums", "bass", "other", "vocals"]),
            )
        assert list((tmp_path / "output").glob("*.wav")) == []

    def test_separate_rejects_checkpoint_without_sources(self, tmp_path: Path) -> None:
        """A checkpoint with no model.sources raises instead of guessing names."""
        with pytest.raises(ProcessingError, match=r"model\.sources"):
            _separate(
                tmp_path,
                None,
                _source_tensor(["drums", "bass", "other", "vocals"]),
            )

    def test_separate_rejects_unknown_two_stems(
        self,
        tmp_path: Path,
        mock_demucs_model: MagicMock,
    ) -> None:
        """two_stems must name a source from the loaded model."""
        sources = list(mock_demucs_model.sources)
        with pytest.raises(ProcessingError, match="Invalid two_stems"):
            _separate(tmp_path, sources, _source_tensor(sources), two_stems="piano")

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
