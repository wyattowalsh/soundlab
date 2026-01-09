"""Tests for stem separation models."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

pytest.importorskip("pydantic")
pydantic = pytest.importorskip("pydantic")

from soundlab.separation.models import DemucsModel, SeparationConfig, StemResult

# --------------------------------------------------------------------------- #
# DemucsModel Enum Tests
# --------------------------------------------------------------------------- #


class TestDemucsModel:
    """Tests for DemucsModel enum."""

    def test_all_models_exist(self) -> None:
        """All expected Demucs models should be defined."""
        assert DemucsModel.HTDEMUCS == "htdemucs"
        assert DemucsModel.HTDEMUCS_FT == "htdemucs_ft"
        assert DemucsModel.HTDEMUCS_6S == "htdemucs_6s"
        assert DemucsModel.MDX_EXTRA == "mdx_extra"
        assert DemucsModel.MDX_EXTRA_Q == "mdx_extra_q"

    def test_stem_count_4_stem_models(self) -> None:
        """4-stem models should report stem_count of 4."""
        four_stem_models = [
            DemucsModel.HTDEMUCS,
            DemucsModel.HTDEMUCS_FT,
            DemucsModel.MDX_EXTRA,
            DemucsModel.MDX_EXTRA_Q,
        ]

        for model in four_stem_models:
            assert model.stem_count == 4, f"{model} should have stem_count=4"

    def test_stem_count_6_stem_model(self) -> None:
        """6-stem model should report stem_count of 6."""
        assert DemucsModel.HTDEMUCS_6S.stem_count == 6

    def test_stems_4_stem_models(self) -> None:
        """4-stem models should return base stems."""
        expected_stems = ["vocals", "drums", "bass", "other"]

        for model in [DemucsModel.HTDEMUCS, DemucsModel.HTDEMUCS_FT]:
            assert model.stems == expected_stems

    def test_stems_6_stem_model(self) -> None:
        """6-stem model should include piano and guitar."""
        expected_stems = ["vocals", "drums", "bass", "other", "piano", "guitar"]
        assert DemucsModel.HTDEMUCS_6S.stems == expected_stems

    def test_model_is_str_enum(self) -> None:
        """DemucsModel should be a StrEnum for JSON serialization."""
        assert isinstance(DemucsModel.HTDEMUCS, str)
        assert str(DemucsModel.HTDEMUCS) == "htdemucs"

    def test_model_iteration(self) -> None:
        """Should be able to iterate over all models."""
        models = list(DemucsModel)
        assert len(models) == 5


# --------------------------------------------------------------------------- #
# SeparationConfig Tests
# --------------------------------------------------------------------------- #


class TestSeparationConfig:
    """Tests for SeparationConfig model."""

    def test_default_values(self) -> None:
        """SeparationConfig should have sensible defaults."""
        config = SeparationConfig()

        assert config.model == DemucsModel.HTDEMUCS_FT
        assert config.segment_length == 7.8
        assert config.overlap == 0.25
        assert config.shifts == 1
        assert config.two_stems is None
        assert config.float32 is False
        assert config.int24 is True
        assert config.mp3_bitrate == 320
        assert config.device == "auto"
        assert config.split is True

    def test_custom_model(self) -> None:
        """SeparationConfig should accept different models."""
        config = SeparationConfig(model=DemucsModel.HTDEMUCS_6S)
        assert config.model == DemucsModel.HTDEMUCS_6S

    def test_segment_length_bounds(self) -> None:
        """segment_length should be bounded [1.0, 30.0]."""
        # Below minimum
        with pytest.raises(pydantic.ValidationError):
            SeparationConfig(segment_length=0.5)

        # Above maximum
        with pytest.raises(pydantic.ValidationError):
            SeparationConfig(segment_length=35.0)

        # Boundary values
        config_min = SeparationConfig(segment_length=1.0)
        config_max = SeparationConfig(segment_length=30.0)

        assert config_min.segment_length == 1.0
        assert config_max.segment_length == 30.0

    def test_overlap_bounds(self) -> None:
        """overlap should be bounded [0.1, 0.9]."""
        with pytest.raises(pydantic.ValidationError):
            SeparationConfig(overlap=0.05)

        with pytest.raises(pydantic.ValidationError):
            SeparationConfig(overlap=0.95)

        config_min = SeparationConfig(overlap=0.1)
        config_max = SeparationConfig(overlap=0.9)

        assert config_min.overlap == 0.1
        assert config_max.overlap == 0.9

    def test_shifts_bounds(self) -> None:
        """shifts should be bounded [0, 5]."""
        with pytest.raises(pydantic.ValidationError):
            SeparationConfig(shifts=-1)

        with pytest.raises(pydantic.ValidationError):
            SeparationConfig(shifts=6)

        config_min = SeparationConfig(shifts=0)
        config_max = SeparationConfig(shifts=5)

        assert config_min.shifts == 0
        assert config_max.shifts == 5

    def test_mp3_bitrate_bounds(self) -> None:
        """mp3_bitrate should be bounded [128, 320]."""
        with pytest.raises(pydantic.ValidationError):
            SeparationConfig(mp3_bitrate=64)

        with pytest.raises(pydantic.ValidationError):
            SeparationConfig(mp3_bitrate=384)

        config_min = SeparationConfig(mp3_bitrate=128)
        config_max = SeparationConfig(mp3_bitrate=320)

        assert config_min.mp3_bitrate == 128
        assert config_max.mp3_bitrate == 320

    def test_two_stems_option(self) -> None:
        """two_stems should accept string or None."""
        config_none = SeparationConfig(two_stems=None)
        config_vocals = SeparationConfig(two_stems="vocals")

        assert config_none.two_stems is None
        assert config_vocals.two_stems == "vocals"

    def test_frozen_model(self) -> None:
        """SeparationConfig should be immutable."""
        config = SeparationConfig()

        with pytest.raises(pydantic.ValidationError):
            config.model = DemucsModel.MDX_EXTRA  # type: ignore[misc]

    def test_custom_device(self) -> None:
        """device should accept custom string values."""
        config_cuda = SeparationConfig(device="cuda:0")
        config_cpu = SeparationConfig(device="cpu")

        assert config_cuda.device == "cuda:0"
        assert config_cpu.device == "cpu"


# --------------------------------------------------------------------------- #
# StemResult Tests
# --------------------------------------------------------------------------- #


class TestStemResult:
    """Tests for StemResult model."""

    def test_minimal_result(self) -> None:
        """StemResult with required fields."""
        result = StemResult(
            stems={"vocals": Path("/output/vocals.wav")},
            source_path=Path("/input/song.mp3"),
            config=SeparationConfig(),
            processing_time_seconds=10.5,
        )

        assert "vocals" in result.stems
        assert result.source_path == Path("/input/song.mp3")
        assert result.processing_time_seconds == 10.5

    def test_full_stems(self) -> None:
        """StemResult with all 4 stems."""
        stems = {
            "vocals": Path("/output/vocals.wav"),
            "drums": Path("/output/drums.wav"),
            "bass": Path("/output/bass.wav"),
            "other": Path("/output/other.wav"),
        }

        result = StemResult(
            stems=stems,
            source_path=Path("/input/song.mp3"),
            config=SeparationConfig(),
            processing_time_seconds=30.0,
        )

        assert len(result.stems) == 4
        assert all(stem in result.stems for stem in ["vocals", "drums", "bass", "other"])

    def test_vocals_property(self) -> None:
        """vocals property should return vocals path."""
        vocals_path = Path("/output/vocals.wav")
        result = StemResult(
            stems={"vocals": vocals_path},
            source_path=Path("/input/song.mp3"),
            config=SeparationConfig(),
            processing_time_seconds=10.0,
        )

        assert result.vocals == vocals_path

    def test_vocals_property_missing(self) -> None:
        """vocals property should return None if no vocals."""
        result = StemResult(
            stems={"drums": Path("/output/drums.wav")},
            source_path=Path("/input/song.mp3"),
            config=SeparationConfig(),
            processing_time_seconds=10.0,
        )

        assert result.vocals is None

    def test_instrumental_property(self) -> None:
        """instrumental property currently returns None."""
        result = StemResult(
            stems={"vocals": Path("/output/vocals.wav")},
            source_path=Path("/input/song.mp3"),
            config=SeparationConfig(),
            processing_time_seconds=10.0,
        )

        # Current implementation returns None
        assert result.instrumental is None

    def test_frozen_model(self) -> None:
        """StemResult should be immutable."""
        result = StemResult(
            stems={"vocals": Path("/output/vocals.wav")},
            source_path=Path("/input/song.mp3"),
            config=SeparationConfig(),
            processing_time_seconds=10.0,
        )

        with pytest.raises(pydantic.ValidationError):
            result.processing_time_seconds = 20.0  # type: ignore[misc]

    def test_nested_config(self) -> None:
        """StemResult should preserve nested SeparationConfig."""
        config = SeparationConfig(
            model=DemucsModel.HTDEMUCS_6S,
            segment_length=15.0,
            shifts=3,
        )

        result = StemResult(
            stems={"vocals": Path("/output/vocals.wav")},
            source_path=Path("/input/song.mp3"),
            config=config,
            processing_time_seconds=45.0,
        )

        assert result.config.model == DemucsModel.HTDEMUCS_6S
        assert result.config.segment_length == 15.0
        assert result.config.shifts == 3


# --------------------------------------------------------------------------- #
# Integration Tests
# --------------------------------------------------------------------------- #


class TestSeparationModelIntegration:
    """Integration tests for separation models."""

    def test_config_matches_model_stems(self) -> None:
        """Config model should determine available stems."""
        config_4 = SeparationConfig(model=DemucsModel.HTDEMUCS_FT)
        config_6 = SeparationConfig(model=DemucsModel.HTDEMUCS_6S)

        assert config_4.model.stem_count == 4
        assert len(config_4.model.stems) == 4

        assert config_6.model.stem_count == 6
        assert len(config_6.model.stems) == 6

    def test_result_with_different_configs(self) -> None:
        """StemResult should work with various config combinations."""
        configs = [
            SeparationConfig(),  # Defaults
            SeparationConfig(model=DemucsModel.MDX_EXTRA, shifts=3),
            SeparationConfig(two_stems="vocals", overlap=0.5),
        ]

        for config in configs:
            result = StemResult(
                stems={"vocals": Path("/output/vocals.wav")},
                source_path=Path("/input/song.mp3"),
                config=config,
                processing_time_seconds=10.0,
            )
            assert result.config is not None

    def test_model_enum_in_config_serialization(self) -> None:
        """DemucsModel should serialize correctly in config."""
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)

        # Model should be JSON serializable via Pydantic
        data = config.model_dump()

        assert data["model"] == "htdemucs"

    def test_all_models_valid_in_config(self) -> None:
        """All DemucsModel values should be valid in SeparationConfig."""
        for model in DemucsModel:
            config = SeparationConfig(model=model)
            assert config.model == model


# --------------------------------------------------------------------------- #
# Instrumental Property Tests
# --------------------------------------------------------------------------- #


class TestInstrumentalProperty:
    """Tests for the instrumental property of StemResult."""

    @pytest.fixture
    def mock_audio_data(self) -> np.ndarray:
        """Sample audio data for testing."""
        return np.random.rand(44100, 2).astype(np.float32)  # 1 second stereo @ 44.1kHz

    @pytest.fixture
    def sample_rate(self) -> int:
        """Standard sample rate for testing."""
        return 44100

    @pytest.fixture
    def stem_result_empty(self, tmp_path: Path) -> StemResult:
        """StemResult with empty stems dict."""
        return StemResult(
            stems={},
            source_path=tmp_path / "song.mp3",
            config=SeparationConfig(),
            processing_time_seconds=10.0,
        )

    @pytest.fixture
    def stem_result_full(self, tmp_path: Path) -> StemResult:
        """StemResult with all 4 stems."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return StemResult(
            stems={
                "vocals": output_dir / "vocals.wav",
                "drums": output_dir / "drums.wav",
                "bass": output_dir / "bass.wav",
                "other": output_dir / "other.wav",
            },
            source_path=tmp_path / "song.mp3",
            config=SeparationConfig(),
            processing_time_seconds=30.0,
        )

    @pytest.fixture
    def stem_result_partial(self, tmp_path: Path) -> StemResult:
        """StemResult with only drums stem."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return StemResult(
            stems={
                "vocals": output_dir / "vocals.wav",
                "drums": output_dir / "drums.wav",
            },
            source_path=tmp_path / "song.mp3",
            config=SeparationConfig(),
            processing_time_seconds=15.0,
        )

    def test_instrumental_property_returns_none_when_no_stems(
        self, stem_result_empty: StemResult
    ) -> None:
        """instrumental should return None when stems dict is empty."""
        assert stem_result_empty.instrumental is None

    def test_instrumental_property_returns_none_when_only_vocals(
        self, tmp_path: Path
    ) -> None:
        """instrumental should return None when only vocals stem exists."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = StemResult(
            stems={"vocals": output_dir / "vocals.wav"},
            source_path=tmp_path / "song.mp3",
            config=SeparationConfig(),
            processing_time_seconds=10.0,
        )

        assert result.instrumental is None

    @patch("soundlab.separation.models.sf.write")
    @patch("soundlab.separation.models.sf.read")
    def test_instrumental_property_computes_on_demand(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
        stem_result_full: StemResult,
        mock_audio_data: np.ndarray,
        sample_rate: int,
    ) -> None:
        """instrumental should load stem files, sum them, and save result."""
        # Setup mock to return audio data
        mock_read.return_value = (mock_audio_data, sample_rate)

        # Make stem files "exist"
        for stem_path in stem_result_full.stems.values():
            stem_path.parent.mkdir(parents=True, exist_ok=True)
            stem_path.touch()

        # Access the property
        instrumental_path = stem_result_full.instrumental

        # Verify reads occurred for non-vocal stems
        expected_read_calls = [
            call(stem_result_full.stems["drums"]),
            call(stem_result_full.stems["bass"]),
            call(stem_result_full.stems["other"]),
        ]
        mock_read.assert_has_calls(expected_read_calls, any_order=True)
        assert mock_read.call_count == 3

        # Verify write occurred
        mock_write.assert_called_once()
        write_args = mock_write.call_args
        assert write_args[0][0] == stem_result_full.stems["drums"].parent / "instrumental.wav"
        assert write_args[0][2] == sample_rate

        # Verify return value
        assert instrumental_path == stem_result_full.stems["drums"].parent / "instrumental.wav"

    @patch("soundlab.separation.models.sf.write")
    @patch("soundlab.separation.models.sf.read")
    def test_instrumental_property_caches_result(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
        stem_result_full: StemResult,
        mock_audio_data: np.ndarray,
        sample_rate: int,
    ) -> None:
        """instrumental should cache result and not recompute on subsequent calls."""
        mock_read.return_value = (mock_audio_data, sample_rate)

        # Make stem files "exist"
        for stem_path in stem_result_full.stems.values():
            stem_path.parent.mkdir(parents=True, exist_ok=True)
            stem_path.touch()

        # First access
        first_result = stem_result_full.instrumental

        # Second access
        second_result = stem_result_full.instrumental

        # Results should be identical
        assert first_result == second_result

        # Read should only be called during first access (3 stems)
        assert mock_read.call_count == 3

        # Write should only be called once
        mock_write.assert_called_once()

    def test_instrumental_property_with_missing_stem_files(
        self, stem_result_full: StemResult
    ) -> None:
        """instrumental should return None when stem paths exist but files don't."""
        # Stem paths are in stems dict but files don't exist on disk
        # The property should handle this gracefully
        result = stem_result_full.instrumental

        assert result is None

    @patch("soundlab.separation.models.sf.write")
    @patch("soundlab.separation.models.sf.read")
    def test_instrumental_property_handles_partial_stems(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
        stem_result_partial: StemResult,
        mock_audio_data: np.ndarray,
        sample_rate: int,
    ) -> None:
        """instrumental should work with only some non-vocal stems (e.g., just drums)."""
        mock_read.return_value = (mock_audio_data, sample_rate)

        # Make only drums file exist (vocals exists but is not used for instrumental)
        drums_path = stem_result_partial.stems["drums"]
        drums_path.parent.mkdir(parents=True, exist_ok=True)
        drums_path.touch()

        # Access the property
        instrumental_path = stem_result_partial.instrumental

        # Verify only drums was read (only non-vocal stem available)
        mock_read.assert_called_once_with(drums_path)

        # Verify write occurred
        mock_write.assert_called_once()

        # Verify return value
        assert instrumental_path == drums_path.parent / "instrumental.wav"

    @patch("soundlab.separation.models.sf.write")
    @patch("soundlab.separation.models.sf.read")
    def test_instrumental_property_handles_mixed_existing_files(
        self,
        mock_read: MagicMock,
        mock_write: MagicMock,
        stem_result_full: StemResult,
        mock_audio_data: np.ndarray,
        sample_rate: int,
    ) -> None:
        """instrumental should skip missing files and use available ones."""
        mock_read.return_value = (mock_audio_data, sample_rate)

        # Create only drums and bass files, not 'other'
        drums_path = stem_result_full.stems["drums"]
        bass_path = stem_result_full.stems["bass"]
        drums_path.parent.mkdir(parents=True, exist_ok=True)
        drums_path.touch()
        bass_path.touch()

        # Access the property
        instrumental_path = stem_result_full.instrumental

        # Verify reads for existing files only
        assert mock_read.call_count == 2
        mock_read.assert_any_call(drums_path)
        mock_read.assert_any_call(bass_path)

        # Verify write occurred
        mock_write.assert_called_once()

        # Verify return value
        assert instrumental_path == drums_path.parent / "instrumental.wav"
