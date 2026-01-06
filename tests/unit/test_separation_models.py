"""Tests for soundlab.separation.models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from soundlab.separation.models import DemucsModel, SeparationConfig, StemResult


class TestDemucsModel:
    """Test DemucsModel enum."""

    def test_htdemucs_value(self):
        """HTDEMUCS should have correct value."""
        assert DemucsModel.HTDEMUCS.value == "htdemucs"

    def test_htdemucs_ft_value(self):
        """HTDEMUCS_FT should have correct value."""
        assert DemucsModel.HTDEMUCS_FT.value == "htdemucs_ft"

    def test_htdemucs_6s_value(self):
        """HTDEMUCS_6S should have correct value."""
        assert DemucsModel.HTDEMUCS_6S.value == "htdemucs_6s"

    def test_stem_count_4_stems(self):
        """Standard models should have 4 stems."""
        assert DemucsModel.HTDEMUCS.stem_count == 4
        assert DemucsModel.HTDEMUCS_FT.stem_count == 4
        assert DemucsModel.MDX_EXTRA.stem_count == 4

    def test_stem_count_6_stems(self):
        """HTDEMUCS_6S should have 6 stems."""
        assert DemucsModel.HTDEMUCS_6S.stem_count == 6

    def test_stems_4_stem_model(self):
        """4-stem models should return correct stem names."""
        stems = DemucsModel.HTDEMUCS_FT.stems
        assert stems == ["vocals", "drums", "bass", "other"]

    def test_stems_6_stem_model(self):
        """6-stem model should include piano and guitar."""
        stems = DemucsModel.HTDEMUCS_6S.stems
        assert "piano" in stems
        assert "guitar" in stems
        assert len(stems) == 6

    def test_description_property(self):
        """All models should have descriptions."""
        for model in DemucsModel:
            assert model.description
            assert isinstance(model.description, str)


class TestSeparationConfig:
    """Test SeparationConfig model."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = SeparationConfig()

        assert config.model == DemucsModel.HTDEMUCS_FT
        assert config.segment_length == 7.8
        assert config.overlap == 0.25
        assert config.shifts == 1
        assert config.device == "auto"
        assert config.split is True

    def test_custom_model(self):
        """Should accept custom model selection."""
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        assert config.model == DemucsModel.HTDEMUCS

    def test_segment_length_bounds(self):
        """Segment length should be validated."""
        # Valid bounds
        config = SeparationConfig(segment_length=1.0)
        assert config.segment_length == 1.0

        config = SeparationConfig(segment_length=30.0)
        assert config.segment_length == 30.0

        # Invalid: too low
        with pytest.raises(ValidationError):
            SeparationConfig(segment_length=0.5)

        # Invalid: too high
        with pytest.raises(ValidationError):
            SeparationConfig(segment_length=35.0)

    def test_overlap_bounds(self):
        """Overlap should be validated."""
        # Valid
        SeparationConfig(overlap=0.1)
        SeparationConfig(overlap=0.9)

        # Invalid
        with pytest.raises(ValidationError):
            SeparationConfig(overlap=0.05)
        with pytest.raises(ValidationError):
            SeparationConfig(overlap=0.95)

    def test_shifts_bounds(self):
        """Shifts should be validated."""
        SeparationConfig(shifts=0)
        SeparationConfig(shifts=5)

        with pytest.raises(ValidationError):
            SeparationConfig(shifts=-1)
        with pytest.raises(ValidationError):
            SeparationConfig(shifts=6)

    def test_two_stems_option(self):
        """Should accept two_stems option."""
        config = SeparationConfig(two_stems="vocals")
        assert config.two_stems == "vocals"

        config = SeparationConfig(two_stems=None)
        assert config.two_stems is None

    def test_mp3_bitrate_bounds(self):
        """MP3 bitrate should be validated."""
        SeparationConfig(mp3_bitrate=128)
        SeparationConfig(mp3_bitrate=320)

        with pytest.raises(ValidationError):
            SeparationConfig(mp3_bitrate=64)
        with pytest.raises(ValidationError):
            SeparationConfig(mp3_bitrate=512)

    def test_config_is_frozen(self):
        """Config should be immutable."""
        config = SeparationConfig()
        with pytest.raises(ValidationError):
            config.model = DemucsModel.HTDEMUCS


class TestStemResult:
    """Test StemResult model."""

    def test_create_result(self):
        """Should create result with stems dict."""
        stems = {
            "vocals": Path("vocals.wav"),
            "drums": Path("drums.wav"),
            "bass": Path("bass.wav"),
            "other": Path("other.wav"),
        }
        result = StemResult(
            stems=stems,
            source_path=Path("input.wav"),
            config=SeparationConfig(),
            processing_time_seconds=10.5,
        )

        assert len(result.stems) == 4
        assert result.processing_time_seconds == 10.5

    def test_vocals_property(self):
        """vocals property should return vocals path."""
        stems = {"vocals": Path("vocals.wav"), "drums": Path("drums.wav")}
        result = StemResult(
            stems=stems,
            source_path=Path("input.wav"),
            config=SeparationConfig(),
            processing_time_seconds=1.0,
        )

        assert result.vocals == Path("vocals.wav")

    def test_vocals_property_when_missing(self):
        """vocals property should return None if not present."""
        result = StemResult(
            stems={"drums": Path("drums.wav")},
            source_path=Path("input.wav"),
            config=SeparationConfig(),
            processing_time_seconds=1.0,
        )

        assert result.vocals is None

    def test_stem_names_property(self):
        """stem_names should return list of stem names."""
        stems = {"vocals": Path("v.wav"), "drums": Path("d.wav")}
        result = StemResult(
            stems=stems,
            source_path=Path("input.wav"),
            config=SeparationConfig(),
            processing_time_seconds=1.0,
        )

        assert set(result.stem_names) == {"vocals", "drums"}
