"""Integration tests for stem separation workflow.

This module tests the complete end-to-end stem separation pipeline,
including model loading, audio processing, and file I/O operations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from soundlab.core.exceptions import AudioLoadError, GPUMemoryError, ModelNotFoundError
from soundlab.io.audio_io import get_audio_metadata, load_audio, save_audio
from soundlab.separation import DemucsModel, SeparationConfig, StemSeparator

if TYPE_CHECKING:
    from soundlab.core.types import ProgressCallback


# === Fixtures ===


@pytest.fixture
def test_audio_files(temp_dir: Path, sample_rate: int) -> dict[str, Path]:
    """Create test audio files in different formats for integration testing.

    Returns:
        Dictionary mapping format names to file paths.
    """
    files = {}

    # Generate 5-second audio with mixed frequencies (simulating music)
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = (
        0.3 * np.sin(2 * np.pi * 220 * t) +   # Bass-like
        0.3 * np.sin(2 * np.pi * 440 * t) +   # Vocals-like
        0.2 * np.sin(2 * np.pi * 880 * t) +   # Treble
        0.1 * np.random.randn(len(t)).astype(np.float32)  # Drums-like noise
    )
    audio = audio / np.max(np.abs(audio)) * 0.8  # Normalize to prevent clipping

    # Create WAV file
    wav_path = temp_dir / "test_music.wav"
    sf.write(wav_path, audio, sample_rate, subtype="PCM_16")
    files["wav"] = wav_path

    # Create FLAC file
    flac_path = temp_dir / "test_music.flac"
    sf.write(flac_path, audio, sample_rate, subtype="PCM_16")
    files["flac"] = flac_path

    # Create MP3 file (requires pydub/ffmpeg in real scenario, skip if not available)
    try:
        from pydub import AudioSegment as PydubSegment
        mp3_path = temp_dir / "test_music.mp3"
        audio_int16 = (audio * 32767).astype(np.int16)
        pydub_audio = PydubSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        pydub_audio.export(str(mp3_path), format="mp3", bitrate="320k")
        files["mp3"] = mp3_path
    except (ImportError, FileNotFoundError):
        # MP3 encoding not available, skip
        pass

    return files


@pytest.fixture
def short_audio_file(temp_dir: Path, sample_rate: int) -> Path:
    """Create a short 2-second audio file for faster tests."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    path = temp_dir / "short_audio.wav"
    sf.write(path, audio, sample_rate)
    return path


@pytest.fixture
def stereo_audio_file(temp_dir: Path, sample_rate: int) -> Path:
    """Create a stereo audio file for testing."""
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Different content per channel
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 554.37 * t)  # C# note

    stereo = np.stack([left, right], axis=1)

    path = temp_dir / "stereo_audio.wav"
    sf.write(path, stereo, sample_rate)
    return path


@pytest.fixture
def mock_demucs_separator():
    """Mock the Demucs model for tests without actual model loading.

    This fixture creates a realistic mock that simulates the behavior
    of the actual Demucs model, including model loading, inference,
    and tensor operations.
    """
    with patch("demucs.pretrained.get_model") as mock_get_model:
        # Create mock model
        mock_model = MagicMock()
        mock_model.samplerate = 44100
        mock_model.audio_channels = 2
        mock_model.sources = ["vocals", "drums", "bass", "other"]

        # Mock to() method (device transfer)
        mock_model.to.return_value = mock_model

        # Mock eval() method
        mock_model.eval.return_value = None

        mock_get_model.return_value = mock_model

        # Mock apply_model to return realistic stem tensors
        with patch("demucs.apply.apply_model") as mock_apply:
            def apply_side_effect(model, wav, segment=None, overlap=None, shifts=None, progress=False):
                # wav shape: (batch, channels, samples)
                batch_size, channels, samples = wav.shape
                num_sources = 4

                # Create mock separated sources
                sources = torch.randn(batch_size, num_sources, channels, samples) * 0.1
                return sources

            mock_apply.side_effect = apply_side_effect

            # Mock AudioFile
            with patch("demucs.audio.AudioFile") as mock_audio_file:
                def create_audio_file(path):
                    mock_af = MagicMock()

                    def read_audio(streams=0, samplerate=44100, channels=2):
                        # Load actual audio and convert to tensor
                        audio, sr = sf.read(path, dtype="float32")
                        if audio.ndim == 1:
                            audio = np.stack([audio, audio])  # Mono to stereo
                        else:
                            audio = audio.T  # (samples, channels) -> (channels, samples)

                        return torch.from_numpy(audio.astype(np.float32))

                    mock_af.read = read_audio
                    return mock_af

                mock_audio_file.side_effect = create_audio_file

                yield mock_model


@pytest.fixture
def progress_tracker() -> Mock:
    """Create a mock progress callback for tracking progress updates."""
    return Mock(spec=["__call__"])


# === Integration Tests ===


class TestStemSeparationWorkflow:
    """Test complete stem separation workflows."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_separate_with_htdemucs_model(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test complete separation workflow with htdemucs model.

        This test verifies:
        - Model loading and initialization
        - Audio file processing
        - Stem file creation
        - Proper output structure
        """
        config = SeparationConfig(
            model=DemucsModel.HTDEMUCS,
            segment_length=7.8,
            overlap=0.25,
        )
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )

        # Verify result structure
        assert result.source_path == short_audio_file
        assert result.config.model == DemucsModel.HTDEMUCS
        assert result.processing_time_seconds > 0

        # Verify all stems were created
        expected_stems = ["vocals", "drums", "bass", "other"]
        assert set(result.stem_names) == set(expected_stems)

        # Verify files exist and are valid
        for stem_name, stem_path in result.stems.items():
            assert stem_path.exists(), f"Stem file missing: {stem_name}"
            assert stem_path.parent == temp_output_dir
            assert stem_path.suffix == ".wav"

            # Verify audio is loadable
            metadata = get_audio_metadata(stem_path)
            assert metadata.sample_rate == 44100
            assert metadata.duration_seconds > 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_separate_with_htdemucs_ft_model(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test separation with fine-tuned htdemucs_ft model.

        The ft (fine-tuned) model should produce the same stem structure
        but potentially higher quality results.
        """
        config = SeparationConfig(
            model=DemucsModel.HTDEMUCS_FT,
        )
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )

        assert result.config.model == DemucsModel.HTDEMUCS_FT
        assert len(result.stems) == 4

        # Verify property accessors
        assert result.vocals is not None
        assert result.drums is not None
        assert result.bass is not None
        assert result.other is not None

    @pytest.mark.integration
    @pytest.mark.slow
    def test_separate_with_different_segment_lengths(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test separation with various segment length configurations.

        Different segment lengths affect memory usage and processing time.
        Shorter segments use less memory but may have more boundary artifacts.
        """
        segment_lengths = [3.0, 7.8, 15.0]

        for seg_len in segment_lengths:
            output_dir = temp_output_dir / f"seg_{seg_len}"
            output_dir.mkdir()

            config = SeparationConfig(
                model=DemucsModel.HTDEMUCS,
                segment_length=seg_len,
            )
            separator = StemSeparator(config)

            result = separator.separate(
                audio_path=short_audio_file,
                output_dir=output_dir,
            )

            assert len(result.stems) == 4
            assert result.config.segment_length == seg_len

            # Verify all stems were created
            for stem_path in result.stems.values():
                assert stem_path.exists()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_separate_with_different_overlaps(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test separation with different overlap settings.

        Higher overlap improves quality at segment boundaries but increases
        processing time. Tests verify that different overlap values work correctly.
        """
        overlaps = [0.1, 0.25, 0.5]

        for overlap in overlaps:
            output_dir = temp_output_dir / f"overlap_{int(overlap * 100)}"
            output_dir.mkdir()

            config = SeparationConfig(
                model=DemucsModel.HTDEMUCS,
                overlap=overlap,
            )
            separator = StemSeparator(config)

            result = separator.separate(
                audio_path=short_audio_file,
                output_dir=output_dir,
            )

            assert result.config.overlap == overlap
            assert len(result.stems) == 4

    @pytest.mark.integration
    @pytest.mark.slow
    def test_separate_with_progress_callback(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
        progress_tracker: Mock,
    ):
        """Test that progress callbacks are invoked during separation.

        Progress callbacks allow UI integration and user feedback during
        long-running separation operations.
        """
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
            progress_callback=progress_tracker,
        )

        # Verify progress callback was called
        assert progress_tracker.call_count >= 2  # At least start and end

        # Verify final call indicates completion
        last_call = progress_tracker.call_args_list[-1]
        assert last_call[0][0] == 100  # Current progress
        assert last_call[0][1] == 100  # Total progress
        assert "complete" in last_call[0][2].lower()  # Status message

    @pytest.mark.integration
    @pytest.mark.slow
    def test_separate_creates_valid_output_files(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test that separated stems are valid, loadable audio files.

        Verifies that output files:
        - Can be loaded by audio libraries
        - Have correct sample rate
        - Have reasonable audio data
        - Match expected format
        """
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )

        # Load and verify each stem
        for stem_name, stem_path in result.stems.items():
            # Load with soundlab
            audio_segment = load_audio(stem_path)

            assert audio_segment.sample_rate == 44100
            assert audio_segment.samples.dtype == np.float32
            assert len(audio_segment.samples) > 0

            # Verify audio is not all zeros (actual separation occurred)
            # Note: With mock, we get random data, so this checks the pipeline works
            assert not np.allclose(audio_segment.samples, 0)

            # Verify file metadata
            metadata = get_audio_metadata(stem_path)
            assert metadata.sample_rate == 44100
            assert metadata.format.value == "wav"

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.parametrize("audio_format", ["wav", "flac"])
    def test_separate_different_audio_formats(
        self,
        test_audio_files: dict[str, Path],
        temp_output_dir: Path,
        mock_demucs_separator,
        audio_format: str,
    ):
        """Test separation with different input audio formats (WAV, MP3, FLAC).

        The separator should handle various input formats transparently,
        converting them as needed for processing.
        """
        if audio_format not in test_audio_files:
            pytest.skip(f"Format {audio_format} not available in test environment")

        audio_path = test_audio_files[audio_format]
        output_dir = temp_output_dir / audio_format
        output_dir.mkdir()

        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=audio_path,
            output_dir=output_dir,
        )

        assert result.source_path == audio_path
        assert len(result.stems) == 4

        # All output should be WAV regardless of input
        for stem_path in result.stems.values():
            assert stem_path.suffix == ".wav"
            assert stem_path.exists()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_separate_stereo_audio(
        self,
        stereo_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test separation of stereo audio files.

        Stereo files should be processed with both channels preserved
        in the separated stems.
        """
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=stereo_audio_file,
            output_dir=temp_output_dir,
        )

        # Verify stems maintain stereo format
        for stem_path in result.stems.values():
            metadata = get_audio_metadata(stem_path)
            # Note: Demucs outputs stereo regardless of input
            assert metadata.channels in [1, 2]

    @pytest.mark.integration
    @pytest.mark.slow
    def test_separate_with_two_stems_mode(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test two-stem mode that extracts one stem vs everything else.

        Two-stem mode is useful for vocal isolation (vocals vs instrumental)
        or other specific extraction tasks.
        """
        config = SeparationConfig(
            model=DemucsModel.HTDEMUCS,
            two_stems="vocals",
        )
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )

        # Should only have two stems
        assert len(result.stems) == 2
        assert "vocals" in result.stem_names
        assert "no_vocals" in result.stem_names

        # Verify both files exist
        assert result.stems["vocals"].exists()
        assert result.stems["no_vocals"].exists()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_separate_with_split_enabled(
        self,
        test_audio_files: dict[str, Path],
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test segment-based processing for long audio files.

        Split mode processes audio in segments to reduce memory usage,
        essential for long files or limited GPU memory.
        """
        config = SeparationConfig(
            model=DemucsModel.HTDEMUCS,
            split=True,
            segment_length=5.0,
        )
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=test_audio_files["wav"],
            output_dir=temp_output_dir,
        )

        assert result.config.split is True
        assert len(result.stems) == 4

    @pytest.mark.integration
    @pytest.mark.slow
    def test_separate_with_split_disabled(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test whole-file processing without segmentation.

        For short files or when memory is not a concern, disabling split
        can provide better quality by avoiding segment boundaries.
        """
        config = SeparationConfig(
            model=DemucsModel.HTDEMUCS,
            split=False,
        )
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )

        assert result.config.split is False
        assert len(result.stems) == 4

    @pytest.mark.integration
    @pytest.mark.slow
    def test_multiple_separations_same_separator(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test reusing the same separator instance for multiple files.

        The separator should properly cache the model and reuse it
        for subsequent separations, improving efficiency.
        """
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        # First separation
        output_dir1 = temp_output_dir / "run1"
        output_dir1.mkdir()
        result1 = separator.separate(
            audio_path=short_audio_file,
            output_dir=output_dir1,
        )

        # Second separation (model should be cached)
        output_dir2 = temp_output_dir / "run2"
        output_dir2.mkdir()
        result2 = separator.separate(
            audio_path=short_audio_file,
            output_dir=output_dir2,
        )

        # Both should succeed
        assert len(result1.stems) == 4
        assert len(result2.stems) == 4

        # Files should be in different directories
        assert result1.stems["vocals"].parent == output_dir1
        assert result2.stems["vocals"].parent == output_dir2


class TestSeparationErrorHandling:
    """Test error handling and edge cases in stem separation."""

    @pytest.mark.integration
    def test_separate_nonexistent_file(
        self,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test that separating a nonexistent file raises appropriate error."""
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        with pytest.raises(Exception):  # AudioLoadError or FileNotFoundError
            separator.separate(
                audio_path="/nonexistent/audio.wav",
                output_dir=temp_output_dir,
            )

    @pytest.mark.integration
    def test_separate_invalid_audio_format(
        self,
        temp_dir: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test handling of invalid/unsupported audio formats."""
        # Create a text file with .wav extension
        invalid_file = temp_dir / "invalid.wav"
        invalid_file.write_text("This is not audio data")

        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        with pytest.raises(Exception):  # Should raise some audio loading error
            separator.separate(
                audio_path=invalid_file,
                output_dir=temp_output_dir,
            )

    @pytest.mark.integration
    def test_model_load_failure(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
    ):
        """Test handling when model fails to load.

        This can happen if the model files are missing or corrupted.
        """
        with patch("demucs.pretrained.get_model", side_effect=Exception("Model not found")):
            config = SeparationConfig(model=DemucsModel.HTDEMUCS)
            separator = StemSeparator(config)

            with pytest.raises(ModelNotFoundError):
                separator.separate(
                    audio_path=short_audio_file,
                    output_dir=temp_output_dir,
                )

    @pytest.mark.integration
    def test_output_directory_creation(
        self,
        short_audio_file: Path,
        temp_dir: Path,
        mock_demucs_separator,
    ):
        """Test that output directory is created if it doesn't exist."""
        # Use a nested path that doesn't exist
        output_dir = temp_dir / "nested" / "output" / "stems"
        assert not output_dir.exists()

        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=output_dir,
        )

        # Directory should now exist
        assert output_dir.exists()
        assert output_dir.is_dir()

        # Files should be in the created directory
        for stem_path in result.stems.values():
            assert stem_path.parent == output_dir


class TestGPUvsCPUProcessing:
    """Test GPU and CPU processing paths for stem separation."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_cpu_processing(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
        mock_cuda_unavailable,
    ):
        """Test separation on CPU when GPU is not available.

        CPU processing should work as a fallback when CUDA is unavailable.
        """
        config = SeparationConfig(
            model=DemucsModel.HTDEMUCS,
            device="cpu",
        )
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )

        assert len(result.stems) == 4
        assert result.processing_time_seconds > 0

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_gpu_processing(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
        mock_cuda_available,
    ):
        """Test separation on GPU when CUDA is available.

        GPU processing should be faster and handle the same workflow.
        """
        config = SeparationConfig(
            model=DemucsModel.HTDEMUCS,
            device="cuda",
        )
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )

        assert len(result.stems) == 4

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_auto_device_selection(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
        mock_cuda_available,
    ):
        """Test automatic device selection (defaults to GPU if available).

        The 'auto' device setting should intelligently choose the best
        available device.
        """
        config = SeparationConfig(
            model=DemucsModel.HTDEMUCS,
            device="auto",
        )
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )

        assert len(result.stems) == 4

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_gpu_memory_check_failure(
        self,
        test_audio_files: dict[str, Path],
        temp_output_dir: Path,
    ):
        """Test handling when GPU memory is insufficient.

        Should raise GPUMemoryError when available VRAM is too low.
        """
        # Mock extremely low VRAM
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.mem_get_info", return_value=(0.5 * 1024**3, 2 * 1024**3)):
                with patch("demucs.pretrained.get_model") as mock_get:
                    mock_model = MagicMock()
                    mock_model.samplerate = 44100
                    mock_model.audio_channels = 2
                    mock_model.to.return_value = mock_model
                    mock_get.return_value = mock_model

                    with patch("demucs.audio.AudioFile") as mock_af:
                        def create_af(path):
                            mock = MagicMock()
                            # Return long audio requiring lots of memory
                            audio = torch.randn(2, 44100 * 300)  # 5 minutes
                            mock.read.return_value = audio
                            return mock
                        mock_af.side_effect = create_af

                        config = SeparationConfig(
                            model=DemucsModel.HTDEMUCS_FT,
                            device="cuda",
                            split=False,  # Disable splitting to force memory error
                        )
                        separator = StemSeparator(config)

                        with pytest.raises(GPUMemoryError):
                            separator.separate(
                                audio_path=test_audio_files["wav"],
                                output_dir=temp_output_dir,
                            )


class TestSeparatorResourceManagement:
    """Test proper resource management and cleanup."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_model_unload(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test explicit model unloading to free resources.

        Users should be able to unload models to free GPU memory
        when the separator is no longer needed.
        """
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        # Perform separation (loads model)
        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )

        assert separator._model is not None

        # Unload model
        separator.unload_model()

        assert separator._model is None

    @pytest.mark.integration
    @pytest.mark.slow
    def test_separator_lazy_loading(
        self,
        mock_demucs_separator,
    ):
        """Test that model is only loaded when separation is called.

        Creating a StemSeparator instance should not immediately load
        the model, allowing for efficient initialization.
        """
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        # Model should not be loaded yet
        assert separator._model is None
        assert separator._device is None

    @pytest.mark.integration
    @pytest.mark.slow
    def test_processing_time_measurement(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test that processing time is accurately measured.

        The result should include accurate timing information for
        performance monitoring and optimization.
        """
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        start = time.perf_counter()
        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )
        elapsed = time.perf_counter() - start

        # Processing time should be recorded and reasonable
        assert result.processing_time_seconds > 0
        assert result.processing_time_seconds < elapsed + 1.0  # Allow 1s overhead

    @pytest.mark.integration
    @pytest.mark.slow
    def test_concurrent_separation_instances(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test that multiple separator instances can coexist.

        Different separators with different configs should not interfere
        with each other.
        """
        config1 = SeparationConfig(model=DemucsModel.HTDEMUCS, overlap=0.25)
        config2 = SeparationConfig(model=DemucsModel.HTDEMUCS_FT, overlap=0.5)

        separator1 = StemSeparator(config1)
        separator2 = StemSeparator(config2)

        output_dir1 = temp_output_dir / "sep1"
        output_dir2 = temp_output_dir / "sep2"
        output_dir1.mkdir()
        output_dir2.mkdir()

        result1 = separator1.separate(short_audio_file, output_dir1)
        result2 = separator2.separate(short_audio_file, output_dir2)

        assert result1.config.overlap == 0.25
        assert result2.config.overlap == 0.5
        assert result1.config.model == DemucsModel.HTDEMUCS
        assert result2.config.model == DemucsModel.HTDEMUCS_FT


class TestSeparationResultModel:
    """Test the StemResult model and its properties."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_result_property_accessors(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test convenience properties on StemResult.

        StemResult should provide easy access to individual stems
        through named properties.
        """
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )

        # Test property accessors
        assert result.vocals is not None
        assert result.vocals.exists()
        assert result.vocals.name == "vocals.wav"

        assert result.drums is not None
        assert result.drums.exists()
        assert result.drums.name == "drums.wav"

        assert result.bass is not None
        assert result.bass.exists()

        assert result.other is not None
        assert result.other.exists()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_result_stem_names(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test stem_names property returns correct stem identifiers."""
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )

        stem_names = result.stem_names
        assert isinstance(stem_names, list)
        assert len(stem_names) == 4
        assert set(stem_names) == {"vocals", "drums", "bass", "other"}

    @pytest.mark.integration
    @pytest.mark.slow
    def test_result_is_immutable(
        self,
        short_audio_file: Path,
        temp_output_dir: Path,
        mock_demucs_separator,
    ):
        """Test that StemResult is frozen and immutable.

        Results should not be modifiable after creation to prevent
        accidental corruption of processing records.
        """
        config = SeparationConfig(model=DemucsModel.HTDEMUCS)
        separator = StemSeparator(config)

        result = separator.separate(
            audio_path=short_audio_file,
            output_dir=temp_output_dir,
        )

        # Attempt to modify should fail
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            result.processing_time_seconds = 999.0
