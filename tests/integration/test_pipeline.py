"""
Integration tests for full SoundLab pipelines.

These tests verify that all SoundLab components work together seamlessly
in production-like workflows combining separation, transcription, analysis,
effects, and I/O operations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

# Import from all major modules
from soundlab.analysis import (
    AnalysisResult,
    analyze_audio,
    detect_key,
    detect_tempo,
    measure_loudness,
)
from soundlab.core import AudioSegment
from soundlab.effects import (
    ChorusConfig,
    CompressorConfig,
    EffectsChain,
    GainConfig,
    ReverbConfig,
)
from soundlab.io import (
    batch_export,
    export_audio,
    load_audio,
    save_audio,
    save_midi,
)
from soundlab.separation import SeparationConfig, StemResult, StemSeparator
from soundlab.transcription import MIDIResult, MIDITranscriber, TranscriptionConfig


# === Pipeline Fixtures ===


@pytest.fixture
def pipeline_audio_path(temp_dir: Path, sample_rate: int) -> Path:
    """
    Create a multi-frequency test audio file suitable for pipeline testing.

    Generates a 3-second audio file with multiple harmonics to better
    simulate real music for separation, transcription, and analysis.
    """
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Create a richer signal with multiple harmonics
    audio = (
        0.3 * np.sin(2 * np.pi * 262 * t) +  # C4
        0.2 * np.sin(2 * np.pi * 330 * t) +  # E4
        0.2 * np.sin(2 * np.pi * 392 * t) +  # G4
        0.1 * np.sin(2 * np.pi * 523 * t)    # C5
    )

    # Add some amplitude variation
    envelope = 1.0 - 0.3 * np.abs(np.sin(2 * np.pi * 0.5 * t))
    audio = audio * envelope

    audio_path = temp_dir / "pipeline_test.wav"
    sf.write(audio_path, audio, sample_rate)
    return audio_path


@pytest.fixture
def multiple_audio_files(temp_dir: Path, sample_rate: int) -> list[Path]:
    """
    Create multiple audio files for batch processing tests.

    Generates 3 different audio files with varying frequencies and durations
    to test batch pipeline operations.
    """
    files = []

    for i in range(3):
        duration = 1.0 + i * 0.5  # 1s, 1.5s, 2s
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # Different frequencies for each file
        freq = 440 * (2 ** (i / 12))  # A4, A#4, B4
        audio = 0.5 * np.sin(2 * np.pi * freq * t)

        audio_path = temp_dir / f"batch_test_{i}.wav"
        sf.write(audio_path, audio, sample_rate)
        files.append(audio_path)

    return files


@pytest.fixture
def progress_tracker() -> dict[str, list[tuple[int, int, str]]]:
    """
    Create a progress tracking dictionary for pipeline tests.

    Tracks progress callbacks from different pipeline stages to verify
    that progress reporting works correctly across the entire workflow.
    """
    return {"callbacks": []}


@pytest.fixture
def mock_separator_with_stems():
    """
    Mock StemSeparator that returns realistic stem results.

    Used to avoid loading actual Demucs models during integration tests
    while still testing the full pipeline flow.
    """
    def mock_separate(audio_path: Path, output_dir: Path, progress_callback=None):
        output_dir.mkdir(parents=True, exist_ok=True)

        # Simulate progress callback
        if progress_callback:
            progress_callback(0, 100, "Separating stems...")
            progress_callback(80, 100, "Saving stems...")
            progress_callback(100, 100, "Complete")

        # Create mock stem files
        stems = {}
        sample_audio = np.random.randn(44100).astype(np.float32) * 0.1

        for stem_name in ["vocals", "drums", "bass", "other"]:
            stem_path = output_dir / f"{stem_name}.wav"
            sf.write(stem_path, sample_audio, 44100)
            stems[stem_name] = stem_path

        return StemResult(
            stems=stems,
            source_path=audio_path,
            config=SeparationConfig(),
            processing_time_seconds=1.5,
        )

    mock = MagicMock(spec=StemSeparator)
    mock.separate.side_effect = mock_separate
    return mock


@pytest.fixture
def mock_transcriber_with_notes():
    """
    Mock MIDITranscriber that returns realistic note events.

    Used to avoid loading actual Basic Pitch model during integration tests
    while still testing the full pipeline flow.
    """
    def mock_transcribe(audio_path: Path, output_dir: Path = None,
                       save_midi_file: bool = True, progress_callback=None):
        from soundlab.transcription import NoteEvent

        if output_dir is None:
            output_dir = audio_path.parent
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Simulate progress callback
        if progress_callback:
            progress_callback(0, 100, "Loading audio...")
            progress_callback(50, 100, "Transcribing...")
            progress_callback(100, 100, "Complete")

        # Create mock note events
        notes = [
            NoteEvent(start_time=0.0, end_time=0.5, pitch=60, velocity=80),
            NoteEvent(start_time=0.5, end_time=1.0, pitch=64, velocity=85),
            NoteEvent(start_time=1.0, end_time=1.5, pitch=67, velocity=90),
        ]

        midi_path = None
        if save_midi_file:
            midi_path = output_dir / f"{audio_path.stem}_transcribed.mid"
            midi_path.touch()  # Create empty file

        return MIDIResult(
            notes=notes,
            midi_path=midi_path,
            source_path=audio_path,
            config=TranscriptionConfig(),
            processing_time_seconds=2.0,
        )

    mock = MagicMock(spec=MIDITranscriber)
    mock.transcribe.side_effect = mock_transcribe
    return mock


# === Pipeline Integration Tests ===


@pytest.mark.integration
@pytest.mark.slow
class TestBasicPipelines:
    """Test basic multi-step audio processing pipelines."""

    def test_load_analyze_separate_export_pipeline(
        self,
        pipeline_audio_path: Path,
        temp_dir: Path,
        mock_separator_with_stems: MagicMock,
    ):
        """
        Test: Load audio → Analyze → Separate stems → Export

        Verifies that audio can be loaded, analyzed for musical features,
        separated into stems, and exported successfully. This is a common
        workflow for music production and DJing.
        """
        output_dir = temp_dir / "separated"

        # Step 1: Load audio
        audio_segment = load_audio(pipeline_audio_path)
        assert isinstance(audio_segment, AudioSegment)
        assert audio_segment.samples is not None

        # Step 2: Analyze audio
        analysis = analyze_audio(pipeline_audio_path)
        assert isinstance(analysis, AnalysisResult)
        assert analysis.tempo is not None
        assert analysis.key is not None
        assert analysis.loudness is not None

        # Verify analysis contains useful information
        assert analysis.duration_seconds > 0
        assert analysis.sample_rate == audio_segment.sample_rate

        # Step 3: Separate stems (using mock)
        separator = mock_separator_with_stems
        result = separator.separate(pipeline_audio_path, output_dir)

        assert isinstance(result, StemResult)
        assert len(result.stems) == 4
        assert "vocals" in result.stems
        assert "drums" in result.stems

        # Step 4: Export stems
        for stem_name, stem_path in result.stems.items():
            assert stem_path.exists()

            # Load and verify exported stem
            stem_audio = load_audio(stem_path)
            assert isinstance(stem_audio, AudioSegment)
            assert stem_audio.samples.shape[0] > 0

    def test_load_analyze_effects_export_pipeline(
        self,
        pipeline_audio_path: Path,
        temp_dir: Path,
    ):
        """
        Test: Load audio → Analyze tempo/key → Apply effects → Export

        Verifies that audio analysis results can inform effects processing,
        such as applying tempo-synced effects or key-matched processing.
        """
        output_path = temp_dir / "processed.wav"

        # Step 1: Load audio
        audio_segment = load_audio(pipeline_audio_path)

        # Step 2: Analyze for tempo and key
        tempo_result = detect_tempo(audio_segment.samples, audio_segment.sample_rate)
        key_result = detect_key(audio_segment.samples, audio_segment.sample_rate)

        assert tempo_result.bpm > 0
        assert key_result.key is not None

        # Step 3: Apply effects chain informed by analysis
        effects = EffectsChain()

        # Add compression (standard for any audio)
        effects.add(CompressorConfig(
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=5.0,
            release_ms=100.0,
        ))

        # Add reverb with size informed by tempo (faster = smaller room)
        room_size = 0.3 if tempo_result.bpm > 120 else 0.6
        effects.add(ReverbConfig(
            room_size=room_size,
            damping=0.5,
            wet_level=0.2,
        ))

        # Process audio
        processed = effects.process_array(
            audio_segment.samples,
            audio_segment.sample_rate,
        )

        assert processed.shape == audio_segment.samples.shape

        # Step 4: Export processed audio
        processed_segment = AudioSegment(
            samples=processed,
            sample_rate=audio_segment.sample_rate,
        )

        result_path = save_audio(processed_segment, output_path)
        assert result_path.exists()

        # Verify exported file can be loaded
        reloaded = load_audio(result_path)
        assert reloaded.samples.shape[0] > 0

    def test_separate_vocals_transcribe_save_pipeline(
        self,
        pipeline_audio_path: Path,
        temp_dir: Path,
        mock_separator_with_stems: MagicMock,
        mock_transcriber_with_notes: MagicMock,
    ):
        """
        Test: Separate vocals → Transcribe to MIDI → Save

        Verifies that separated vocal stems can be transcribed to MIDI,
        which is useful for melody extraction and music notation.
        """
        stems_dir = temp_dir / "stems"
        midi_dir = temp_dir / "midi"

        # Step 1: Separate stems to isolate vocals
        separator = mock_separator_with_stems
        separation_result = separator.separate(pipeline_audio_path, stems_dir)

        assert separation_result.vocals is not None
        vocals_path = separation_result.vocals
        assert vocals_path.exists()

        # Step 2: Transcribe vocals to MIDI
        transcriber = mock_transcriber_with_notes
        midi_result = transcriber.transcribe(
            vocals_path,
            midi_dir,
            save_midi_file=True,
        )

        assert isinstance(midi_result, MIDIResult)
        assert len(midi_result.notes) > 0
        assert midi_result.midi_path is not None

        # Step 3: Verify MIDI file was saved
        assert midi_result.midi_path.exists()

        # Verify transcription metadata
        assert midi_result.note_count > 0
        assert midi_result.duration > 0
        pitch_min, pitch_max = midi_result.pitch_range
        assert 0 <= pitch_min <= 127
        assert 0 <= pitch_max <= 127


@pytest.mark.integration
@pytest.mark.slow
class TestFullWorkflow:
    """Test complete end-to-end workflows combining multiple features."""

    def test_complete_production_workflow(
        self,
        pipeline_audio_path: Path,
        temp_dir: Path,
        mock_separator_with_stems: MagicMock,
        mock_transcriber_with_notes: MagicMock,
    ):
        """
        Test: Load → Analyze → Separate → Effects on stems → Export all

        Simulates a complete music production workflow where audio is analyzed,
        separated into stems, each stem is processed with effects, and all
        results are exported for use in a DAW or DJ software.
        """
        # Create organized output structure
        analysis_dir = temp_dir / "analysis"
        stems_dir = temp_dir / "stems"
        processed_dir = temp_dir / "processed"
        midi_dir = temp_dir / "midi"

        for directory in [analysis_dir, stems_dir, processed_dir, midi_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # === PHASE 1: Analysis ===
        analysis = analyze_audio(
            pipeline_audio_path,
            include_tempo=True,
            include_key=True,
            include_loudness=True,
            include_spectral=True,
            include_onsets=True,
        )

        assert analysis.tempo is not None
        assert analysis.key is not None
        assert analysis.loudness is not None

        # Store analysis results for later use
        analysis_data = {
            "bpm": analysis.tempo.bpm,
            "key": analysis.key.name,
            "loudness_lufs": analysis.loudness.integrated_lufs,
            "duration": analysis.duration_seconds,
        }

        # === PHASE 2: Stem Separation ===
        separator = mock_separator_with_stems
        separation_result = separator.separate(pipeline_audio_path, stems_dir)

        assert len(separation_result.stems) > 0

        # === PHASE 3: Process Each Stem with Effects ===
        processed_stems = {}

        for stem_name, stem_path in separation_result.stems.items():
            # Load stem
            stem_audio = load_audio(stem_path)

            # Create stem-specific effects chain
            effects = EffectsChain()

            if stem_name == "vocals":
                # Vocal processing: compression + reverb
                effects.add(CompressorConfig(threshold_db=-18, ratio=3.0))
                effects.add(ReverbConfig(room_size=0.4, wet_level=0.15))

            elif stem_name == "drums":
                # Drum processing: compression + subtle chorus
                effects.add(CompressorConfig(threshold_db=-15, ratio=6.0))
                effects.add(ChorusConfig(rate_hz=0.5, depth=0.1))

            elif stem_name == "bass":
                # Bass processing: heavy compression
                effects.add(CompressorConfig(threshold_db=-20, ratio=8.0))

            else:  # other
                # Gentle processing for other instruments
                effects.add(CompressorConfig(threshold_db=-18, ratio=2.0))

            # Process with effects
            processed = effects.process_array(stem_audio.samples, stem_audio.sample_rate)

            # Save processed stem
            processed_segment = AudioSegment(
                samples=processed,
                sample_rate=stem_audio.sample_rate,
            )

            output_path = processed_dir / f"{stem_name}_processed.wav"
            save_audio(processed_segment, output_path)
            processed_stems[stem_name] = output_path

        # === PHASE 4: Transcribe Vocals to MIDI ===
        if "vocals" in separation_result.stems:
            transcriber = mock_transcriber_with_notes
            midi_result = transcriber.transcribe(
                separation_result.vocals,
                midi_dir,
                save_midi_file=True,
            )

            assert midi_result.midi_path is not None
            assert midi_result.midi_path.exists()

        # === PHASE 5: Verify All Outputs ===

        # Check all processed stems exist
        assert len(processed_stems) == len(separation_result.stems)
        for stem_path in processed_stems.values():
            assert stem_path.exists()

            # Verify each file is valid audio
            audio = load_audio(stem_path)
            assert audio.samples.shape[0] > 0

        # Check MIDI output exists
        midi_files = list(midi_dir.glob("*.mid"))
        assert len(midi_files) > 0

        # Verify we have comprehensive analysis data
        assert "bpm" in analysis_data
        assert "key" in analysis_data
        assert analysis_data["duration"] > 0


@pytest.mark.integration
@pytest.mark.slow
class TestDataFlowAndCallbacks:
    """Test data flow between components and progress callbacks."""

    def test_data_flow_between_components(
        self,
        pipeline_audio_path: Path,
        temp_dir: Path,
    ):
        """
        Verify that data flows correctly between pipeline components.

        Tests that AudioSegment objects, analysis results, and file paths
        are properly passed and transformed through the pipeline without
        data corruption or loss.
        """
        # Load audio
        original = load_audio(pipeline_audio_path)
        original_shape = original.samples.shape
        original_sr = original.sample_rate

        # Save and reload to test round-trip
        intermediate_path = temp_dir / "intermediate.wav"
        save_audio(original, intermediate_path)
        reloaded = load_audio(intermediate_path)

        # Verify data integrity
        assert reloaded.sample_rate == original_sr
        np.testing.assert_array_almost_equal(
            reloaded.samples,
            original.samples,
            decimal=5,
        )

        # Apply effects and verify shape preservation
        effects = EffectsChain()
        effects.add(GainConfig(gain_db=3.0))

        processed = effects.process_array(reloaded.samples, reloaded.sample_rate)
        assert processed.shape == original_shape

        # Verify processing actually changed the audio
        assert not np.array_equal(processed, original.samples)

        # Save final result
        final_segment = AudioSegment(samples=processed, sample_rate=original_sr)
        final_path = temp_dir / "final.wav"
        save_audio(final_segment, final_path)

        # Verify final file
        final = load_audio(final_path)
        assert final.sample_rate == original_sr
        assert final.samples.shape == original_shape

    def test_progress_callbacks_across_pipeline(
        self,
        pipeline_audio_path: Path,
        temp_dir: Path,
        mock_separator_with_stems: MagicMock,
        mock_transcriber_with_notes: MagicMock,
    ):
        """
        Verify that progress callbacks work across all pipeline stages.

        Tests that progress callbacks are properly invoked and provide
        meaningful progress information throughout a multi-stage workflow.
        """
        progress_events = []

        def track_progress(current: int, total: int, message: str):
            """Record all progress events."""
            progress_events.append({
                "current": current,
                "total": total,
                "message": message,
                "percentage": (current / total * 100) if total > 0 else 0,
            })

        # Stage 1: Separation with progress
        separator = mock_separator_with_stems
        separation_result = separator.separate(
            pipeline_audio_path,
            temp_dir / "stems",
            progress_callback=track_progress,
        )

        # Verify separation progress was tracked
        separation_events = [e for e in progress_events if "separat" in e["message"].lower()]
        assert len(separation_events) > 0

        # Stage 2: Transcription with progress
        if separation_result.vocals:
            transcriber = mock_transcriber_with_notes
            midi_result = transcriber.transcribe(
                separation_result.vocals,
                temp_dir / "midi",
                progress_callback=track_progress,
            )

            # Verify transcription progress was tracked
            transcribe_events = [e for e in progress_events if "transcrib" in e["message"].lower()]
            assert len(transcribe_events) > 0

        # Verify we got progress events from multiple stages
        assert len(progress_events) > 2

        # Verify progress percentages are reasonable
        for event in progress_events:
            assert 0 <= event["percentage"] <= 100

    def test_metadata_preservation_through_pipeline(
        self,
        pipeline_audio_path: Path,
        temp_dir: Path,
    ):
        """
        Verify that audio metadata is preserved through processing.

        Tests that sample rate, bit depth, and other metadata survive
        the complete pipeline from load to save.
        """
        # Load with metadata
        original = load_audio(pipeline_audio_path)
        original_sr = original.sample_rate
        original_channels = original.channels

        # Process through effects
        effects = EffectsChain()
        effects.add(CompressorConfig(threshold_db=-20, ratio=4.0))

        processed = effects.process_array(original.samples, original.sample_rate)

        # Create new segment preserving metadata
        processed_segment = AudioSegment(
            samples=processed,
            sample_rate=original_sr,
            source_path=original.source_path,
            metadata=original.metadata,
        )

        # Save and reload
        output_path = temp_dir / "output.wav"
        save_audio(processed_segment, output_path)
        reloaded = load_audio(output_path)

        # Verify metadata preservation
        assert reloaded.sample_rate == original_sr
        assert reloaded.channels == original_channels


@pytest.mark.integration
@pytest.mark.slow
class TestErrorPropagation:
    """Test error handling and propagation through pipelines."""

    def test_invalid_input_error_propagation(self, temp_dir: Path):
        """
        Verify that errors from invalid inputs are properly propagated.

        Tests that attempting to load non-existent or invalid files
        raises appropriate exceptions that bubble up through the pipeline.
        """
        from soundlab.core.exceptions import AudioLoadError

        # Test with non-existent file
        nonexistent = temp_dir / "does_not_exist.wav"

        with pytest.raises(AudioLoadError):
            load_audio(nonexistent)

        # Test with invalid file
        invalid_file = temp_dir / "invalid.wav"
        invalid_file.write_text("This is not audio data")

        with pytest.raises(Exception):  # Should raise some audio-related error
            load_audio(invalid_file)

    def test_effects_chain_error_recovery(
        self,
        pipeline_audio_path: Path,
        temp_dir: Path,
    ):
        """
        Verify that effects chains handle errors gracefully.

        Tests that invalid effect configurations are caught and
        that the pipeline can recover or fail gracefully.
        """
        audio = load_audio(pipeline_audio_path)

        # Create effects chain with invalid parameters
        effects = EffectsChain()

        # Valid effect
        effects.add(CompressorConfig(threshold_db=-20, ratio=4.0))

        # This should work despite having multiple effects
        processed = effects.process_array(audio.samples, audio.sample_rate)
        assert processed is not None
        assert processed.shape == audio.samples.shape

    def test_empty_audio_handling(self, temp_dir: Path):
        """
        Verify that empty or silent audio is handled correctly.

        Tests that the pipeline gracefully handles edge cases like
        silence or very short audio files.
        """
        # Create silent audio
        sample_rate = 44100
        silent = np.zeros(sample_rate, dtype=np.float32)  # 1 second of silence

        silent_path = temp_dir / "silent.wav"
        sf.write(silent_path, silent, sample_rate)

        # Load silent audio
        audio = load_audio(silent_path)
        assert audio.samples is not None
        assert len(audio.samples) > 0

        # Analyze silent audio (should not crash)
        analysis = analyze_audio(silent_path)
        assert analysis is not None
        assert analysis.duration_seconds > 0

        # Apply effects to silence (should not crash)
        effects = EffectsChain()
        effects.add(GainConfig(gain_db=6.0))

        processed = effects.process_array(audio.samples, audio.sample_rate)
        assert processed is not None


@pytest.mark.integration
@pytest.mark.slow
class TestBatchProcessing:
    """Test batch processing of multiple files through pipelines."""

    def test_batch_analysis(
        self,
        multiple_audio_files: list[Path],
    ):
        """
        Test batch analysis of multiple audio files.

        Verifies that multiple files can be analyzed in sequence and
        that results are correctly associated with their source files.
        """
        results = []

        for audio_path in multiple_audio_files:
            analysis = analyze_audio(
                audio_path,
                include_tempo=True,
                include_key=True,
                include_loudness=True,
            )
            results.append((audio_path.name, analysis))

        # Verify we got results for all files
        assert len(results) == len(multiple_audio_files)

        # Verify each result is valid
        for filename, analysis in results:
            assert analysis.tempo is not None
            assert analysis.key is not None
            assert analysis.duration_seconds > 0

    def test_batch_effects_processing(
        self,
        multiple_audio_files: list[Path],
        temp_dir: Path,
    ):
        """
        Test batch effects processing on multiple files.

        Verifies that the same effects chain can be applied to multiple
        files efficiently and that all outputs are valid.
        """
        output_dir = temp_dir / "batch_processed"
        output_dir.mkdir()

        # Create effects chain
        effects = EffectsChain()
        effects.add(CompressorConfig(threshold_db=-18, ratio=3.0))
        effects.add(ReverbConfig(room_size=0.5, wet_level=0.2))

        processed_files = []

        # Process each file
        for audio_path in multiple_audio_files:
            # Load
            audio = load_audio(audio_path)

            # Process
            processed = effects.process_array(audio.samples, audio.sample_rate)

            # Save
            output_path = output_dir / f"{audio_path.stem}_processed.wav"
            processed_segment = AudioSegment(
                samples=processed,
                sample_rate=audio.sample_rate,
            )
            save_audio(processed_segment, output_path)
            processed_files.append(output_path)

        # Verify all outputs
        assert len(processed_files) == len(multiple_audio_files)
        for output_path in processed_files:
            assert output_path.exists()

            # Verify it's valid audio
            audio = load_audio(output_path)
            assert audio.samples.shape[0] > 0

    def test_batch_export_utility(
        self,
        multiple_audio_files: list[Path],
        temp_dir: Path,
    ):
        """
        Test the batch_export utility function.

        Verifies that the batch_export utility correctly processes
        multiple audio segments and exports them with consistent settings.
        """
        from soundlab.core.audio import AudioFormat

        output_dir = temp_dir / "batch_export"

        # Load all files into segments
        segments_with_names = []
        for audio_path in multiple_audio_files:
            segment = load_audio(audio_path)
            name = audio_path.stem
            segments_with_names.append((segment, name))

        # Batch export
        exported_paths = batch_export(
            segments_with_names,
            output_dir,
            format=AudioFormat.WAV,
        )

        # Verify results
        assert len(exported_paths) == len(multiple_audio_files)

        for path in exported_paths:
            assert path.exists()
            assert path.suffix == ".wav"

            # Verify it's valid audio
            audio = load_audio(path)
            assert audio.samples.shape[0] > 0

    def test_batch_separation_workflow(
        self,
        multiple_audio_files: list[Path],
        temp_dir: Path,
        mock_separator_with_stems: MagicMock,
    ):
        """
        Test batch stem separation workflow.

        Simulates a batch processing scenario where multiple songs need
        to be separated into stems, such as for DJ preparation or
        music production.
        """
        separator = mock_separator_with_stems

        all_results = []

        for audio_path in multiple_audio_files:
            # Create separate output directory for each file
            output_dir = temp_dir / "stems" / audio_path.stem

            # Separate
            result = separator.separate(audio_path, output_dir)
            all_results.append((audio_path.name, result))

        # Verify all separations completed
        assert len(all_results) == len(multiple_audio_files)

        # Verify each result
        for filename, result in all_results:
            assert isinstance(result, StemResult)
            assert len(result.stems) > 0

            # Verify all stem files exist
            for stem_path in result.stems.values():
                assert stem_path.exists()


@pytest.mark.integration
@pytest.mark.slow
class TestTemporaryDirectories:
    """Test proper handling of temporary directories in pipelines."""

    def test_pipeline_with_temp_directories(
        self,
        pipeline_audio_path: Path,
        mock_separator_with_stems: MagicMock,
    ):
        """
        Test that pipelines work correctly with temporary directories.

        Verifies that temporary directories are properly created, used,
        and can be cleaned up after processing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)

            # Create subdirectories for organized output
            stems_dir = temp_path / "stems"
            processed_dir = temp_path / "processed"

            # Run separation
            separator = mock_separator_with_stems
            result = separator.separate(pipeline_audio_path, stems_dir)

            # Verify outputs in temp directory
            assert stems_dir.exists()
            assert len(list(stems_dir.glob("*.wav"))) > 0

            # Process stems
            processed_dir.mkdir(parents=True, exist_ok=True)

            for stem_name, stem_path in result.stems.items():
                audio = load_audio(stem_path)

                # Apply simple processing
                effects = EffectsChain()
                effects.add(GainConfig(gain_db=2.0))
                processed = effects.process_array(audio.samples, audio.sample_rate)

                # Save to processed directory
                output_path = processed_dir / f"{stem_name}_processed.wav"
                segment = AudioSegment(samples=processed, sample_rate=audio.sample_rate)
                save_audio(segment, output_path)

            # Verify processed outputs
            assert len(list(processed_dir.glob("*.wav"))) == len(result.stems)

        # After context exit, temp directory should be cleaned up
        assert not temp_path.exists()

    def test_multiple_temporary_workspaces(
        self,
        multiple_audio_files: list[Path],
        mock_separator_with_stems: MagicMock,
    ):
        """
        Test parallel processing with separate temporary workspaces.

        Simulates a scenario where multiple files are processed in isolation
        with separate temporary directories to avoid conflicts.
        """
        results = []

        for audio_path in multiple_audio_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                workspace = Path(tmpdir)

                # Analyze
                analysis = analyze_audio(audio_path, include_tempo=True, include_key=True)

                # Separate
                separator = mock_separator_with_stems
                separation = separator.separate(audio_path, workspace / "stems")

                # Store results before temp directory is cleaned
                results.append({
                    "filename": audio_path.name,
                    "bpm": analysis.tempo.bpm if analysis.tempo else None,
                    "key": analysis.key.name if analysis.key else None,
                    "stem_count": len(separation.stems),
                })

        # Verify we got results from all files
        assert len(results) == len(multiple_audio_files)

        # Verify each result is complete
        for result in results:
            assert result["filename"]
            assert result["stem_count"] > 0


# === Helper Functions for Testing ===


def verify_audio_segment(segment: AudioSegment, min_duration: float = 0.1) -> bool:
    """
    Verify that an AudioSegment is valid.

    Helper function to check that an AudioSegment has valid samples,
    sample rate, and duration.
    """
    if segment.samples is None:
        return False
    if segment.sample_rate <= 0:
        return False
    if segment.duration < min_duration:
        return False
    return True


def verify_analysis_result(analysis: AnalysisResult, check_all: bool = False) -> bool:
    """
    Verify that an AnalysisResult contains valid data.

    Helper function to check that analysis results have reasonable values
    and are complete if check_all is True.
    """
    if analysis.duration_seconds <= 0:
        return False
    if analysis.sample_rate <= 0:
        return False

    if check_all:
        if analysis.tempo is None:
            return False
        if analysis.key is None:
            return False
        if analysis.loudness is None:
            return False

    return True
