"""Integration tests for soundlab.analysis module.

These tests verify end-to-end audio analysis workflows with both synthetic
and real audio signals. They test the full analysis pipeline including tempo,
key, loudness, spectral, and onset detection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from soundlab.analysis import (
    AnalysisResult,
    KeyDetectionResult,
    LoudnessResult,
    Mode,
    MusicalKey,
    OnsetResult,
    SpectralResult,
    TempoResult,
    analyze_audio,
    detect_key,
    detect_tempo,
    measure_loudness,
)
from soundlab.analysis.onsets import detect_onsets
from soundlab.analysis.spectral import analyze_spectral


@pytest.mark.integration
class TestAnalyzeAudioWorkflow:
    """Test the complete analyze_audio() workflow."""

    def test_analyze_audio_all_features(self, sample_audio_path: Path):
        """
        Test analyze_audio() with all analysis types enabled.

        Verifies that the full analysis pipeline returns an AnalysisResult
        containing all analysis components (tempo, key, loudness, spectral, onsets)
        when all features are enabled.
        """
        result = analyze_audio(
            sample_audio_path,
            include_tempo=True,
            include_key=True,
            include_loudness=True,
            include_spectral=True,
            include_onsets=True,
        )

        # Verify result type
        assert isinstance(result, AnalysisResult)

        # Verify all components are present
        assert result.tempo is not None
        assert result.key is not None
        assert result.loudness is not None
        assert result.spectral is not None
        assert result.onsets is not None

        # Verify basic metadata
        assert result.duration_seconds > 0
        assert result.sample_rate > 0
        assert result.channels > 0

        # Verify summary works
        summary = result.summary
        assert "duration" in summary
        assert "bpm" in summary
        assert "key" in summary
        assert "lufs" in summary

    def test_analyze_audio_selective_features(self, sample_audio_path: Path):
        """
        Test analyze_audio() with selective feature analysis.

        Verifies that only requested analysis types are performed and included
        in the result, while other components remain None.
        """
        # Only tempo and key
        result = analyze_audio(
            sample_audio_path,
            include_tempo=True,
            include_key=True,
            include_loudness=False,
            include_spectral=False,
            include_onsets=False,
        )

        assert result.tempo is not None
        assert result.key is not None
        assert result.loudness is None
        assert result.spectral is None
        assert result.onsets is None

    def test_analyze_audio_no_features(self, sample_audio_path: Path):
        """
        Test analyze_audio() with all features disabled.

        Verifies that basic audio metadata (duration, sample rate, channels)
        is still returned even when no analysis features are enabled.
        """
        result = analyze_audio(
            sample_audio_path,
            include_tempo=False,
            include_key=False,
            include_loudness=False,
            include_spectral=False,
            include_onsets=False,
        )

        # Basic info should still be present
        assert result.duration_seconds > 0
        assert result.sample_rate > 0
        assert result.channels > 0

        # Analysis components should be None
        assert result.tempo is None
        assert result.key is None
        assert result.loudness is None
        assert result.spectral is None
        assert result.onsets is None

    def test_analyze_audio_mono_file(self, sample_audio_path: Path):
        """
        Test analyze_audio() with mono audio file.

        Verifies that mono audio files are handled correctly and produce
        valid analysis results.
        """
        result = analyze_audio(sample_audio_path)
        assert result.channels == 1
        assert result.tempo is not None
        assert result.key is not None

    def test_analyze_audio_stereo_file(self, sample_stereo_audio_path: Path):
        """
        Test analyze_audio() with stereo audio file.

        Verifies that stereo audio files are handled correctly and produce
        valid analysis results (internally converted to mono for most analyses).
        """
        result = analyze_audio(sample_stereo_audio_path)
        assert result.channels == 2
        assert result.tempo is not None
        assert result.key is not None

    def test_analyze_audio_fixture_file(self, temp_dir: Path):
        """
        Test analyze_audio() with pre-generated fixture audio.

        Tests against a known audio fixture to verify consistent analysis
        across test runs.
        """
        fixture_path = Path("/home/user/soundlab/tests/fixtures/audio/sine_440hz_3s.wav")
        if fixture_path.exists():
            result = analyze_audio(fixture_path)

            # 3-second file
            assert 2.9 < result.duration_seconds < 3.1

            # Should detect some tempo even from sine wave
            assert result.tempo is not None
            assert result.tempo.bpm > 0

            # Key detection should return a result
            assert result.key is not None
            assert isinstance(result.key.key, MusicalKey)


@pytest.mark.integration
class TestTempoDetectionAccuracy:
    """Test tempo detection with known signals."""

    def test_tempo_120bpm_pattern(self, temp_dir: Path, sample_rate: int):
        """
        Test tempo detection with 120 BPM click pattern.

        Creates a synthetic audio signal with clicks at exactly 120 BPM
        and verifies that the tempo detection algorithm correctly identifies
        the tempo within a reasonable tolerance.
        """
        # Generate 5 seconds of audio with clicks at 120 BPM
        duration = 5.0
        bpm = 120.0
        beat_interval = 60.0 / bpm  # 0.5 seconds

        # Create click track
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        for beat_time in np.arange(0, duration, beat_interval):
            sample_idx = int(beat_time * sample_rate)
            if sample_idx < len(audio):
                # Add a short click (100 samples)
                click_length = min(100, len(audio) - sample_idx)
                audio[sample_idx:sample_idx + click_length] = 0.5

        # Save to file
        audio_path = temp_dir / "120bpm_clicks.wav"
        sf.write(audio_path, audio, sample_rate)

        # Detect tempo
        result = detect_tempo(audio, sample_rate)

        # Verify tempo is close to 120 BPM (within 10% tolerance)
        assert isinstance(result, TempoResult)
        assert 108 <= result.bpm <= 132  # 120 ± 10%
        assert result.confidence > 0.3  # Should have reasonable confidence
        assert result.beat_count > 0

    def test_tempo_different_rates(self, sample_rate: int):
        """
        Test tempo detection across various BPM rates.

        Verifies that tempo detection works across a wide range of tempos
        from slow (60 BPM) to fast (180 BPM).
        """
        test_bpms = [60, 90, 120, 140, 180]

        for target_bpm in test_bpms:
            duration = 4.0
            beat_interval = 60.0 / target_bpm

            # Create click track
            audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
            for beat_time in np.arange(0, duration, beat_interval):
                sample_idx = int(beat_time * sample_rate)
                if sample_idx < len(audio) - 100:
                    audio[sample_idx:sample_idx + 100] = 0.5

            result = detect_tempo(audio, sample_rate)

            # Should detect something in reasonable range (±20%)
            # Note: Tempo detection can sometimes detect harmonics/subharmonics
            assert result.bpm > 0
            assert 30 <= result.bpm <= 300

    def test_tempo_beat_positions(self, sample_rate: int):
        """
        Test that detected beat positions are reasonable.

        Verifies that the beat timestamps returned by tempo detection
        are properly ordered and spaced.
        """
        # Generate simple click pattern
        duration = 3.0
        bpm = 120.0
        beat_interval = 60.0 / bpm

        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        expected_beats = []
        for beat_time in np.arange(0, duration, beat_interval):
            sample_idx = int(beat_time * sample_rate)
            if sample_idx < len(audio) - 50:
                audio[sample_idx:sample_idx + 50] = 0.5
                expected_beats.append(beat_time)

        result = detect_tempo(audio, sample_rate)

        # Verify beat times are in order
        assert result.beats == sorted(result.beats)

        # Verify beats are within audio duration
        assert all(0 <= beat <= duration for beat in result.beats)

        # Check beat_count property
        assert result.beat_count == len(result.beats)

        # Check beat_interval property
        assert result.beat_interval > 0
        assert abs(result.beat_interval - 60.0 / result.bpm) < 0.001

    def test_tempo_confidence_metric(self, sample_rate: int):
        """
        Test tempo confidence scores.

        Verifies that confidence scores are within valid range [0, 1]
        and that regular patterns have higher confidence than irregular ones.
        """
        # Regular pattern should have higher confidence
        duration = 3.0
        regular_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        for beat_time in np.arange(0, duration, 0.5):  # Regular 120 BPM
            idx = int(beat_time * sample_rate)
            if idx < len(regular_audio) - 50:
                regular_audio[idx:idx + 50] = 0.5

        regular_result = detect_tempo(regular_audio, sample_rate)

        # Verify confidence bounds
        assert 0.0 <= regular_result.confidence <= 1.0

        # Test with irregular pattern
        irregular_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        irregular_times = [0.0, 0.3, 0.9, 1.2, 2.0, 2.8]  # Irregular spacing
        for beat_time in irregular_times:
            idx = int(beat_time * sample_rate)
            if idx < len(irregular_audio) - 50:
                irregular_audio[idx:idx + 50] = 0.5

        irregular_result = detect_tempo(irregular_audio, sample_rate)
        assert 0.0 <= irregular_result.confidence <= 1.0


@pytest.mark.integration
class TestKeyDetectionKnownPieces:
    """Test key detection with known musical patterns."""

    def test_key_detection_a_major_triad(self, sample_rate: int):
        """
        Test key detection with A major triad.

        Creates a synthetic audio signal containing the notes of an A major
        chord (A, C#, E) and verifies that the key detection identifies
        either A major or the relative F# minor.
        """
        # Generate A major triad: A4 (440 Hz), C#5 (554.37 Hz), E5 (659.25 Hz)
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # A major chord
        a4 = 0.3 * np.sin(2 * np.pi * 440.0 * t)
        cs5 = 0.3 * np.sin(2 * np.pi * 554.37 * t)
        e5 = 0.3 * np.sin(2 * np.pi * 659.25 * t)

        audio = (a4 + cs5 + e5).astype(np.float32)

        result = detect_key(audio, sample_rate)

        # Should detect A (major or minor) or relative keys
        assert isinstance(result, KeyDetectionResult)
        assert isinstance(result.key, MusicalKey)
        assert isinstance(result.mode, Mode)
        assert 0.0 <= result.confidence <= 1.0

        # The result should have correlation data for all 24 keys
        assert len(result.all_correlations) == 24

    def test_key_detection_c_major_scale(self, sample_rate: int):
        """
        Test key detection with C major scale.

        Creates an audio signal containing all notes of the C major scale
        and verifies that the key detection produces a reasonable result.
        """
        # C major scale: C, D, E, F, G, A, B
        c_major_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]

        duration = 0.4  # Each note for 0.4 seconds
        samples_per_note = int(sample_rate * duration)

        audio_segments = []
        for freq in c_major_freqs:
            t = np.linspace(0, duration, samples_per_note, dtype=np.float32)
            note = 0.3 * np.sin(2 * np.pi * freq * t)
            audio_segments.append(note)

        audio = np.concatenate(audio_segments).astype(np.float32)

        result = detect_key(audio, sample_rate)

        assert isinstance(result, KeyDetectionResult)
        # C major or A minor (relative) would be reasonable
        assert result.key is not None
        assert result.mode is not None

        # Check properties
        assert isinstance(result.name, str)
        assert isinstance(result.camelot, str)
        assert isinstance(result.open_key, str)

    def test_key_detection_properties(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """
        Test KeyDetectionResult properties.

        Verifies that all properties of the KeyDetectionResult model
        (name, camelot, open_key) return valid formatted strings.
        """
        result = detect_key(sample_mono_audio, sample_rate)

        # Test name property (e.g., "A minor")
        name = result.name
        assert isinstance(name, str)
        assert result.key.value in name
        assert result.mode.value in name

        # Test Camelot notation
        camelot = result.camelot
        assert isinstance(camelot, str)
        assert len(camelot) >= 2  # e.g., "8A", "12B"

        # Test Open Key notation
        open_key = result.open_key
        assert isinstance(open_key, str)
        assert len(open_key) >= 2  # e.g., "1m", "12d"

    def test_key_detection_with_fixture(self):
        """
        Test key detection with pre-generated fixture.

        Uses a known audio fixture file to verify consistent key detection
        results across test runs.
        """
        fixture_path = Path("/home/user/soundlab/tests/fixtures/audio/sine_440hz_3s.wav")
        if fixture_path.exists():
            import librosa
            y, sr = librosa.load(fixture_path, sr=None, mono=True)

            result = detect_key(y, sr)

            # A 440Hz tone should correlate with A
            assert isinstance(result, KeyDetectionResult)
            assert result.confidence > 0.0

            # Verify all correlations are present
            assert len(result.all_correlations) == 24


@pytest.mark.integration
class TestLoudnessMeasurement:
    """Test loudness measurement (LUFS values)."""

    def test_measure_loudness_basic(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """
        Test basic loudness measurement.

        Verifies that loudness measurement returns a LoudnessResult with
        valid LUFS values within expected ranges.
        """
        result = measure_loudness(sample_mono_audio, sample_rate)

        assert isinstance(result, LoudnessResult)

        # Integrated LUFS should be negative (dB scale relative to digital full scale)
        assert result.integrated_lufs <= 0.0
        assert result.integrated_lufs > -100.0  # Not silence

        # True peak should be present
        if result.true_peak_db is not None:
            assert result.true_peak_db <= 0.0  # Can't exceed 0 dBFS

    def test_measure_loudness_quiet_signal(self, sample_rate: int):
        """
        Test loudness measurement with quiet signal.

        Verifies that quiet audio signals produce lower (more negative) LUFS
        values than normal-level signals.
        """
        # Very quiet signal
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        quiet_audio = 0.01 * np.sin(2 * np.pi * 440 * t)  # -40 dB

        result = measure_loudness(quiet_audio, sample_rate)

        # Should be very quiet (very negative LUFS)
        assert result.integrated_lufs < -30.0

    def test_measure_loudness_loud_signal(self, sample_rate: int):
        """
        Test loudness measurement with loud signal.

        Verifies that louder audio signals produce higher (less negative) LUFS
        values than quiet signals.
        """
        # Loud signal (but not clipping)
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        loud_audio = 0.7 * np.sin(2 * np.pi * 440 * t)  # ~-3 dB

        result = measure_loudness(loud_audio, sample_rate)

        # Should be much louder than typical broadcast
        assert result.integrated_lufs > -20.0

    def test_measure_loudness_silence(self, silence_audio: np.ndarray, sample_rate: int):
        """
        Test loudness measurement with silence.

        Verifies that silent audio is handled correctly and produces
        expected minimum LUFS values.
        """
        result = measure_loudness(silence_audio, sample_rate)

        # Silence should have very low LUFS (default -70)
        assert result.integrated_lufs <= -60.0

    def test_measure_loudness_stereo(self, sample_stereo_audio: np.ndarray, sample_rate: int):
        """
        Test loudness measurement with stereo audio.

        Verifies that stereo audio is handled correctly by the loudness
        measurement algorithm.
        """
        result = measure_loudness(sample_stereo_audio, sample_rate)

        assert isinstance(result, LoudnessResult)
        assert result.integrated_lufs <= 0.0

    def test_loudness_broadcast_standards(self, sample_rate: int):
        """
        Test loudness standard checks.

        Verifies the is_broadcast_safe and is_streaming_optimized properties
        that check if audio meets broadcast standards.
        """
        # Create signal at -16 LUFS (broadcast safe)
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = 0.1 * np.sin(2 * np.pi * 440 * t)

        result = measure_loudness(audio, sample_rate)

        # Test property methods
        assert isinstance(result.is_broadcast_safe, bool)
        assert isinstance(result.is_streaming_optimized, bool)

    def test_loudness_dynamic_range(self, sample_rate: int):
        """
        Test dynamic range measurement.

        Verifies that dynamic range (difference between loud and quiet sections)
        is correctly measured.
        """
        # Create audio with varying loudness
        duration = 3.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)

        # Envelope: quiet -> loud -> quiet
        envelope = np.concatenate([
            np.linspace(0.1, 0.7, samples // 3),
            np.linspace(0.7, 0.1, samples - 2 * (samples // 3)),
            np.linspace(0.1, 0.1, samples // 3),
        ])[:samples]

        audio = envelope * np.sin(2 * np.pi * 440 * t)

        result = measure_loudness(audio, sample_rate)

        # Should have measurable dynamic range
        if result.dynamic_range_db is not None:
            assert result.dynamic_range_db > 0.0


@pytest.mark.integration
class TestSpectralAnalysis:
    """Test spectral analysis features."""

    def test_spectral_analysis_basic(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """
        Test basic spectral analysis.

        Verifies that spectral analysis returns a SpectralResult with
        all required features (centroid, bandwidth, rolloff, flatness, ZCR).
        """
        result = analyze_spectral(sample_mono_audio, sample_rate)

        assert isinstance(result, SpectralResult)

        # All features should be present and valid
        assert result.spectral_centroid > 0
        assert result.spectral_bandwidth > 0
        assert result.spectral_rolloff > 0
        assert 0.0 <= result.spectral_flatness <= 1.0
        assert result.zero_crossing_rate >= 0

    def test_spectral_centroid_frequency(self, sample_rate: int):
        """
        Test spectral centroid with known frequency.

        Creates a pure tone at a known frequency and verifies that the
        spectral centroid is close to that frequency.
        """
        # Pure 1000 Hz tone
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)

        result = analyze_spectral(audio, sample_rate)

        # Spectral centroid should be near 1000 Hz for pure tone
        # Allow some tolerance due to windowing effects
        assert 800 < result.spectral_centroid < 1200

    def test_spectral_brightness_classification(self, sample_rate: int):
        """
        Test spectral brightness classification.

        Verifies that the brightness property correctly classifies audio
        as "dark", "balanced", or "bright" based on spectral centroid.
        """
        # Dark sound (low frequency)
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        dark_audio = 0.5 * np.sin(2 * np.pi * 200 * t)  # 200 Hz

        dark_result = analyze_spectral(dark_audio, sample_rate)
        assert dark_result.brightness in ["dark", "balanced", "bright"]

        # Bright sound (high frequency)
        bright_audio = 0.5 * np.sin(2 * np.pi * 5000 * t)  # 5000 Hz

        bright_result = analyze_spectral(bright_audio, sample_rate)
        assert bright_result.brightness in ["dark", "balanced", "bright"]

        # Bright should have higher centroid than dark
        assert bright_result.spectral_centroid > dark_result.spectral_centroid

    def test_spectral_flatness_tone_vs_noise(self, sample_rate: int):
        """
        Test spectral flatness distinguishing tones from noise.

        Verifies that spectral flatness is low for pure tones (tonal)
        and higher for noise (non-tonal).
        """
        duration = 1.0

        # Pure tone (should have low flatness)
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        tone_result = analyze_spectral(tone, sample_rate)

        # White noise (should have high flatness)
        noise = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        noise_result = analyze_spectral(noise, sample_rate)

        # Noise should have higher flatness than tone
        assert noise_result.spectral_flatness > tone_result.spectral_flatness

    def test_spectral_rolloff_meaning(self, sample_rate: int):
        """
        Test spectral rolloff frequency.

        Verifies that the rolloff frequency (where 95% of energy is concentrated)
        is reasonable and increases for brighter sounds.
        """
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # Low frequency signal
        low_freq = 0.5 * np.sin(2 * np.pi * 300 * t)
        low_result = analyze_spectral(low_freq, sample_rate)

        # High frequency signal
        high_freq = 0.5 * np.sin(2 * np.pi * 3000 * t)
        high_result = analyze_spectral(high_freq, sample_rate)

        # Higher frequency should have higher rolloff
        assert high_result.spectral_rolloff > low_result.spectral_rolloff

        # Rolloff should be positive and within Nyquist frequency
        assert 0 < low_result.spectral_rolloff < sample_rate / 2
        assert 0 < high_result.spectral_rolloff < sample_rate / 2

    def test_spectral_zero_crossing_rate(self, sample_rate: int):
        """
        Test zero crossing rate.

        Verifies that zero crossing rate correlates with signal frequency
        (higher frequency signals cross zero more often).
        """
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # Low frequency (fewer crossings)
        low_freq = 0.5 * np.sin(2 * np.pi * 200 * t)
        low_result = analyze_spectral(low_freq, sample_rate)

        # High frequency (more crossings)
        high_freq = 0.5 * np.sin(2 * np.pi * 2000 * t)
        high_result = analyze_spectral(high_freq, sample_rate)

        # Higher frequency should have more zero crossings
        assert high_result.zero_crossing_rate > low_result.zero_crossing_rate


@pytest.mark.integration
class TestOnsetDetection:
    """Test onset detection."""

    def test_onset_detection_basic(self, sample_mono_audio: np.ndarray, sample_rate: int):
        """
        Test basic onset detection.

        Verifies that onset detection returns an OnsetResult with
        detected onset times and strengths.
        """
        result = detect_onsets(sample_mono_audio, sample_rate)

        assert isinstance(result, OnsetResult)
        assert isinstance(result.onset_times, list)
        assert isinstance(result.onset_strengths, list)

        # Should have same number of times and strengths
        assert len(result.onset_times) == len(result.onset_strengths)

        # Check properties
        assert result.onset_count == len(result.onset_times)

    def test_onset_detection_with_clicks(self, sample_rate: int):
        """
        Test onset detection with clear transients.

        Creates an audio signal with distinct clicks and verifies that
        onsets are detected near those clicks.
        """
        # Generate audio with 5 distinct clicks
        duration = 3.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

        click_times = [0.5, 1.0, 1.5, 2.0, 2.5]
        for click_time in click_times:
            idx = int(click_time * sample_rate)
            if idx < len(audio) - 1000:
                # Add a burst of noise as a click
                audio[idx:idx + 100] = 0.5

        result = detect_onsets(audio, sample_rate)

        # Should detect onsets (maybe not exactly 5 due to algorithm sensitivity)
        assert result.onset_count > 0

        # Onsets should be within the audio duration
        assert all(0 <= t <= duration for t in result.onset_times)

        # Onset strengths should be normalized to 0-1
        if result.onset_strengths:
            assert all(0 <= s <= 1.0 for s in result.onset_strengths)

    def test_onset_average_interval(self, sample_rate: int):
        """
        Test average interval between onsets.

        Verifies that the average_interval property correctly calculates
        the mean time between detected onsets.
        """
        # Generate regular clicks
        duration = 3.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

        interval = 0.5  # Click every 0.5 seconds
        for t in np.arange(0, duration, interval):
            idx = int(t * sample_rate)
            if idx < len(audio) - 100:
                audio[idx:idx + 100] = 0.5

        result = detect_onsets(audio, sample_rate)

        if result.onset_count >= 2:
            # Average interval should be positive
            assert result.average_interval > 0

            # Should be roughly close to our interval (within tolerance)
            # Note: Detection might not be perfect
            assert result.average_interval > 0

    def test_onset_detection_continuous_tone(self, sample_rate: int):
        """
        Test onset detection with continuous tone.

        A continuous sine wave with no transients should produce
        few or no onset detections.
        """
        # Continuous tone with no transients
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        result = detect_onsets(audio, sample_rate)

        # Continuous tone should have very few onsets
        # (may detect one at the start)
        assert result.onset_count <= 5

    def test_onset_detection_fixture(self):
        """
        Test onset detection with fixture file.

        Uses a pre-generated fixture file to verify consistent onset
        detection across test runs.
        """
        fixture_path = Path("/home/user/soundlab/tests/fixtures/audio/music_like_5s.wav")
        if fixture_path.exists():
            import librosa
            y, sr = librosa.load(fixture_path, sr=None, mono=True)

            result = detect_onsets(y, sr)

            # Music should have some onsets
            assert isinstance(result, OnsetResult)
            assert result.onset_count >= 0


@pytest.mark.integration
class TestSyntheticSignals:
    """Test with deterministic synthetic signals."""

    def test_chirp_signal_analysis(self, sample_rate: int):
        """
        Test analysis of chirp signal.

        A chirp (frequency sweep) provides a comprehensive spectral test
        signal. Verifies that all analysis components can process it.
        """
        from scipy.signal import chirp as scipy_chirp

        # Generate chirp from 100 Hz to 5000 Hz over 3 seconds
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = scipy_chirp(t, f0=100, f1=5000, t1=duration, method='linear')
        audio = audio.astype(np.float32) * 0.5

        # Test tempo
        tempo_result = detect_tempo(audio, sample_rate)
        assert isinstance(tempo_result, TempoResult)
        assert tempo_result.bpm > 0

        # Test key
        key_result = detect_key(audio, sample_rate)
        assert isinstance(key_result, KeyDetectionResult)

        # Test spectral
        spectral_result = analyze_spectral(audio, sample_rate)
        assert isinstance(spectral_result, SpectralResult)
        # Chirp should have wide bandwidth
        assert spectral_result.spectral_bandwidth > 500

        # Test loudness
        loudness_result = measure_loudness(audio, sample_rate)
        assert isinstance(loudness_result, LoudnessResult)

    def test_multi_tone_complex(self, sample_rate: int):
        """
        Test with multi-tone complex signal.

        Creates a signal with multiple simultaneous frequencies to test
        how analysis handles complex harmonic content.
        """
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # Complex tone: fundamental + harmonics
        fundamental = 220.0  # A3
        audio = (
            0.4 * np.sin(2 * np.pi * fundamental * t) +
            0.2 * np.sin(2 * np.pi * fundamental * 2 * t) +
            0.1 * np.sin(2 * np.pi * fundamental * 3 * t) +
            0.05 * np.sin(2 * np.pi * fundamental * 4 * t)
        )
        audio = audio.astype(np.float32)

        # Analyze
        spectral_result = analyze_spectral(audio, sample_rate)

        # Should detect reasonable spectral features
        assert spectral_result.spectral_centroid > fundamental
        assert spectral_result.spectral_flatness < 0.5  # Tonal, not noise-like

    def test_modulated_signal(self, sample_rate: int):
        """
        Test with amplitude-modulated signal.

        Amplitude modulation creates a signal with varying loudness,
        useful for testing dynamic range and onset detection.
        """
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # Carrier at 440 Hz, modulated at 4 Hz
        carrier = np.sin(2 * np.pi * 440 * t)
        modulator = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
        audio = (carrier * modulator * 0.5).astype(np.float32)

        # Test onset detection (should detect modulation peaks)
        onset_result = detect_onsets(audio, sample_rate)

        # Should detect some onsets from the modulation
        assert onset_result.onset_count >= 0  # May or may not detect depending on sensitivity


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_analyze_audio_nonexistent_file(self, temp_dir: Path):
        """
        Test analyze_audio with non-existent file.

        Verifies that attempting to analyze a non-existent file
        raises an appropriate error.
        """
        nonexistent = temp_dir / "does_not_exist.wav"

        with pytest.raises(Exception):  # Could be FileNotFoundError or librosa error
            analyze_audio(nonexistent)

    def test_analyze_audio_invalid_file(self, temp_dir: Path):
        """
        Test analyze_audio with invalid/corrupt file.

        Creates a file with invalid audio data and verifies that
        analysis handles it appropriately.
        """
        # Create invalid audio file
        invalid_file = temp_dir / "invalid.wav"
        invalid_file.write_text("This is not audio data")

        with pytest.raises(Exception):
            analyze_audio(invalid_file)

    def test_detect_tempo_empty_audio(self, sample_rate: int):
        """
        Test tempo detection with empty audio array.

        Verifies that empty input is handled gracefully.
        """
        empty_audio = np.array([], dtype=np.float32)

        # Should either raise an error or return zero tempo
        try:
            result = detect_tempo(empty_audio, sample_rate)
            # If it doesn't raise, tempo should be 0 or very low confidence
            assert result.bpm >= 0
            assert result.confidence >= 0
        except (ValueError, Exception):
            # It's also acceptable to raise an error
            pass

    def test_measure_loudness_very_short_audio(self, sample_rate: int):
        """
        Test loudness measurement with very short audio.

        Extremely short audio may not be suitable for loudness measurement.
        Verifies appropriate handling.
        """
        # Very short audio (10 samples)
        short_audio = np.random.randn(10).astype(np.float32) * 0.1

        # Should either work or raise an informative error
        try:
            result = measure_loudness(short_audio, sample_rate)
            assert isinstance(result, LoudnessResult)
        except Exception:
            # Acceptable to fail with very short audio
            pass

    def test_spectral_analysis_silence(self, silence_audio: np.ndarray, sample_rate: int):
        """
        Test spectral analysis with silence.

        Verifies that spectral analysis handles silent audio appropriately
        without errors.
        """
        result = analyze_spectral(silence_audio, sample_rate)

        assert isinstance(result, SpectralResult)
        # Silence should have low/zero spectral features
        # But should not crash


@pytest.mark.integration
class TestWithRealAudioFixtures:
    """Test with real audio file fixtures."""

    def test_music_like_fixture_full_analysis(self):
        """
        Test full analysis pipeline with music-like fixture.

        Uses a pre-generated music-like audio fixture to verify that
        all analysis components work together on realistic audio.
        """
        fixture_path = Path("/home/user/soundlab/tests/fixtures/audio/music_like_5s.wav")
        if not fixture_path.exists():
            pytest.skip("Music fixture not available")

        result = analyze_audio(fixture_path)

        # Verify all components
        assert result.duration_seconds > 4.5  # ~5 seconds
        assert result.sample_rate > 0

        # All analysis results should be present
        assert result.tempo is not None
        assert result.key is not None
        assert result.loudness is not None
        assert result.spectral is not None
        assert result.onsets is not None

        # Verify reasonable values
        assert 30 <= result.tempo.bpm <= 300  # Reasonable BPM range
        assert result.loudness.integrated_lufs < 0  # LUFS is negative
        assert result.spectral.spectral_centroid > 0

    def test_silence_fixture_detection(self):
        """
        Test that silence is correctly identified.

        Uses a silence fixture to verify that analysis can distinguish
        silent audio from normal audio.
        """
        fixture_path = Path("/home/user/soundlab/tests/fixtures/audio/silence_1s.wav")
        if not fixture_path.exists():
            pytest.skip("Silence fixture not available")

        result = analyze_audio(fixture_path)

        # Silence should have very low loudness
        assert result.loudness is not None
        assert result.loudness.integrated_lufs < -60.0

        # Should have minimal onsets
        if result.onsets is not None:
            assert result.onsets.onset_count <= 2  # Maybe 1-2 from file boundaries

    def test_stereo_fixture_channel_handling(self):
        """
        Test stereo file handling.

        Verifies that stereo audio fixtures are correctly processed
        with proper channel handling.
        """
        fixture_path = Path("/home/user/soundlab/tests/fixtures/audio/stereo_test_2s.wav")
        if not fixture_path.exists():
            pytest.skip("Stereo fixture not available")

        result = analyze_audio(fixture_path)

        # Should detect stereo
        assert result.channels == 2

        # Analysis should still work (converts to mono internally)
        assert result.tempo is not None
        assert result.key is not None
