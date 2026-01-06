"""Audio analysis module for SoundLab."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from soundlab.analysis.key import (
    detect_key,
    get_compatible_keys,
    get_parallel_key,
    get_relative_key,
)
from soundlab.analysis.loudness import measure_loudness, normalize_loudness
from soundlab.analysis.models import (
    AnalysisResult,
    KeyDetectionResult,
    LoudnessResult,
    Mode,
    MusicalKey,
    OnsetResult,
    SpectralResult,
    TempoResult,
)
from soundlab.analysis.onsets import (
    detect_beats_and_onsets,
    detect_onsets,
    get_onset_strength_envelope,
    segment_by_onsets,
)
from soundlab.analysis.spectral import (
    analyze_spectral,
    compute_chroma,
    compute_mel_spectrogram,
    compute_mfcc,
    get_frequency_bands_energy,
)
from soundlab.analysis.tempo import detect_tempo, detect_tempo_with_alternatives

if TYPE_CHECKING:
    from soundlab.core.types import PathLike


def analyze_audio(
    audio_path: PathLike,
    *,
    include_tempo: bool = True,
    include_key: bool = True,
    include_loudness: bool = True,
    include_spectral: bool = True,
    include_onsets: bool = True,
) -> AnalysisResult:
    """
    Perform comprehensive audio analysis.

    Parameters
    ----------
    audio_path
        Path to audio file.
    include_tempo
        Include tempo/BPM detection.
    include_key
        Include key detection.
    include_loudness
        Include loudness analysis.
    include_spectral
        Include spectral analysis.
    include_onsets
        Include onset detection.

    Returns
    -------
    AnalysisResult
        Comprehensive analysis results.

    Examples
    --------
    >>> result = analyze_audio("song.mp3")
    >>> print(result.summary)
    {'bpm': '120.5', 'key': 'A minor', 'lufs': '-14.2', ...}
    """
    import librosa

    audio_path = Path(audio_path)
    logger.info(f"Analyzing: {audio_path}")

    # Load audio
    y, sr = librosa.load(audio_path, sr=None, mono=False)

    # Get basic info
    if y.ndim == 1:
        channels = 1
        duration = len(y) / sr
    else:
        channels = y.shape[0]
        duration = y.shape[1] / sr

    # Convert to mono for analysis
    if y.ndim > 1:
        y_mono = librosa.to_mono(y)
    else:
        y_mono = y

    # Run analyses
    tempo_result = None
    key_result = None
    loudness_result = None
    spectral_result = None
    onset_result = None

    if include_tempo:
        logger.debug("Detecting tempo...")
        tempo_result = detect_tempo(y_mono, sr)

    if include_key:
        logger.debug("Detecting key...")
        key_result = detect_key(y_mono, sr)

    if include_loudness:
        logger.debug("Measuring loudness...")
        loudness_result = measure_loudness(y, sr)

    if include_spectral:
        logger.debug("Analyzing spectral features...")
        spectral_result = analyze_spectral(y_mono, sr)

    if include_onsets:
        logger.debug("Detecting onsets...")
        onset_result = detect_onsets(y_mono, sr)

    logger.info(f"Analysis complete for {audio_path.name}")

    return AnalysisResult(
        duration_seconds=duration,
        sample_rate=sr,
        channels=channels,
        tempo=tempo_result,
        key=key_result,
        loudness=loudness_result,
        spectral=spectral_result,
        onsets=onset_result,
    )


__all__ = [
    # Main convenience function
    "analyze_audio",
    # Models
    "AnalysisResult",
    "KeyDetectionResult",
    "LoudnessResult",
    "Mode",
    "MusicalKey",
    "OnsetResult",
    "SpectralResult",
    "TempoResult",
    # Key detection
    "detect_key",
    "get_compatible_keys",
    "get_parallel_key",
    "get_relative_key",
    # Loudness
    "measure_loudness",
    "normalize_loudness",
    # Onsets
    "detect_beats_and_onsets",
    "detect_onsets",
    "get_onset_strength_envelope",
    "segment_by_onsets",
    # Spectral
    "analyze_spectral",
    "compute_chroma",
    "compute_mel_spectrogram",
    "compute_mfcc",
    "get_frequency_bands_energy",
    # Tempo
    "detect_tempo",
    "detect_tempo_with_alternatives",
]
