"""Spectral analysis using librosa."""

from __future__ import annotations

from typing import TYPE_CHECKING

import librosa
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from soundlab.analysis.models import SpectralResult

if TYPE_CHECKING:
    pass


__all__ = [
    "analyze_spectral",
    "compute_mel_spectrogram",
    "compute_mfcc",
    "compute_chroma",
    "get_frequency_bands_energy",
]


def analyze_spectral(
    y: NDArray[np.float32],
    sr: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> SpectralResult:
    """
    Compute spectral features from audio.

    Parameters
    ----------
    y
        Audio time series (mono).
    sr
        Sample rate.
    n_fft
        FFT window size.
    hop_length
        Hop length between frames.

    Returns
    -------
    SpectralResult
        Spectral features including centroid, bandwidth, rolloff.

    Examples
    --------
    >>> y, sr = librosa.load("song.mp3", sr=22050, mono=True)
    >>> result = analyze_spectral(y, sr)
    >>> print(f"Centroid: {result.spectral_centroid:.0f} Hz ({result.brightness})")
    """
    logger.debug(f"Analyzing spectral features (sr={sr}, n_fft={n_fft})")

    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Compute magnitude spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Spectral centroid (brightness indicator)
    centroid = librosa.feature.spectral_centroid(
        S=S,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    centroid_mean = float(np.mean(centroid))

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        S=S,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    bandwidth_mean = float(np.mean(bandwidth))

    # Spectral rolloff (95% energy threshold)
    rolloff = librosa.feature.spectral_rolloff(
        S=S,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        roll_percent=0.95,
    )
    rolloff_mean = float(np.mean(rolloff))

    # Spectral flatness (tonality indicator)
    flatness = librosa.feature.spectral_flatness(S=S)
    flatness_mean = float(np.mean(flatness))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    zcr_mean = float(np.mean(zcr))

    logger.debug(
        f"Spectral: centroid={centroid_mean:.0f}Hz, "
        f"bandwidth={bandwidth_mean:.0f}Hz, "
        f"rolloff={rolloff_mean:.0f}Hz"
    )

    return SpectralResult(
        spectral_centroid=centroid_mean,
        spectral_bandwidth=bandwidth_mean,
        spectral_rolloff=rolloff_mean,
        spectral_flatness=flatness_mean,
        zero_crossing_rate=zcr_mean,
    )


def compute_mel_spectrogram(
    y: NDArray[np.float32],
    sr: int,
    *,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> NDArray[np.float32]:
    """
    Compute mel spectrogram.

    Parameters
    ----------
    y
        Audio time series.
    sr
        Sample rate.
    n_mels
        Number of mel bands.
    n_fft
        FFT window size.
    hop_length
        Hop length.
    fmin
        Minimum frequency.
    fmax
        Maximum frequency (default: sr/2).

    Returns
    -------
    NDArray[np.float32]
        Mel spectrogram (n_mels, time).
    """
    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )

    return mel_spec.astype(np.float32)


def compute_mfcc(
    y: NDArray[np.float32],
    sr: int,
    *,
    n_mfcc: int = 13,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> NDArray[np.float32]:
    """
    Compute Mel-frequency cepstral coefficients.

    Parameters
    ----------
    y
        Audio time series.
    sr
        Sample rate.
    n_mfcc
        Number of MFCCs to return.
    n_mels
        Number of mel bands.
    n_fft
        FFT window size.
    hop_length
        Hop length.

    Returns
    -------
    NDArray[np.float32]
        MFCCs (n_mfcc, time).
    """
    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    return mfcc.astype(np.float32)


def compute_chroma(
    y: NDArray[np.float32],
    sr: int,
    *,
    n_chroma: int = 12,
    hop_length: int = 512,
) -> NDArray[np.float32]:
    """
    Compute chromagram (pitch class profile).

    Parameters
    ----------
    y
        Audio time series.
    sr
        Sample rate.
    n_chroma
        Number of chroma bins.
    hop_length
        Hop length.

    Returns
    -------
    NDArray[np.float32]
        Chromagram (n_chroma, time).
    """
    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        n_chroma=n_chroma,
        hop_length=hop_length,
    )

    return chroma.astype(np.float32)


def get_frequency_bands_energy(
    y: NDArray[np.float32],
    sr: int,
    *,
    bands: dict[str, tuple[float, float]] | None = None,
    n_fft: int = 2048,
) -> dict[str, float]:
    """
    Compute energy in frequency bands.

    Parameters
    ----------
    y
        Audio time series.
    sr
        Sample rate.
    bands
        Dictionary of band_name -> (low_hz, high_hz).
        Defaults to sub-bass, bass, low-mid, mid, upper-mid, presence, brilliance.
    n_fft
        FFT window size.

    Returns
    -------
    dict[str, float]
        Relative energy in each band (0-1).
    """
    if bands is None:
        bands = {
            "sub_bass": (20, 60),
            "bass": (60, 250),
            "low_mid": (250, 500),
            "mid": (500, 2000),
            "upper_mid": (2000, 4000),
            "presence": (4000, 6000),
            "brilliance": (6000, 20000),
        }

    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Compute power spectrum
    S = np.abs(librosa.stft(y, n_fft=n_fft)) ** 2

    # Get frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Total energy
    total_energy = np.sum(S)

    if total_energy == 0:
        return {name: 0.0 for name in bands}

    result = {}
    for name, (low, high) in bands.items():
        # Find bins in this frequency range
        mask = (freqs >= low) & (freqs < high)
        band_energy = np.sum(S[mask, :])
        result[name] = float(band_energy / total_energy)

    return result
