"""Biquad filters for controls the pedalboard filters do not expose."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

from soundlab.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from numpy.typing import NDArray


def apply_rbj_filter(
    samples: NDArray[np.floating],
    sample_rate: int,
    cutoff_hz: float,
    q: float,
    kind: str,
) -> NDArray[np.float32]:
    """Apply a causal RBJ cookbook high-pass or low-pass biquad.

    ``kind`` is ``"highpass"`` or ``"lowpass"``. The cutoff must sit below
    the Nyquist frequency for ``sample_rate``.
    """
    if kind not in {"highpass", "lowpass"}:
        raise ConfigurationError(f"Unsupported biquad kind: {kind}")
    if q <= 0:
        raise ConfigurationError("Filter Q must be greater than zero.")
    nyquist = sample_rate / 2
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        raise ConfigurationError(
            f"{kind} cutoff {cutoff_hz} Hz is outside the range for sample rate {sample_rate}."
        )

    b_coeff, a_coeff = _rbj_coefficients(kind, cutoff_hz, q, sample_rate)
    return _filter_channels(np.asarray(samples), b_coeff, a_coeff)


def _rbj_coefficients(
    kind: str,
    cutoff_hz: float,
    q: float,
    sample_rate: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    omega = 2.0 * math.pi * cutoff_hz / sample_rate
    cosine = math.cos(omega)
    alpha = math.sin(omega) / (2.0 * q)
    if kind == "highpass":
        b0 = (1.0 + cosine) / 2.0
        b1 = -(1.0 + cosine)
        b2 = (1.0 + cosine) / 2.0
    else:
        b0 = (1.0 - cosine) / 2.0
        b1 = 1.0 - cosine
        b2 = (1.0 - cosine) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cosine
    a2 = 1.0 - alpha
    b_coeff = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a_coeff = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b_coeff, a_coeff


def _filter_channels(
    samples: NDArray[np.floating],
    b_coeff: NDArray[np.float64],
    a_coeff: NDArray[np.float64],
) -> NDArray[np.float32]:
    if samples.ndim == 1:
        filtered = signal.lfilter(b_coeff, a_coeff, samples)
        return filtered.astype(np.float32, copy=False)
    if samples.shape[0] <= samples.shape[-1]:
        filtered = np.stack(
            [signal.lfilter(b_coeff, a_coeff, channel) for channel in samples],
            axis=0,
        )
        return filtered.astype(np.float32, copy=False)
    filtered = np.stack(
        [signal.lfilter(b_coeff, a_coeff, samples[:, index]) for index in range(samples.shape[1])],
        axis=1,
    )
    return filtered.astype(np.float32, copy=False)
