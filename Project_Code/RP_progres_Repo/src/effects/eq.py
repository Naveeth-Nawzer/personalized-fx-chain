"""Parametric EQ (single peaking band) — numpy/scipy backend."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from src.effects.base import BaseEffect, EffectError


def peaking_biquad(frequency: float, gain_db: float, q: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Peaking-EQ biquad coefficients from the RBJ Audio EQ Cookbook.

    Returns:
        ``(b, a)`` normalised so that ``a[0] == 1``.
    """
    if not 0.0 < frequency < sample_rate / 2:
        raise EffectError(f"EQ frequency {frequency} Hz must be in (0, {sample_rate / 2}) Hz")
    if q <= 0:
        raise EffectError(f"EQ q must be > 0 (got {q})")
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * frequency / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b = np.array([1.0 + alpha * amp, -2.0 * cos_w0, 1.0 - alpha * amp])
    a = np.array([1.0 + alpha / amp, -2.0 * cos_w0, 1.0 - alpha / amp])
    return b / a[0], a / a[0]


class EQEffect(BaseEffect):
    """Single-band peaking equaliser.

    Parameters (physical units):
        frequency: centre frequency in Hz
        gain: boost/cut in dB
        q: bandwidth quality factor
    """

    name = "EQ"
    backend = "scipy"
    parameter_names = ("frequency", "gain", "q")

    def _process(self, audio: np.ndarray, sample_rate: int, parameters: dict[str, float]) -> np.ndarray:
        b, a = peaking_biquad(parameters["frequency"], parameters["gain"], parameters["q"], sample_rate)
        return lfilter(b, a, audio.astype(np.float64), axis=-1)
