"""Dynamic range compressor — numpy backend.

Feed-forward, hard-knee, log-domain design with a branching attack/release
smoother (Giannoulis, Massberg & Reiss, "Digital Dynamic Range Compressor
Design — A Tutorial and Analysis", JAES 2012). Channels are linked: one gain
curve, computed from the per-sample peak across channels, is applied to all.
"""

from __future__ import annotations

import math

import numpy as np

from src.effects.base import BaseEffect, EffectError

_EPS = 1e-12


def gain_reduction_db(level_db: np.ndarray, threshold_db: float, ratio: float) -> np.ndarray:
    """Static hard-knee gain computer: dB of gain change (<= 0) for each input level."""
    over = np.maximum(level_db - threshold_db, 0.0)
    return -over * (1.0 - 1.0 / ratio)


def smooth_gain_db(target_db: np.ndarray, attack_coeff: float, release_coeff: float) -> np.ndarray:
    """Branching one-pole smoother: attack when gain reduction increases, release otherwise."""
    out = np.empty_like(target_db)
    state = 0.0
    a_att, a_rel = attack_coeff, release_coeff
    # Plain-float loop: the recursion is inherently sequential.
    for i, g in enumerate(target_db.tolist()):
        if g < state:
            state = a_att * state + (1.0 - a_att) * g
        else:
            state = a_rel * state + (1.0 - a_rel) * g
        out[i] = state
    return out


def time_constant_coeff(time_ms: float, sample_rate: int) -> float:
    """One-pole coefficient for a time constant given in milliseconds."""
    return math.exp(-1.0 / (max(time_ms, 1e-3) * 1e-3 * sample_rate))


class CompressorEffect(BaseEffect):
    """Feed-forward compressor with makeup gain.

    Parameters (physical units):
        threshold: dBFS level above which compression starts
        ratio: compression ratio (>= 1)
        attack: attack time in ms
        release: release time in ms
        makeup_gain: gain applied after compression, in dB
    """

    name = "Compressor"
    backend = "scipy"
    parameter_names = ("threshold", "ratio", "attack", "release", "makeup_gain")

    def _process(self, audio: np.ndarray, sample_rate: int, parameters: dict[str, float]) -> np.ndarray:
        if parameters["ratio"] < 1.0:
            raise EffectError(f"Compressor ratio must be >= 1 (got {parameters['ratio']})")
        x = audio.astype(np.float64)
        level_db = 20.0 * np.log10(np.maximum(np.max(np.abs(x), axis=0), _EPS))
        target = gain_reduction_db(level_db, parameters["threshold"], parameters["ratio"])
        smoothed = smooth_gain_db(
            target,
            time_constant_coeff(parameters["attack"], sample_rate),
            time_constant_coeff(parameters["release"], sample_rate),
        )
        gain = 10.0 ** ((smoothed + parameters["makeup_gain"]) / 20.0)
        return x * gain[np.newaxis, :]
