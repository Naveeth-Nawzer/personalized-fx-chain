"""Algorithmic reverb — numpy/scipy backend.

The reverb is a convolution with a synthetic impulse response (IR) built from
exponentially decaying noise, a standard model of late reverberation:

* ``room_size`` maps linearly to the reverberation time RT60
  (``rt60_min`` .. ``rt60_max`` seconds).
* ``damping`` shortens the decay of the band above ``damping_crossover_hz``
  (0 = bright, 1 = high frequencies decay 5x faster).
* ``width`` decorrelates left/right IRs for stereo output (no effect on mono).
* ``wet_mix`` / ``dry_mix`` set the output mix ``dry_mix*x + wet_mix*(x * ir)``.

The IR is energy-normalised and uses a fixed noise seed, so the parameters
fully determine the output. The output is truncated to the input length so
dry and wet stay sample-aligned (the reverb tail past the clip end is cut).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.signal import butter, oaconvolve, sosfilt

from src.effects.base import BaseEffect

DEFAULT_OPTIONS: dict[str, float] = {
    "rt60_min": 0.3,
    "rt60_max": 3.0,
    "pre_delay_ms": 10.0,
    "damping_crossover_hz": 3000.0,
    "ir_seed": 1234,
}
_DECAY_60DB = np.log(1000.0)  # amplitude decays by 60 dB when exp(-k t) = 1e-3


def synthetic_ir(
    sample_rate: int,
    channels: int,
    room_size: float,
    damping: float,
    width: float,
    options: Mapping[str, Any],
) -> np.ndarray:
    """Build an energy-normalised ``(channels, length)`` impulse response."""
    rt60 = options["rt60_min"] + float(np.clip(room_size, 0.0, 1.0)) * (options["rt60_max"] - options["rt60_min"])
    rt60_hf = rt60 * (1.0 - 0.8 * float(np.clip(damping, 0.0, 1.0)))
    length = max(int(round(rt60 * sample_rate)), 1)
    pre_delay = int(round(options["pre_delay_ms"] * 1e-3 * sample_rate))

    rng = np.random.default_rng(int(options["ir_seed"]))
    noise = rng.standard_normal((2, length))
    t = np.arange(length) / sample_rate

    crossover = min(float(options["damping_crossover_hz"]), 0.45 * sample_rate)
    sos_lo = butter(4, crossover, btype="lowpass", fs=sample_rate, output="sos")
    sos_hi = butter(4, crossover, btype="highpass", fs=sample_rate, output="sos")
    env_lo = np.exp(-_DECAY_60DB * t / rt60)
    env_hi = np.exp(-_DECAY_60DB * t / max(rt60_hf, 1e-3))
    tails = sosfilt(sos_lo, noise, axis=-1) * env_lo + sosfilt(sos_hi, noise, axis=-1) * env_hi

    if channels == 1:
        ir = tails[:1]
    else:
        w = float(np.clip(width, 0.0, 1.0))
        left = tails[0]
        right = w * tails[1] + (1.0 - w) * tails[0]
        ir = np.stack([left, right])
    ir = np.concatenate([np.zeros((ir.shape[0], pre_delay)), ir], axis=1)
    energy = np.sqrt(np.sum(ir**2, axis=1, keepdims=True))
    return ir / np.maximum(energy, 1e-12)


class ReverbEffect(BaseEffect):
    """Convolution reverb with a synthetic, parameter-controlled impulse response.

    Parameters:
        room_size: 0..1, controls RT60
        damping: 0..1, high-frequency decay shortening
        wet_mix: 0..1, level of the reverberated signal
        dry_mix: 0..1, level of the direct signal
        width: 0..1, stereo decorrelation (ignored for mono)
    """

    name = "Reverb"
    backend = "scipy"
    parameter_names = ("room_size", "damping", "wet_mix", "dry_mix", "width")

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self.options = {**DEFAULT_OPTIONS, **dict(options or {})}

    def _process(self, audio: np.ndarray, sample_rate: int, parameters: dict[str, float]) -> np.ndarray:
        channels, n = audio.shape
        ir = synthetic_ir(
            sample_rate, channels, parameters["room_size"], parameters["damping"], parameters["width"], self.options
        )
        x = audio.astype(np.float64)
        wet = oaconvolve(x, ir, mode="full", axes=-1)[:, :n]
        return parameters["dry_mix"] * x + parameters["wet_mix"] * wet
