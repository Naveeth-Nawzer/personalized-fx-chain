"""Phase 1 effects (Pedalboard), parameter sampling, combinations and labels.

* Label = binary ``[EQ, Compressor, Reverb]`` vector (effect presence only).
* Chains are always processed in the fixed order EQ -> Compressor -> Reverb.
* Parameters are sampled from ``config.yaml`` with a NumPy RNG seeded per
  sample, so the same seed always gives the same parameters and audio.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pedalboard

from audio_utils import residual_db

LABEL_NAMES: tuple[str, ...] = ("EQ", "Compressor", "Reverb")
LABEL_COLUMNS = {"EQ": "eq_label", "Compressor": "compressor_label", "Reverb": "reverb_label"}
PARAM_COLUMNS = {"EQ": "eq_params", "Compressor": "compressor_params", "Reverb": "reverb_params"}
VALUE_DECIMALS = 4  # sampled values are rounded so metadata == the value given to Pedalboard


# --------------------------------------------------------------- combinations / labels

def effect_combinations(order: Sequence[str]) -> list[tuple[str, ...]]:
    """All non-empty subsets in processing order:
    EQ, Compressor, Reverb, EQ+Compressor, EQ+Reverb, Compressor+Reverb, EQ+Compressor+Reverb."""
    return [combo for r in range(1, len(order) + 1) for combo in combinations(order, r)]


def label_vector(chain: Sequence[str]) -> list[int]:
    return [int(name in chain) for name in LABEL_NAMES]


# --------------------------------------------------------------- Pedalboard plugins

def build_plugins(effect: str, p: Mapping[str, float]) -> list[Any]:
    """Map our parameter names onto Pedalboard plugin arguments."""
    if effect == "EQ":
        return [pedalboard.PeakFilter(cutoff_frequency_hz=p["frequency"], gain_db=p["gain"], q=p["q"])]
    if effect == "Compressor":
        return [pedalboard.Compressor(threshold_db=p["threshold"], ratio=p["ratio"],
                                      attack_ms=p["attack"], release_ms=p["release"]),
                pedalboard.Gain(gain_db=p["makeup_gain"])]
    if effect == "Reverb":
        return [pedalboard.Reverb(room_size=p["room_size"], damping=p["damping"], wet_level=p["wet_mix"],
                                  dry_level=p["dry_mix"], width=p["width"], freeze_mode=0.0)]
    raise ValueError(f"Unknown effect: {effect}")


def apply_effect(effect: str, audio: np.ndarray, sample_rate: int, params: Mapping[str, float]) -> np.ndarray:
    board = pedalboard.Pedalboard(build_plugins(effect, params))
    out = np.asarray(board(np.ascontiguousarray(audio, dtype=np.float32), float(sample_rate), reset=True),
                     dtype=np.float32)
    if out.shape != audio.shape or not np.all(np.isfinite(out)):
        raise RuntimeError(f"{effect}: invalid output (shape {out.shape}, input {audio.shape})")
    return out


# --------------------------------------------------------------- parameters

class ParameterSampler:
    """``fixed`` mode returns each spec's ``fixed`` value; ``random`` samples in [min, max]."""

    def __init__(self, specs: Mapping[str, Mapping[str, Mapping[str, Any]]], mode: str) -> None:
        if mode not in ("fixed", "random"):
            raise ValueError(f"parameter_mode must be 'fixed' or 'random' (got {mode!r})")
        self.specs, self.mode = specs, mode

    def sample(self, effect: str, rng: np.random.Generator) -> dict[str, float]:
        out = {}
        for name, s in self.specs[effect].items():
            if self.mode == "fixed":
                value = float(s["fixed"])
            else:
                lo, hi = float(s["min"]), float(s["max"])
                if lo == hi:
                    value = lo
                elif s.get("scale") == "log":
                    value = math.exp(rng.uniform(math.log(lo), math.log(hi)))
                else:
                    value = rng.uniform(lo, hi)
                if s.get("random_sign") and rng.random() < 0.5:
                    value = -value
            out[name] = round(float(value), VALUE_DECIMALS)
        return out

    def normalize(self, effect: str, params: Mapping[str, float]) -> dict[str, float]:
        """Physical values -> [0, 1] (log-aware), for later parameter-prediction phases."""
        out = {}
        for name, value in params.items():
            s = self.specs[effect][name]
            if "norm_range" in s:
                lo, hi = map(float, s["norm_range"])
            elif s.get("random_sign"):
                lo, hi = -float(s["max"]), float(s["max"])
            else:
                lo, hi = float(s["min"]), float(s["max"])
            if hi == lo:
                norm = 0.0
            elif s.get("scale") == "log" and lo > 0 and value > 0:
                norm = (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
            else:
                norm = (value - lo) / (hi - lo)
            out[name] = round(min(max(norm, 0.0), 1.0), 6)
        return out


# --------------------------------------------------------------- rendering

def render_chain(dry: np.ndarray, sample_rate: int, chain: Sequence[str], order: Sequence[str],
                 sampler: ParameterSampler, rng: np.random.Generator,
                 min_change_db: float, max_retries: int) -> dict[str, Any]:
    """Apply ``chain`` to ``dry`` sequentially in the fixed ``order``.

    Each effect is rendered step by step so its parameters can be re-sampled
    (random mode) if it changes the signal by less than ``min_change_db``.
    The output equals running the chained plugins with the returned parameters.
    """
    x = dry
    parameters: dict[str, dict[str, float]] = {}
    changes: dict[str, float] = {}
    low: list[str] = []
    for effect in [e for e in order if e in chain]:
        for _ in range(max_retries + 1):
            params = sampler.sample(effect, rng)
            y = apply_effect(effect, x, sample_rate, params)
            change = residual_db(x, y)
            if change >= min_change_db or sampler.mode == "fixed":
                break
        if change < min_change_db:
            low.append(effect)
        parameters[effect] = params
        changes[effect] = round(change, 2)
        x = y
    return {"wet": x, "parameters": parameters, "effect_change_db": changes, "low_effect_change": low}
