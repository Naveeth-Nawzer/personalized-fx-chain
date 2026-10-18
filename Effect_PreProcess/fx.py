"""Phase 1 effects (Pedalboard), parameter sampling, chains and labels.

Two chain modes (``generation.chain_mode``):

* ``subset``  — the 7 non-empty subsets, always processed in the fixed order
  EQ -> Compressor -> Reverb. Label = presence ``[EQ, Compressor, Reverb]``.
* ``ordered`` — every order of every non-empty subset (15 chains), processed
  in the chain's own order. Label = position ``[pos_EQ, pos_Comp, pos_Rev]``
  (0 = absent, 1..3 = position in the chain).

Parameters are sampled from ``config.yaml`` with a NumPy RNG seeded per
sample, so the same seed always gives the same parameters and audio.
"""

from __future__ import annotations

import math
from itertools import combinations, permutations
from typing import Any, Mapping, Sequence

import numpy as np
import pedalboard

from audio_utils import residual_db

LABEL_NAMES: tuple[str, ...] = ("EQ", "Compressor", "Reverb")
LABEL_COLUMNS = {"EQ": "eq_label", "Compressor": "compressor_label", "Reverb": "reverb_label"}
POSITION_COLUMNS = {"EQ": "eq_position", "Compressor": "compressor_position", "Reverb": "reverb_position"}
PARAM_COLUMNS = {"EQ": "eq_params", "Compressor": "compressor_params", "Reverb": "reverb_params"}
CHAIN_MODES = ("subset", "ordered")
VALUE_DECIMALS = 4  # sampled values are rounded so metadata == the value given to Pedalboard


# --------------------------------------------------------------- chains / labels

def effect_combinations(order: Sequence[str]) -> list[tuple[str, ...]]:
    """All non-empty subsets in processing order:
    EQ, Compressor, Reverb, EQ+Compressor, EQ+Reverb, Compressor+Reverb, EQ+Compressor+Reverb."""
    return [combo for r in range(1, len(order) + 1) for combo in combinations(order, r)]


def effect_chains(order: Sequence[str], mode: str = "subset") -> list[tuple[str, ...]]:
    """Chains to render. In ``ordered`` mode the list index is the chain's class id (0..14) and the
    canonical order of each subset always comes first among its permutations."""
    if mode == "subset":
        return effect_combinations(order)
    if mode == "ordered":
        return [chain for subset in effect_combinations(order) for chain in permutations(subset)]
    raise ValueError(f"chain_mode must be one of {CHAIN_MODES} (got {mode!r})")


def chain_name(chain: Sequence[str], mode: str) -> str:
    """``EQ+Reverb`` for a subset, ``Reverb->EQ`` for an ordered chain."""
    return ("+" if mode == "subset" else "->").join(chain)


def label_vector(chain: Sequence[str]) -> list[int]:
    """Presence: 1 if the effect is in the chain."""
    return [int(name in chain) for name in LABEL_NAMES]


def position_vector(chain: Sequence[str]) -> list[int]:
    """Position of each effect in the chain (1-based), 0 if absent. Reverb->EQ -> [2, 0, 1]."""
    return [list(chain).index(name) + 1 if name in chain else 0 for name in LABEL_NAMES]


def slot_vector(chain: Sequence[str]) -> list[int]:
    """Effect at each chain slot (1=EQ, 2=Compressor, 3=Reverb), 0 = empty. Reverb->EQ -> [3, 1, 0]."""
    return [LABEL_NAMES.index(e) + 1 for e in chain] + [0] * (len(LABEL_NAMES) - len(chain))


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

def render_chain(dry: np.ndarray, sample_rate: int, chain: Sequence[str],
                 sampler: ParameterSampler, rng: np.random.Generator,
                 min_change_db: float, max_retries: int,
                 fixed_params: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, Any]:
    """Apply the effects of ``chain`` to ``dry`` one after another, in chain order.

    Each effect is rendered step by step so its parameters can be re-sampled
    (random mode) if it changes the signal by less than ``min_change_db``.
    With ``fixed_params`` (effect -> parameters) nothing is sampled: those exact
    values are used, e.g. to render another order of an already rendered effect set.
    The output equals running the chained plugins with the returned parameters.
    """
    x = dry
    parameters: dict[str, dict[str, float]] = {}
    changes: dict[str, float] = {}
    low: list[str] = []
    for effect in chain:
        for _ in range(max_retries + 1):
            params = dict(fixed_params[effect]) if fixed_params else sampler.sample(effect, rng)
            y = apply_effect(effect, x, sample_rate, params)
            change = residual_db(x, y)
            if change >= min_change_db or sampler.mode == "fixed" or fixed_params:
                break
        if change < min_change_db:
            low.append(effect)
        parameters[effect] = params
        changes[effect] = round(change, 2)
        x = y
    return {"wet": x, "parameters": parameters, "effect_change_db": changes, "low_effect_change": low}
