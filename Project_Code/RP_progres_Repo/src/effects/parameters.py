"""Effect-parameter generation (fixed or random within safe ranges).

Parameters are not Phase 1 prediction targets, but every value used is stored
as ground truth for the later parameter-prediction phase. Alongside the
physical values, a normalised [0, 1] representation is produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

#: Decimal places kept for sampled values, so the value written to metadata is
#: exactly the value given to the DSP code.
VALUE_DECIMALS = 4


@dataclass(frozen=True)
class ParameterSpec:
    """Sampling/normalisation spec for one effect parameter."""

    name: str
    fixed: float
    min: float
    max: float
    scale: str = "linear"
    random_sign: bool = False
    unit: str = ""
    norm_range: tuple[float, float] | None = None

    @classmethod
    def from_config(cls, name: str, spec: Mapping[str, Any]) -> "ParameterSpec":
        norm = spec.get("norm_range")
        return cls(
            name=name,
            fixed=float(spec["fixed"]),
            min=float(spec["min"]),
            max=float(spec["max"]),
            scale=str(spec.get("scale", "linear")),
            random_sign=bool(spec.get("random_sign", False)),
            unit=str(spec.get("unit", "")),
            norm_range=(float(norm[0]), float(norm[1])) if norm is not None else None,
        )

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a value: uniform (linear) or log-uniform (log) in [min, max], optionally with a random sign."""
        if self.min == self.max:
            value = self.min
        elif self.scale == "log":
            value = math.exp(rng.uniform(math.log(self.min), math.log(self.max)))
        else:
            value = rng.uniform(self.min, self.max)
        if self.random_sign and rng.random() < 0.5:
            value = -value
        return round(float(value), VALUE_DECIMALS)

    def normalization_range(self) -> tuple[float, float]:
        if self.norm_range is not None:
            return self.norm_range
        if self.random_sign:
            return (-self.max, self.max)
        return (self.min, self.max)

    def normalize(self, value: float) -> float:
        """Map a physical value to [0, 1] (log-aware when scale is log and the range is positive)."""
        lo, hi = self.normalization_range()
        if hi == lo:
            return 0.0
        if self.scale == "log" and lo > 0 and value > 0:
            norm = (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            norm = (value - lo) / (hi - lo)
        return round(float(min(max(norm, 0.0), 1.0)), 6)


class ParameterSampler:
    """Produces effect parameters in ``fixed`` or ``random`` mode.

    Args:
        specs: ``{effect_name: {param_name: ParameterSpec}}``.
        mode: ``"fixed"`` or ``"random"``.
    """

    def __init__(self, specs: Mapping[str, Mapping[str, ParameterSpec]], mode: str) -> None:
        if mode not in ("fixed", "random"):
            raise ValueError(f"parameter mode must be 'fixed' or 'random' (got {mode!r})")
        self.specs = {effect: dict(params) for effect, params in specs.items()}
        self.mode = mode

    @classmethod
    def from_config(cls, parameters_cfg: Mapping[str, Any], mode: str) -> "ParameterSampler":
        specs = {
            effect: {pname: ParameterSpec.from_config(pname, spec) for pname, spec in params.items()}
            for effect, params in parameters_cfg.items()
        }
        return cls(specs, mode)

    @property
    def is_random(self) -> bool:
        return self.mode == "random"

    def sample(self, effect: str, rng: np.random.Generator | None = None) -> dict[str, float]:
        """Parameters for one effect instance. ``rng`` is required in random mode."""
        specs = self._specs_for(effect)
        if self.mode == "fixed":
            return {name: round(spec.fixed, VALUE_DECIMALS) for name, spec in specs.items()}
        if rng is None:
            raise ValueError("random parameter mode needs an rng")
        return {name: spec.sample(rng) for name, spec in specs.items()}

    def normalize(self, effect: str, parameters: Mapping[str, float]) -> dict[str, float]:
        """Normalised [0, 1] version of physical parameters."""
        specs = self._specs_for(effect)
        return {name: specs[name].normalize(value) for name, value in parameters.items()}

    def _specs_for(self, effect: str) -> dict[str, ParameterSpec]:
        if effect not in self.specs:
            raise KeyError(f"No parameter specs configured for effect '{effect}'")
        return self.specs[effect]
