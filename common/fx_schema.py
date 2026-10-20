"""
common/fx_schema.py
===================
Canonical FX-chain schema loader and validator.

Responsibilities
----------------
- Load and cache the effects_config.yaml configuration.
- Parse the canonical FX-chain dict (see Phase 3 contract).
- Reject chains that violate any structural rule.
- Provide a clean ``FXChainSpec`` dataclass consumed by DAPN.

Schema contract (input dict)
----------------------------
{
  "chain": [
    {
      "effect": "<EffectName>",
      "parameters": {
        "<param_name>": <float in [0, 1]>,
        ...
      }
    },
    ...
  ]
}

Rules enforced here
-------------------
1. Unknown effects → rejected.
2. Unknown parameters → rejected.
3. Parameters outside [0, 1] → rejected.
4. Required parameters missing → rejected.
5. Chain length outside [min, max] → rejected.
6. Repeated effects (when disabled) → rejected.
7. STOP effect inside a DAPN chain → rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.audio_types import EffectParameterSpec
from common.config_loader import load_effects_config, load_system_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectStep:
    """One step in a validated FX chain.

    Attributes
    ----------
    effect : str
        Canonical effect name (e.g. ``"EQ"``).
    parameters : dict[str, float]
        Normalized parameter values, all in [0, 1].
    """

    effect: str
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FXChainSpec:
    """A fully validated, immutable FX chain specification.

    This is the object consumed by ``FXChain`` and ``DAPN``.
    """

    steps: tuple[EffectStep, ...]

    def __len__(self) -> int:
        return len(self.steps)

    def effect_names(self) -> list[str]:
        """Ordered list of effect names in the chain."""
        return [s.effect for s in self.steps]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class FXChainValidator:
    """Validates raw FX-chain dicts against the effects config and system config.

    Parameters
    ----------
    effects_config_path : Path, optional
        Override path to ``effects_config.yaml``.
    system_config_path : Path, optional
        Override path to ``system_config.yaml``.
    """

    def __init__(
        self,
        effects_config_path: Path | None = None,
        system_config_path: Path | None = None,
    ) -> None:
        self._effects_cfg = load_effects_config(effects_config_path)
        self._system_cfg = load_system_config(system_config_path)

        chain_cfg = self._system_cfg["chain"]
        self._min_chain_length: int = int(chain_cfg["min_chain_length"])
        self._max_chain_length: int = int(chain_cfg["max_chain_length"])
        self._allow_repeated_effects: bool = bool(chain_cfg["allow_repeated_effects"])

        # Build parameter specs indexed by effect name
        self._param_specs: dict[str, dict[str, EffectParameterSpec]] = (
            self._build_param_specs()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, chain_dict: dict[str, Any]) -> FXChainSpec:
        """Validate *chain_dict* and return a ``FXChainSpec``.

        Parameters
        ----------
        chain_dict : dict
            Raw FX-chain dict, e.g.::

                {"chain": [{"effect": "EQ", "parameters": {"gain": 0.5, ...}}, ...]}

        Returns
        -------
        FXChainSpec
            A validated, immutable representation ready for ``DAPN``.

        Raises
        ------
        ValueError
            If any validation rule is violated.
        """
        if not isinstance(chain_dict, dict) or "chain" not in chain_dict:
            raise ValueError(
                "FX chain dict must have a top-level 'chain' key. "
                f"Got keys: {list(chain_dict.keys()) if isinstance(chain_dict, dict) else type(chain_dict)}"
            )

        raw_chain: list[Any] = chain_dict["chain"]
        if not isinstance(raw_chain, list):
            raise ValueError(
                f"'chain' must be a list of effect steps, got {type(raw_chain).__name__}"
            )

        # --- Chain length ---------------------------------------------------
        n = len(raw_chain)
        if n < self._min_chain_length:
            raise ValueError(
                f"Chain too short: got {n} step(s), minimum is {self._min_chain_length}."
            )
        if n > self._max_chain_length:
            raise ValueError(
                f"Chain too long: got {n} step(s), maximum is {self._max_chain_length}."
            )

        steps: list[EffectStep] = []
        seen_effects: set[str] = set()

        for idx, raw_step in enumerate(raw_chain):
            step = self._validate_step(raw_step, idx, seen_effects)
            if not self._allow_repeated_effects:
                seen_effects.add(step.effect)
            steps.append(step)

        spec = FXChainSpec(steps=tuple(steps))
        logger.debug("FX chain validated: %s", spec.effect_names())
        return spec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_step(
        self,
        raw_step: Any,
        idx: int,
        seen_effects: set[str],
    ) -> EffectStep:
        """Validate one step dict and return an ``EffectStep``."""
        if not isinstance(raw_step, dict):
            raise ValueError(
                f"Step {idx}: expected a dict, got {type(raw_step).__name__}"
            )

        # --- Effect name ---------------------------------------------------
        effect_name: str = raw_step.get("effect", "")
        if not effect_name:
            raise ValueError(f"Step {idx}: missing 'effect' key.")

        known_effects = set(self._param_specs.keys())
        if effect_name not in known_effects:
            raise ValueError(
                f"Step {idx}: unknown effect '{effect_name}'. "
                f"Supported effects: {sorted(known_effects)}."
            )

        if effect_name == "STOP":
            raise ValueError(
                f"Step {idx}: 'STOP' is a reserved GRRL-Agent action and cannot "
                "appear inside a DAPN audio-processing chain."
            )

        if not self._allow_repeated_effects and effect_name in seen_effects:
            raise ValueError(
                f"Step {idx}: effect '{effect_name}' appears more than once and "
                "'allow_repeated_effects' is disabled in system_config.yaml."
            )

        # --- Parameters ----------------------------------------------------
        raw_params: Any = raw_step.get("parameters", {})
        if not isinstance(raw_params, dict):
            raise ValueError(
                f"Step {idx} ({effect_name}): 'parameters' must be a dict, "
                f"got {type(raw_params).__name__}."
            )

        param_specs = self._param_specs[effect_name]
        validated_params: dict[str, float] = {}

        # Check for unknown parameters
        for param_name in raw_params:
            if param_name not in param_specs:
                raise ValueError(
                    f"Step {idx} ({effect_name}): unknown parameter '{param_name}'. "
                    f"Valid parameters: {sorted(param_specs.keys())}."
                )

        # Check required parameters and value range
        for param_name, spec in param_specs.items():
            if spec.required and param_name not in raw_params:
                raise ValueError(
                    f"Step {idx} ({effect_name}): required parameter '{param_name}' is missing."
                )

            if param_name in raw_params:
                value = raw_params[param_name]
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Step {idx} ({effect_name}): parameter '{param_name}' "
                        f"must be a float, got {type(raw_params[param_name]).__name__}."
                    )

                norm_min = spec.normalized_min
                norm_max = spec.normalized_max
                if not (norm_min - 1e-7 <= value <= norm_max + 1e-7):
                    raise ValueError(
                        f"Step {idx} ({effect_name}): parameter '{param_name}' = {value} "
                        f"is outside normalized range [{norm_min}, {norm_max}]."
                    )
                validated_params[param_name] = float(value)

        return EffectStep(effect=effect_name, parameters=validated_params)

    def _build_param_specs(self) -> dict[str, dict[str, EffectParameterSpec]]:
        """Parse effects_config.yaml into ``EffectParameterSpec`` objects."""
        effects_raw: dict[str, Any] = self._effects_cfg.get("effects", {})
        result: dict[str, dict[str, EffectParameterSpec]] = {}

        for effect_name, effect_data in effects_raw.items():
            params_raw: dict[str, Any] = effect_data.get("parameters", {})
            param_specs: dict[str, EffectParameterSpec] = {}

            for param_name, pd in params_raw.items():
                norm_range = pd.get("normalized_range", [0.0, 1.0])
                phys_range = pd.get("physical_range", [0.0, 1.0])
                param_specs[param_name] = EffectParameterSpec(
                    name=param_name,
                    normalized_min=float(norm_range[0]),
                    normalized_max=float(norm_range[1]),
                    physical_min=float(phys_range[0]),
                    physical_max=float(phys_range[1]),
                    unit=str(pd.get("unit", "dimensionless")),
                    mapping_type=pd.get("mapping_type", "linear"),
                    required=bool(pd.get("required", True)),
                )

            result[effect_name] = param_specs

        return result

    def get_param_specs(self, effect_name: str) -> dict[str, EffectParameterSpec]:
        """Return parameter specs for a given effect name."""
        if effect_name not in self._param_specs:
            raise KeyError(f"Unknown effect: '{effect_name}'")
        return self._param_specs[effect_name]
