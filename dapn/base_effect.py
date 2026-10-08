"""
dapn/base_effect.py
===================
Abstract base class for all DAPN audio effects.

Interface contract
------------------
- Subclass :class:`BaseEffect` and implement :meth:`forward`.
- ``forward`` receives *normalized* parameter tensors (all in [0, 1]).
- ``forward`` must return an audio tensor of the **same shape** as the input.
- The computational graph must be preserved end-to-end (no detach, no NumPy).
- Each subclass creates its own :class:`ParameterMapper` instances via the
  shared :func:`build_mappers_for_effect` factory so mapping logic is
  never duplicated.

Audio tensor shape convention
------------------------------
    (B, C, T) — batch × channels × time-samples  [primary]
    (C, T)    — channels × time-samples           [accepted, wrapped internally]

Parameters passed to ``forward``
---------------------------------
    dict[str, torch.Tensor]
        Keys = parameter names as defined in effects_config.yaml.
        Values = normalized scalar tensors in [0, 1] with ``requires_grad=True``
                 when gradient-based optimisation is active.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path

import torch
import torch.nn as nn

from common.audio_types import EffectParameterSpec
from common.config_loader import load_effects_config
from common.parameter_mapper import ParameterMapper, build_mappers_for_effect

logger = logging.getLogger(__name__)


class BaseEffect(nn.Module, abc.ABC):
    """Abstract base class for a differentiable audio effect.

    Subclasses must implement :meth:`forward` and set the class-level
    attribute ``effect_name`` to the canonical name used in
    ``effects_config.yaml`` (e.g. ``"EQ"``).

    Parameters
    ----------
    effects_config_path : Path, optional
        Override path to ``effects_config.yaml`` for testing.
    """

    effect_name: str  # override in each concrete subclass

    def __init__(
        self,
        effects_config_path: Path | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(self, "effect_name") or not self.effect_name:
            raise TypeError(
                f"{type(self).__name__} must define a non-empty class attribute 'effect_name'."
            )

        cfg = load_effects_config(effects_config_path)
        effects_raw = cfg.get("effects", {})

        if self.effect_name not in effects_raw:
            raise ValueError(
                f"Effect '{self.effect_name}' is not defined in effects_config.yaml. "
                f"Available effects: {list(effects_raw.keys())}."
            )

        # Build parameter specs and mappers from config
        params_raw = effects_raw[self.effect_name].get("parameters", {})
        self._param_specs: dict[str, EffectParameterSpec] = {}
        for param_name, pd in params_raw.items():
            norm_range = pd.get("normalized_range", [0.0, 1.0])
            phys_range = pd.get("physical_range", [0.0, 1.0])
            self._param_specs[param_name] = EffectParameterSpec(
                name=param_name,
                normalized_min=float(norm_range[0]),
                normalized_max=float(norm_range[1]),
                physical_min=float(phys_range[0]),
                physical_max=float(phys_range[1]),
                unit=str(pd.get("unit", "dimensionless")),
                mapping_type=pd.get("mapping_type", "linear"),
                required=bool(pd.get("required", True)),
            )

        self._mappers: dict[str, ParameterMapper] = build_mappers_for_effect(
            self._param_specs
        )
        logger.debug("Initialized %s with parameters: %s", type(self).__name__, list(self._mappers))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def forward(
        self,
        audio: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Process audio with the given normalized parameters.

        Parameters
        ----------
        audio : torch.Tensor
            Input audio, shape ``(B, C, T)`` or ``(C, T)``, dtype float32.
        parameters : dict[str, torch.Tensor]
            Normalized parameter tensors in [0, 1].  Values may have
            ``requires_grad=True`` — do **not** detach them.

        Returns
        -------
        torch.Tensor
            Processed audio tensor with the **same shape** as *audio*.
        """

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def map_to_physical(self, name: str, normalized: torch.Tensor) -> torch.Tensor:
        """Convert a named normalized parameter to its physical DSP value.

        Parameters
        ----------
        name : str
            Parameter name (must match effects_config.yaml).
        normalized : torch.Tensor
            Normalized value in [0, 1].

        Returns
        -------
        torch.Tensor
            Physical value — gradient preserved.
        """
        if name not in self._mappers:
            raise KeyError(
                f"Effect '{self.effect_name}': unknown parameter '{name}'. "
                f"Known parameters: {list(self._mappers.keys())}."
            )
        return self._mappers[name].to_physical(normalized)

    def ensure_3d(self, audio: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Add a batch dimension if audio is 2-D (C, T).

        Returns
        -------
        audio_3d : torch.Tensor
            Shape ``(B, C, T)``.
        was_2d : bool
            True when the input was 2-D — use to strip the batch dim on return.
        """
        if audio.ndim == 2:
            return audio.unsqueeze(0), True
        if audio.ndim == 3:
            return audio, False
        raise ValueError(
            f"Audio must be 2-D (C, T) or 3-D (B, C, T), got shape {tuple(audio.shape)}."
        )

    def param_specs(self) -> dict[str, EffectParameterSpec]:
        """Return the parameter specifications for this effect."""
        return dict(self._param_specs)

    def __repr__(self) -> str:
        params = list(self._mappers.keys())
        return f"{type(self).__name__}(effect='{self.effect_name}', params={params})"
