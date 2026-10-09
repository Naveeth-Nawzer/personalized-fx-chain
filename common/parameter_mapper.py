"""
common/parameter_mapper.py
==========================
Reusable, differentiable parameter mapper.

Converts normalized [0, 1] values (ML-facing) to physical DSP values
and back.  This module is the *single authoritative implementation* of
the mapping; individual effects never duplicate this logic.

Design principles
-----------------
- 100% PyTorch-native in the differentiable path.
- No NumPy inside ``to_physical`` or ``to_normalized``.
- No in-place ops that would break autograd.
- Supports scalar tensors and batched tensors.
- Clamping policy: soft-clamp using ``torch.clamp`` before mapping;
  this is differentiable almost everywhere (gradients = 0 only outside
  the valid range, which is the desired behaviour for learned params).
- ``mapping_type`` field is present so future ``"log"`` mapping can be
  added without changing callers.

Physical mapping (linear, MVP)
--------------------------------
  physical = physical_min + normalized * (physical_max - physical_min)

Inverse:
  normalized = (physical - physical_min) / (physical_max - physical_min)
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

from common.audio_types import EffectParameterSpec

logger = logging.getLogger(__name__)

MappingType = Literal["linear", "log"]


class ParameterMapper:
    """Maps normalized [0, 1] scalars / tensors to physical DSP values.

    Parameters
    ----------
    spec : EffectParameterSpec
        The parameter specification loaded from effects_config.yaml.

    Examples
    --------
    >>> spec = EffectParameterSpec(
    ...     name="gain",
    ...     normalized_min=0.0, normalized_max=1.0,
    ...     physical_min=-12.0, physical_max=12.0,
    ...     unit="dB", mapping_type="linear", required=True,
    ... )
    >>> mapper = ParameterMapper(spec)
    >>> t = torch.tensor(0.5, requires_grad=True)
    >>> physical = mapper.to_physical(t)   # → tensor(0.0, grad_fn=...)
    >>> mapper.to_normalized(physical)     # → tensor(0.5, grad_fn=...)
    """

    def __init__(self, spec: EffectParameterSpec) -> None:
        if spec.mapping_type not in ("linear", "log"):
            raise ValueError(
                f"Unsupported mapping_type '{spec.mapping_type}' for parameter '{spec.name}'. "
                "MVP supports: 'linear'."
            )
        self._spec = spec
        self._range = spec.physical_max - spec.physical_min
        if self._range == 0.0:
            raise ValueError(
                f"Parameter '{spec.name}' has equal physical_min and physical_max "
                f"({spec.physical_min}). The range must be non-zero."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_physical(self, normalized: torch.Tensor) -> torch.Tensor:
        """Convert a normalized tensor in [0, 1] to a physical DSP value.

        Parameters
        ----------
        normalized : torch.Tensor
            Normalized value(s), expected in [0, 1].  Values outside
            this range are soft-clamped before mapping.

        Returns
        -------
        torch.Tensor
            Physical value in [physical_min, physical_max].
            Preserves ``requires_grad``.
        """
        # Soft clamp: differentiable, gradient = 0 only outside [0,1]
        clamped = torch.clamp(normalized, 0.0, 1.0)
        physical = self._spec.physical_min + clamped * self._range
        return physical

    def to_normalized(self, physical: torch.Tensor) -> torch.Tensor:
        """Convert a physical DSP value back to [0, 1] normalized form.

        Parameters
        ----------
        physical : torch.Tensor
            Physical value, expected in [physical_min, physical_max].

        Returns
        -------
        torch.Tensor
            Normalized value in [0, 1].  Preserves ``requires_grad``.
        """
        normalized = (physical - self._spec.physical_min) / self._range
        clamped = torch.clamp(normalized, 0.0, 1.0)
        return clamped

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def spec(self) -> EffectParameterSpec:
        """The underlying parameter specification."""
        return self._spec

    @property
    def physical_min(self) -> float:
        return self._spec.physical_min

    @property
    def physical_max(self) -> float:
        return self._spec.physical_max

    @property
    def mapping_type(self) -> MappingType:
        return self._spec.mapping_type  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"ParameterMapper(name={self._spec.name!r}, "
            f"physical=[{self._spec.physical_min}, {self._spec.physical_max}] "
            f"{self._spec.unit}, mapping={self._spec.mapping_type})"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_mappers_for_effect(
    param_specs: dict[str, EffectParameterSpec],
) -> dict[str, ParameterMapper]:
    """Create one :class:`ParameterMapper` per parameter in *param_specs*.

    Parameters
    ----------
    param_specs : dict[str, EffectParameterSpec]
        Specs as returned by ``FXChainValidator.get_param_specs(effect_name)``.

    Returns
    -------
    dict[str, ParameterMapper]
        Mappers keyed by parameter name.
    """
    return {name: ParameterMapper(spec) for name, spec in param_specs.items()}
