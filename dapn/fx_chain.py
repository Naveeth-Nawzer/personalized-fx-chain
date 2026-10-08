"""
dapn/fx_chain.py
================
Sequential FX chain processor.

:class:`FXChain` receives an audio tensor and a validated
:class:`~common.fx_schema.FXChainSpec`, instantiates the required effect
processors from the EFFECT_REGISTRY, and applies them sequentially.

Design decisions
----------------
- Effects are instantiated lazily on first call (or during construction
  if ``eager=True``) to avoid unnecessary object creation.
- The chain respects the exact ordering of ``FXChainSpec.steps``.
- Parameter tensors are created fresh for each forward pass from the
  float values stored in ``FXChainSpec``; they are detached from any
  previous graph.  When gradient optimisation is needed, callers should
  pass *learnable* parameter tensors directly via the
  ``param_override`` argument of :meth:`forward`.
- ``nn.ModuleDict`` stores effect instances so their (future) learnable
  parameters are properly registered with PyTorch.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from common.fx_schema import FXChainSpec
from dapn.effects import EFFECT_REGISTRY

logger = logging.getLogger(__name__)


class FXChain(nn.Module):
    """Sequential effect chain.

    Parameters
    ----------
    chain_spec : FXChainSpec
        A validated chain specification (output of :class:`FXChainValidator`).

    Examples
    --------
    >>> spec = validator.validate({"chain": [{"effect": "EQ", "parameters": {...}}]})
    >>> chain = FXChain(spec)
    >>> out = chain(audio_tensor)
    """

    def __init__(self, chain_spec: FXChainSpec) -> None:
        super().__init__()
        self._spec = chain_spec
        self._effects = nn.ModuleDict()

        # Pre-instantiate all effects in order.
        # Key = "<index>_<effect_name>" to handle repeated effects (if allowed).
        for idx, step in enumerate(chain_spec.steps):
            effect_name = step.effect
            if effect_name not in EFFECT_REGISTRY:
                raise ValueError(
                    f"Effect '{effect_name}' at position {idx} is not registered. "
                    f"Registered effects: {list(EFFECT_REGISTRY.keys())}.\n"
                    "If this effect is implemented, add it to dapn/effects/__init__.py."
                )
            key = f"{idx}_{effect_name}"
            self._effects[key] = EFFECT_REGISTRY[effect_name]()
            logger.debug("FXChain: registered effect [%s] %s", idx, effect_name)

        logger.info("FXChain initialized: %s", chain_spec.effect_names())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        audio: torch.Tensor,
        param_override: dict[int, dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Process *audio* through the full effect chain.

        Parameters
        ----------
        audio : torch.Tensor
            Input audio, shape ``(B, C, T)`` or ``(C, T)``.
        param_override : dict[int, dict[str, torch.Tensor]], optional
            Map from step index → {param_name → normalized tensor}.
            When provided, these tensors (which may have ``requires_grad=True``)
            replace the static float values stored in the chain spec.
            This is the primary entry point for gradient-based optimisation.

        Returns
        -------
        torch.Tensor
            Processed audio tensor, same shape as *audio*.
        """
        x = audio
        param_override = param_override or {}

        for idx, step in enumerate(self._spec.steps):
            key = f"{idx}_{step.effect}"
            effect = self._effects[key]

            # Resolve parameters: override tensors if provided, else use spec floats
            if idx in param_override:
                params = param_override[idx]
            else:
                params = self._spec_params_as_tensors(step.parameters, audio.device, audio.dtype)

            logger.debug("Applying effect [%d] %s", idx, step.effect)
            x = effect(x, params)

        return x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spec_params_as_tensors(
        params: dict[str, float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Convert float parameter dict from spec to tensors (no grad)."""
        return {
            name: torch.tensor(value, device=device, dtype=dtype)
            for name, value in params.items()
        }

    @property
    def spec(self) -> FXChainSpec:
        """The validated chain specification."""
        return self._spec

    def __repr__(self) -> str:
        names = self._spec.effect_names()
        return f"FXChain(steps={names})"
