"""
dapn/dapn.py
============
Top-level Differentiable Audio Processing Network (DAPN).

:class:`DAPN` is the single public entry point for the entire
audio-processing pipeline.  It:

1. Validates the canonical FX-chain schema.
2. Constructs an :class:`FXChain`.
3. Validates the input audio tensor.
4. Delegates processing to the chain.
5. Returns the processed audio tensor with the full computational graph intact.

GRRL integration point
-----------------------
When the GRRL-Agent is implemented, it will:
  1. Choose an effect name and normalized parameters.
  2. Accumulate steps into a canonical chain dict.
  3. Pass the dict to ``DAPN.process(audio, chain_dict)``.
  4. Receive processed audio and compute a reward.

The DAPN is intentionally stateless across calls — no audio is
stored between forward passes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from common.config_loader import get_sample_rate
from common.fx_schema import FXChainSpec, FXChainValidator
from common.validation import validate_audio_tensor
from dapn.fx_chain import FXChain

logger = logging.getLogger(__name__)


class DAPN(nn.Module):
    """Differentiable Audio Processing Network.

    Parameters
    ----------
    effects_config_path : Path, optional
        Override path to ``effects_config.yaml``.
    system_config_path : Path, optional
        Override path to ``system_config.yaml``.

    Examples
    --------
    >>> dapn = DAPN()
    >>> chain_dict = {
    ...     "chain": [
    ...         {"effect": "EQ",         "parameters": {"frequency": 0.5, "gain": 0.7, "q": 0.4}},
    ...         {"effect": "Compressor", "parameters": {"threshold": 0.6, "ratio": 0.3,
    ...                                                  "attack": 0.2, "release": 0.4,
    ...                                                  "makeup_gain": 0.3}},
    ...     ]
    ... }
    >>> out = dapn.process(audio_tensor, chain_dict)
    """

    def __init__(
        self,
        effects_config_path: Path | None = None,
        system_config_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._validator = FXChainValidator(
            effects_config_path=effects_config_path,
            system_config_path=system_config_path,
        )
        self._sample_rate: int = get_sample_rate()
        logger.info("DAPN initialised (sr=%d Hz)", self._sample_rate)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def process(
        self,
        audio: torch.Tensor,
        chain_dict: dict[str, Any],
        param_override: dict[int, dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Validate, build, and run the FX chain on *audio*.

        Parameters
        ----------
        audio : torch.Tensor
            Input audio, shape ``(B, C, T)`` or ``(C, T)``, float32.
        chain_dict : dict
            Canonical FX-chain schema (see Phase 3 contract).
        param_override : dict[int, dict[str, torch.Tensor]], optional
            Learnable parameter tensors keyed by step index, forwarded
            directly to :class:`FXChain`.  Use for gradient-based optimisation.

        Returns
        -------
        torch.Tensor
            Processed audio, same shape as *audio*.

        Raises
        ------
        common.validation.AudioValidationError
            If the audio tensor fails validation.
        ValueError
            If the FX chain schema is invalid.
        """
        # ---- Validate audio -------------------------------------------------
        validate_audio_tensor(audio, sample_rate=self._sample_rate)

        # ---- Validate chain -------------------------------------------------
        chain_spec: FXChainSpec = self._validator.validate(chain_dict)
        logger.info("Processing chain: %s", chain_spec.effect_names())

        # ---- Build chain and run --------------------------------------------
        chain = FXChain(chain_spec)
        out = chain(audio, param_override=param_override)

        # ---- Validate output (non-graph, informational only) ----------------
        if not torch.isfinite(out).all():
            n_bad = (~torch.isfinite(out)).sum().item()
            logger.warning("Output contains %d non-finite values — check DSP parameters.", n_bad)

        return out

    def forward(
        self,
        audio: torch.Tensor,
        chain_dict: dict[str, Any],
        param_override: dict[int, dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Alias of :meth:`process` so DAPN can be used as an ``nn.Module``."""
        return self.process(audio, chain_dict, param_override=param_override)

    @property
    def sample_rate(self) -> int:
        """Configured audio sample rate."""
        return self._sample_rate

    @property
    def validator(self) -> FXChainValidator:
        """The FX chain validator instance."""
        return self._validator

    def __repr__(self) -> str:
        return f"DAPN(sample_rate={self._sample_rate} Hz)"
