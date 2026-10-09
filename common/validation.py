"""
common/validation.py
====================
Audio-tensor validation utilities used by DAPN at runtime.

These helpers check shape, dtype, duration, and finiteness of audio
tensors *before* they enter the differentiable DSP chain.  They are
intentionally kept outside the computational graph so they do not
pollute autograd.
"""

from __future__ import annotations

import logging

import torch

from common.config_loader import get_max_duration_seconds, get_sample_rate

logger = logging.getLogger(__name__)


class AudioValidationError(ValueError):
    """Raised when an audio tensor fails a validation check."""


def validate_audio_tensor(
    audio: torch.Tensor,
    sample_rate: int | None = None,
    *,
    check_finite: bool = True,
) -> None:
    """Validate an audio tensor against DAPN system requirements.

    Parameters
    ----------
    audio : torch.Tensor
        Tensor with shape ``(C, T)`` or ``(B, C, T)``.
    sample_rate : int, optional
        Sample rate of the audio.  If provided, duration is checked
        against ``max_duration_seconds`` from system_config.yaml.
    check_finite : bool
        If True, verify that all values are finite (no NaN or Inf).

    Raises
    ------
    AudioValidationError
        If any check fails.
    """
    if not isinstance(audio, torch.Tensor):
        raise AudioValidationError(
            f"Expected a torch.Tensor, got {type(audio).__name__}."
        )

    # --- Shape -----------------------------------------------------------
    if audio.ndim not in (2, 3):
        raise AudioValidationError(
            f"Audio tensor must be 2-D (C, T) or 3-D (B, C, T), "
            f"got shape {tuple(audio.shape)}."
        )

    # --- dtype -----------------------------------------------------------
    if audio.dtype not in (torch.float32, torch.float64):
        raise AudioValidationError(
            f"Audio tensor must have float32 or float64 dtype, got {audio.dtype}."
        )

    # --- Duration --------------------------------------------------------
    if sample_rate is not None:
        if sample_rate <= 0:
            raise AudioValidationError(
                f"sample_rate must be positive, got {sample_rate}."
            )
        num_samples = audio.shape[-1]
        duration_s = num_samples / sample_rate
        max_dur = get_max_duration_seconds()
        if duration_s > max_dur:
            raise AudioValidationError(
                f"Audio duration {duration_s:.3f}s exceeds the maximum allowed "
                f"{max_dur}s (configured in system_config.yaml)."
            )

    # --- Finiteness ------------------------------------------------------
    if check_finite:
        if not torch.isfinite(audio).all():
            num_bad = (~torch.isfinite(audio)).sum().item()
            raise AudioValidationError(
                f"Audio tensor contains {num_bad} non-finite value(s) (NaN or Inf)."
            )

    logger.debug("Audio tensor validated: shape=%s dtype=%s", tuple(audio.shape), audio.dtype)


def validate_normalized_params(
    params: dict[str, torch.Tensor],
    effect_name: str = "<unknown>",
) -> None:
    """Check that all parameter tensors are in [0, 1].

    Parameters
    ----------
    params : dict[str, torch.Tensor]
        Normalized parameter tensors for one effect.
    effect_name : str
        Effect name — used only in error messages.

    Raises
    ------
    AudioValidationError
        If any parameter is outside [0, 1] by more than a small epsilon.
    """
    eps = 1e-6
    for name, tensor in params.items():
        if not isinstance(tensor, torch.Tensor):
            raise AudioValidationError(
                f"Effect '{effect_name}', parameter '{name}': "
                f"expected torch.Tensor, got {type(tensor).__name__}."
            )
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        if min_val < -eps or max_val > 1.0 + eps:
            raise AudioValidationError(
                f"Effect '{effect_name}', parameter '{name}': "
                f"values outside [0, 1] — min={min_val:.6f}, max={max_val:.6f}."
            )
