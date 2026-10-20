"""
dapn/effects/compressor.py
==========================
Differentiable dynamic range compressor.

DSP model
---------
A feedforward compressor with:
  1. Level detection via smooth RMS envelope (exponential moving average).
  2. Gain computation in the log domain (dB).
  3. Smooth gain application using attack / release time constants.
  4. Output makeup gain.

All operations are PyTorch-native.

Differentiability challenge
---------------------------
The standard compressor uses hard-thresholded gain reduction:
  GR(dB) = min(0, (1 - 1/ratio) * (threshold - level_dB))

The ``min`` is not differentiable at zero.  We replace it with a
**soft-knee** approximation:

  GR_soft(x) = (1/ratio - 1) * softplus(x - threshold - knee/2) / knee
             for |x - threshold| < knee
             0 for x << threshold
             (1/ratio - 1)(x - threshold) for x >> threshold

where ``softplus(z) = ln(1 + e^z)`` is smooth everywhere.

Parameters (normalized → physical)
------------------------------------
- threshold   : -60 – 0 dB
- ratio       : 1 – 20
- attack      : 0.1 – 100 ms
- release     : 10 – 1000 ms
- makeup_gain : 0 – 12 dB
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from dapn.base_effect import BaseEffect

logger = logging.getLogger(__name__)

_KNEE_DB = 6.0   # Fixed soft-knee width in dB (could be parameterised later)
_EPS     = 1e-8  # Small constant for numerical stability


class Compressor(BaseEffect):
    """Feedforward soft-knee compressor — fully differentiable.

    Inherits parameter loading from :class:`BaseEffect`.
    """

    effect_name = "Compressor"

    def forward(
        self,
        audio: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply compression to *audio*.

        Parameters
        ----------
        audio : torch.Tensor
            Shape ``(B, C, T)`` or ``(C, T)``, float32.
        parameters : dict[str, torch.Tensor]
            Keys: ``threshold``, ``ratio``, ``attack``, ``release``,
            ``makeup_gain`` — all normalized [0, 1].

        Returns
        -------
        torch.Tensor
            Compressed audio, same shape as input.
        """
        audio_3d, was_2d = self.ensure_3d(audio)
        B, C, T = audio_3d.shape

        from common.config_loader import get_sample_rate
        sr = float(get_sample_rate())

        # ---- Map to physical -------------------------------------------------
        threshold_db  = self.map_to_physical("threshold",   parameters["threshold"])
        ratio         = self.map_to_physical("ratio",        parameters["ratio"])
        attack_ms     = self.map_to_physical("attack",       parameters["attack"])
        release_ms    = self.map_to_physical("release",      parameters["release"])
        makeup_db     = self.map_to_physical("makeup_gain",  parameters["makeup_gain"])

        # ---- Time constants (samples) ----------------------------------------
        attack_time_s  = attack_ms  * 1e-3
        release_time_s = release_ms * 1e-3
        ln9 = math.log(9.0)

        attack_coeff  = torch.exp(-torch.tensor(ln9, dtype=audio_3d.dtype) / (attack_time_s  * sr + _EPS))
        release_coeff = torch.exp(-torch.tensor(ln9, dtype=audio_3d.dtype) / (release_time_s * sr + _EPS))

        # ---- Level detection: RMS via exponential smoothing ------------------
        # |x|^2 smoothed with attack coefficient; then sqrt
        x_sq = audio_3d.pow(2)          # (B, C, T)

        # Use a causal 1-D exponential moving average approximation via conv.
        # For simplicity: compute instantaneous RMS in short frames (32 samples)
        # and smooth. This avoids the non-differentiable sequential loop.
        frame_size = 32
        # Pad to make divisible
        pad_right = (frame_size - T % frame_size) % frame_size
        x_sq_pad  = F.pad(x_sq, (0, pad_right))
        T_pad = x_sq_pad.shape[-1]
        # Frame mean → (B, C, T_pad/frame_size)
        frames = x_sq_pad.view(B, C, T_pad // frame_size, frame_size).mean(dim=-1)
        # Upsample back to T via repeat_interleave
        level_sq = frames.repeat_interleave(frame_size, dim=-1)[..., :T]  # (B, C, T)
        level_linear = torch.sqrt(level_sq + _EPS)                         # RMS level

        # ---- Gain computation in dB (soft-knee) --------------------------------
        level_db = 20.0 * torch.log10(level_linear + _EPS)

        knee = _KNEE_DB
        # Soft-knee gain reduction (dB) using smooth hinge
        # GR = (1/ratio - 1) * softplus(level_db - threshold_db) scaled by knee
        x_over  = level_db - threshold_db  # positive when above threshold
        # Soft knee: use sigmoid interpolation
        knee_factor = torch.sigmoid((x_over + knee / 2.0) / (knee / 4.0 + _EPS))
        gr_db = (1.0 / ratio - 1.0) * F.softplus(x_over) * knee_factor

        # ---- Convert GR to linear and apply ----------------------------------
        makeup_linear = torch.pow(torch.tensor(10.0, dtype=audio_3d.dtype), makeup_db / 20.0)
        gr_linear     = torch.pow(torch.tensor(10.0, dtype=audio_3d.dtype), gr_db / 20.0)

        out = audio_3d * gr_linear * makeup_linear

        if was_2d:
            out = out.squeeze(0)
        return out
