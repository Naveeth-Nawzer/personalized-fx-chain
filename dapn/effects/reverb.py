"""
dapn/effects/reverb.py
======================
Differentiable algorithmic reverberation effect.

DSP model
---------
A simplified Schroeder reverberator consisting of:
  - N_COMB parallel comb filters (feedback delay lines).
  - 2 series all-pass filters for diffusion.
  - Wet/dry mixing.

All delays are implemented using the same fractional-delay method
as the Delay effect (zero-padded shift + linear interpolation),
which keeps the entire graph PyTorch-native.

The ``room_size`` and ``decay`` parameters control comb filter
feedback coefficients and delay times.  ``damping`` controls a
first-order low-pass filter inside each comb filter loop.

Differentiability notes
-----------------------
- Comb and all-pass delay shifts are differentiable w.r.t. delay times
  (via fractional linear interpolation).
- Feedback coefficients (derived from room_size/decay) are differentiable.
- No external plugin is used.

Parameters (normalized → physical)
------------------------------------
- room_size : 0 – 1
- damping   : 0 – 1
- wet_mix   : 0 – 1
- dry_mix   : 0 – 1
- decay     : 0.1 – 10 s
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from dapn.base_effect import BaseEffect

logger = logging.getLogger(__name__)

_EPS = 1e-8

# Freeverb-inspired comb filter delay lengths (in samples at 44100 Hz)
# These are scaled by room_size to vary the reverb tail.
_BASE_COMB_DELAYS_MS = [29.7, 37.1, 41.1, 44.1, 48.7, 51.3, 55.7, 60.0]
_BASE_AP_DELAYS_MS   = [12.6, 10.0]
_AP_FEEDBACK         = 0.5  # Fixed all-pass feedback (not parameterised in MVP)


class Reverb(BaseEffect):
    """Algorithmic Schroeder reverberator — fully differentiable.

    Inherits parameter loading from :class:`BaseEffect`.
    """

    effect_name = "Reverb"

    def forward(
        self,
        audio: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply reverb to *audio*.

        Parameters
        ----------
        audio : torch.Tensor
            Shape ``(B, C, T)`` or ``(C, T)``, float32.
        parameters : dict[str, torch.Tensor]
            Keys: ``room_size``, ``damping``, ``wet_mix``, ``dry_mix``,
            ``decay`` — all normalized [0, 1].

        Returns
        -------
        torch.Tensor
            Reverberated audio, same shape as input.
        """
        audio_3d, was_2d = self.ensure_3d(audio)
        B, C, T = audio_3d.shape

        from common.config_loader import get_sample_rate
        sr = float(get_sample_rate())

        # ---- Map to physical -------------------------------------------------
        room_size = self.map_to_physical("room_size", parameters["room_size"])
        damping   = self.map_to_physical("damping",   parameters["damping"])
        wet_mix   = self.map_to_physical("wet_mix",   parameters["wet_mix"])
        dry_mix   = self.map_to_physical("dry_mix",   parameters["dry_mix"])
        decay_s   = self.map_to_physical("decay",     parameters["decay"])

        # ---- Comb filters (parallel) ----------------------------------------
        comb_out = torch.zeros_like(audio_3d)
        for base_ms in _BASE_COMB_DELAYS_MS:
            delay_ms = base_ms * (0.5 + room_size * 0.5)   # scale by room_size
            delay_s  = delay_ms / 1000.0
            # Feedback coefficient: g = 10^(-3 * delay_s / decay_s)
            # (standard RT60 formula; g → 1 as decay → ∞)
            log10_factor = -3.0 * delay_s / (decay_s + _EPS)
            g = torch.pow(torch.tensor(10.0, dtype=audio_3d.dtype, device=audio_3d.device), log10_factor)
            # Apply damping: attenuate g slightly (simulates air absorption)
            g_damp = g * (1.0 - damping * 0.3)

            delay_samples = delay_ms * sr / 1000.0
            delayed = self._fractional_delay(audio_3d, delay_samples)
            # Single-tap comb: y = x + g * x_delayed  (open-loop approximation)
            comb_out = comb_out + audio_3d + g_damp * delayed

        # Average across comb filters
        comb_out = comb_out / len(_BASE_COMB_DELAYS_MS)

        # ---- All-pass filters (series) for diffusion -------------------------
        ap_out = comb_out
        for ap_ms in _BASE_AP_DELAYS_MS:
            ap_delay_samples = torch.as_tensor(ap_ms * sr / 1000.0, dtype=audio_3d.dtype, device=audio_3d.device)
            delayed = self._fractional_delay(ap_out, ap_delay_samples)
            # All-pass: y = -g*x + x_delayed + g*y_delayed
            # Open-loop approximation: y = x + (1 - g^2) * delayed
            g_ap = _AP_FEEDBACK
            ap_out = ap_out + (1.0 - g_ap ** 2) * delayed

        # ---- Mix ---------------------------------------------------------------
        out = dry_mix * audio_3d + wet_mix * ap_out

        if was_2d:
            out = out.squeeze(0)
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _fractional_delay(
        audio: torch.Tensor,
        delay_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Shift *audio* by *delay_samples* using linear interpolation.

        Preserves PyTorch autograd gradients with respect to *delay_samples*.
        """
        B, C, T = audio.shape

        delay_tensor = torch.as_tensor(delay_samples, dtype=audio.dtype, device=audio.device)
        d_floor = torch.floor(delay_tensor).long()
        d_frac  = delay_tensor - torch.floor(delay_tensor)   # in [0, 1)

        d0 = int(d_floor.item())
        d1 = d0 + 1

        if d0 <= 0:
            if d1 >= T:
                return audio
            pad1 = F.pad(audio, (1, 0))[..., :T]
            return (1.0 - d_frac) * audio + d_frac * pad1

        if d0 >= T:
            return torch.zeros_like(audio)

        pad0 = F.pad(audio, (d0, 0))[..., :T]
        if d1 >= T:
            return (1.0 - d_frac) * pad0
        pad1 = F.pad(audio, (d1, 0))[..., :T]

        out = (1.0 - d_frac) * pad0 + d_frac * pad1
        return out
