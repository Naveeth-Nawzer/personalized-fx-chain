"""
dapn/effects/delay.py
=====================
Differentiable tape-style delay / echo effect.

DSP model
---------
Output = dry_mix * x(t) + wet_mix * y_delayed(t)

where ``y_delayed`` is a feedback echo computed iteratively.

For a differentiable implementation, we avoid Python loops over time
samples.  Instead, we precompute the delay buffer as a **fractional
sample shift** via linear interpolation and accumulate feedback echos
as a geometric series truncated at a fixed number of taps.

Fractional delay (differentiable)
-----------------------------------
The delay time in samples ``d = delay_time_ms * sr / 1000`` may be
fractional.  We implement fractional delay using linear interpolation
between the two nearest integer-sample positions.  This is differentiable
w.r.t. ``delay_time_ms``.

Feedback accumulation
---------------------
Instead of an unrolled IIR feedback loop (which would require storing
full-length tensors at every tap), we approximate the infinite feedback
series with ``N_TAPS`` echos:

  y(t) = Σ_{k=0}^{N_TAPS-1}  feedback^k * x(t - k*d)

This converges quickly for ``feedback < 1`` (guaranteed by config).

Parameters (normalized → physical)
------------------------------------
- delay_time : 1 – 2000 ms
- feedback   : 0 – 0.95
- wet_mix    : 0 – 1
- dry_mix    : 0 – 1
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from dapn.base_effect import BaseEffect

logger = logging.getLogger(__name__)

N_TAPS = 8       # Number of feedback echoes to accumulate
_EPS   = 1e-8


class Delay(BaseEffect):
    """Tape-style echo delay with fractional-sample interpolation.

    Inherits parameter loading from :class:`BaseEffect`.
    """

    effect_name = "Delay"

    def forward(
        self,
        audio: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply delay to *audio*.

        Parameters
        ----------
        audio : torch.Tensor
            Shape ``(B, C, T)`` or ``(C, T)``, float32.
        parameters : dict[str, torch.Tensor]
            Keys: ``delay_time``, ``feedback``, ``wet_mix``, ``dry_mix``
            — all normalized [0, 1].

        Returns
        -------
        torch.Tensor
            Delayed audio, same shape as input.
        """
        audio_3d, was_2d = self.ensure_3d(audio)
        B, C, T = audio_3d.shape

        from common.config_loader import get_sample_rate
        sr = float(get_sample_rate())

        # ---- Map to physical ------------------------------------------------
        delay_ms  = self.map_to_physical("delay_time", parameters["delay_time"])
        feedback  = self.map_to_physical("feedback",   parameters["feedback"])
        wet_mix   = self.map_to_physical("wet_mix",    parameters["wet_mix"])
        dry_mix   = self.map_to_physical("dry_mix",    parameters["dry_mix"])

        # Delay in samples (possibly fractional)
        delay_samples = delay_ms * sr / 1000.0   # float tensor

        # ---- Build delayed signal via fractional shift ----------------------
        # For each tap k (0…N_TAPS-1), shift audio by k * delay_samples
        # and weight by feedback^k.  Sum all taps.
        wet = torch.zeros_like(audio_3d)

        for k in range(1, N_TAPS + 1):
            fb_weight = feedback ** (k - 1)                  # feedback attenuation per tap
            shift_s   = delay_samples * float(k)             # shift is k * delay_time

            shifted = self._fractional_delay(audio_3d, shift_s)
            wet = wet + fb_weight * shifted

        out = dry_mix * audio_3d + wet_mix * wet

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

        Parameters
        ----------
        audio : torch.Tensor
            Shape ``(B, C, T)``.
        delay_samples : torch.Tensor
            Scalar tensor — fractional number of samples to delay.

        Returns
        -------
        torch.Tensor
            Delayed audio, shape ``(B, C, T)``.  Zero-padded on the left.
        """
        B, C, T = audio.shape

        d_floor = torch.floor(delay_samples).long()
        d_frac  = delay_samples - torch.floor(delay_samples)   # in [0, 1)

        # Clip to valid range
        d0 = int(d_floor.item())
        d1 = d0 + 1

        # Build integer-delayed versions via F.pad + slice
        # floor sample
        if d0 >= T:
            return torch.zeros_like(audio)
        pad0  = F.pad(audio, (d0, 0))[..., :T]

        if d1 >= T:
            return (1.0 - d_frac) * pad0
        pad1  = F.pad(audio, (d1, 0))[..., :T]

        # Linear interpolation (differentiable w.r.t. d_frac = delay_samples)
        out = (1.0 - d_frac) * pad0 + d_frac * pad1
        return out
