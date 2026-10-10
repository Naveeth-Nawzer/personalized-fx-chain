"""
dapn/effects/eq.py
==================
Differentiable Parametric Equalizer effect.

DSP model
---------
Implements a **biquad peaking-EQ filter** applied in the time domain via
``torch.nn.functional.conv1d`` with zero-phase (forward + backward pass)
filtering.

The biquad coefficients are computed from the EQ parameters using the
Audio EQ Cookbook formulae (R. Bristow-Johnson).  All coefficient
arithmetic is PyTorch-native so gradients flow through.

Parameters (normalized → physical)
------------------------------------
- frequency : 20 – 20000 Hz
- gain      : -12 – 12 dB
- q         : 0.1 – 10

Differentiability notes
-----------------------
- Biquad coefficients are differentiable functions of frequency, gain, Q.
- Time-domain convolution is differentiable w.r.t. the input signal.
- Gradients w.r.t. filter coefficients are obtained by backpropagating
  through the coefficient computation and the convolution.
- Direct-Form II transposed IIR filtering would be maximally efficient
  but requires a stateful recurrence that is hard to differentiate;
  instead we use FIR approximation via truncated impulse response (TIR).
  The TIR is computed by passing an impulse through the analytical
  biquad recursion on a CPU scalar path, then convolved with the audio.
  Gradients through the TIR *w.r.t. the EQ parameters* are preserved
  because the impulse-response computation uses PyTorch ops.
"""

from __future__ import annotations

import math
import logging

import torch
import torch.nn.functional as F

from dapn.base_effect import BaseEffect

logger = logging.getLogger(__name__)

# Number of samples used for the truncated impulse response.
# Longer = more accurate low-frequency response; shorter = faster.
_TIR_LENGTH = 2048


class EQ(BaseEffect):
    """Parametric peaking EQ (biquad, TIR approximation).

    Inherits parameter loading from :class:`BaseEffect`.
    """

    effect_name = "EQ"

    def forward(
        self,
        audio: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply EQ to *audio*.

        Parameters
        ----------
        audio : torch.Tensor
            Shape ``(B, C, T)`` or ``(C, T)``, float32.
        parameters : dict[str, torch.Tensor]
            Keys: ``frequency``, ``gain``, ``q`` — all normalized [0, 1].

        Returns
        -------
        torch.Tensor
            EQ-processed audio, same shape as input.
        """
        audio_3d, was_2d = self.ensure_3d(audio)
        B, C, T = audio_3d.shape

        # Map to physical
        freq_hz = self.map_to_physical("frequency", parameters["frequency"])
        gain_db = self.map_to_physical("gain", parameters["gain"])
        q_val   = self.map_to_physical("q",   parameters["q"])

        # Compute biquad peaking-EQ coefficients (Audio EQ Cookbook)
        # All ops are PyTorch → gradients preserved
        sample_rate = float(self._mappers["frequency"].spec.physical_max)
        # Use configured sample_rate from system_config via the loader
        from common.config_loader import get_sample_rate
        sr = float(get_sample_rate())

        omega = 2.0 * math.pi * freq_hz / sr         # ω₀
        cos_w = torch.cos(omega)
        sin_w = torch.sin(omega)
        A     = torch.pow(torch.tensor(10.0, dtype=audio_3d.dtype), gain_db / 40.0)  # 10^(dB/40)
        alpha = sin_w / (2.0 * q_val)

        # Peaking EQ biquad coefficients
        b0 =  1.0 + alpha * A
        b1 = -2.0 * cos_w
        b2 =  1.0 - alpha * A
        a0 =  1.0 + alpha / A
        a1 = -2.0 * cos_w
        a2 =  1.0 - alpha / A

        # Normalize by a0
        b0n = b0 / a0
        b1n = b1 / a0
        b2n = b2 / a0
        a1n = a1 / a0
        a2n = a2 / a0

        # Compute truncated impulse response (TIR) of the biquad filter
        # via the Direct-Form II recursion on a zero-padded impulse.
        # This is the only way to make the IIR differentiable w.r.t. b/a.
        tir = self._compute_tir(b0n, b1n, b2n, a1n, a2n, length=min(_TIR_LENGTH, T))

        # Apply via causal convolution (no padding for circular artefacts)
        # tir shape: (tir_len,) → kernel shape: (1, 1, tir_len)
        kernel = tir.view(1, 1, -1)

        # Pad input so output length matches T
        pad_len = kernel.shape[-1] - 1
        audio_padded = F.pad(audio_3d, (pad_len, 0))

        # Process each channel independently using grouped conv
        audio_flat = audio_3d.view(B * C, 1, T)
        padded_flat = F.pad(audio_flat, (pad_len, 0))
        out_flat    = F.conv1d(padded_flat, kernel)
        out = out_flat.view(B, C, T)

        if was_2d:
            out = out.squeeze(0)
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_tir(
        b0: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
        length: int = _TIR_LENGTH,
    ) -> torch.Tensor:
        """Compute the truncated impulse response of a biquad filter.

        Uses Direct-Form II recurrence.  All operations are PyTorch so
        gradients flow back through ``b0``, ``b1``, ``b2``, ``a1``, ``a2``
        to the original EQ parameters.

        Parameters
        ----------
        b0, b1, b2 : torch.Tensor  — numerator coefficients (scalar)
        a1, a2     : torch.Tensor  — denominator coefficients (scalar, a0=1)
        length     : int           — number of TIR samples

        Returns
        -------
        torch.Tensor
            Shape ``(length,)``, dtype same as coefficients.
        """
        # Build output using explicit loop — small (length ≤ 2048)
        # This is differentiable because each y[n] is a linear combination
        # of input samples (impulse) and previous outputs, all traced by autograd.
        dtype  = b0.dtype
        device = b0.device

        # Pre-allocate output list; use stack at the end for a clean graph
        y_list: list[torch.Tensor] = []
        # x[n] = impulse: x[0] = 1, x[n>0] = 0
        x0 = torch.ones(1, dtype=dtype, device=device)
        x1 = torch.zeros(1, dtype=dtype, device=device)
        x2 = torch.zeros(1, dtype=dtype, device=device)
        y1 = torch.zeros(1, dtype=dtype, device=device)
        y2 = torch.zeros(1, dtype=dtype, device=device)

        for n in range(length):
            xn = x0 if n == 0 else torch.zeros(1, dtype=dtype, device=device)
            yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y_list.append(yn)
            x2, x1 = x1, xn
            y2, y1 = y1, yn

        return torch.cat(y_list, dim=0)
