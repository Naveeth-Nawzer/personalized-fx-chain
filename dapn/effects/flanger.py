"""
dapn/effects/flanger.py
=======================
Differentiable Flanger modulation effect with feedback comb-filtering.

DSP model
---------
A flanger creates dynamic sweeping comb-filtering notches by modulating an
ultra-short delay line (0.5 – 5 ms) with feedback:

  d(t) = delay_base + depth * max_mod * (0.5 + 0.5 * sin(2 * pi * rate * t))
  y_wet(t) = sum_{k=1}^K feedback^(k-1) * x(t - k * d(t))
  output = (1 - mix) * x(t) + mix * y_wet(t)

Parameters (normalized [0, 1] -> physical)
------------------------------------------
- rate     : 0.1 – 10.0 Hz (LFO sweep speed)
- depth    : 0.0 – 1.0 (modulation depth)
- feedback : 0.0 – 0.95 (resonance / comb depth)
- mix      : 0.0 – 1.0 (wet/dry blend)
"""

from __future__ import annotations

import logging
import math
import torch
import torch.nn.functional as F

from dapn.base_effect import BaseEffect

logger = logging.getLogger(__name__)

_FLANGER_TAPS = 4


class Flanger(BaseEffect):
    """Differentiable resonant Flanger effect."""

    effect_name = "Flanger"

    def forward(
        self,
        audio: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply flanger effect to audio.

        Parameters
        ----------
        audio : torch.Tensor
            Shape (B, C, T) or (C, T), float32.
        parameters : dict[str, torch.Tensor]
            Keys: 'rate', 'depth', 'feedback', 'mix' in [0, 1].

        Returns
        -------
        torch.Tensor
            Flanged audio, same shape as input.
        """
        audio_3d, was_2d = self.ensure_3d(audio)
        B, C, T = audio_3d.shape

        from common.config_loader import get_sample_rate
        sr = float(get_sample_rate())

        # 1. Map parameters to physical values
        rate_hz = self.map_to_physical("rate", parameters["rate"])
        depth = self.map_to_physical("depth", parameters["depth"])
        feedback = self.map_to_physical("feedback", parameters["feedback"])
        mix = self.map_to_physical("mix", parameters["mix"])

        # 2. Base delay and modulation swing (ultra-short: 1.0 ms to 4.0 ms)
        base_delay_samples = 1.0 * sr / 1000.0
        max_mod_samples = 3.0 * sr / 1000.0

        # 3. LFO time vector and continuous modulation buffer
        t = torch.arange(T, dtype=audio_3d.dtype, device=audio_3d.device)
        lfo = 0.5 + 0.5 * torch.sin(2.0 * math.pi * rate_hz * t / sr)
        single_delay_t = base_delay_samples + depth * max_mod_samples * lfo

        # 4. Multi-tap feedback accumulation via F.grid_sample
        audio_flat_4d = audio_3d.view(B * C, 1, 1, T)
        wet_accum = torch.zeros_like(audio_3d)

        for k in range(1, _FLANGER_TAPS + 1):
            delay_k = float(k) * single_delay_t
            read_pos_k = t - delay_k

            # Normalize to grid coordinates [-1.0, 1.0]
            norm_x_k = 2.0 * read_pos_k / max(T - 1, 1) - 1.0
            grid_k = torch.stack([norm_x_k, torch.zeros_like(norm_x_k)], dim=-1)
            grid_4d_k = grid_k.view(1, 1, T, 2).expand(B * C, -1, -1, -1)

            sampled_k = F.grid_sample(
                audio_flat_4d,
                grid_4d_k,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).view(B, C, T)

            fb_gain = torch.pow(feedback, float(k - 1))
            wet_accum = wet_accum + fb_gain * sampled_k

        # 5. Wet/Dry mix
        out = (1.0 - mix) * audio_3d + mix * wet_accum

        if was_2d:
            out = out.squeeze(0)
        return out
