"""
dapn/effects/chorus.py
======================
Differentiable Chorus modulation effect.

DSP model
---------
Chorus introduces a shimmering doubling effect by mixing the input signal with
a continuously time-varying delayed copy modulated by a low-frequency oscillator (LFO):

  d(t) = delay_base + depth * max_mod * sin(2 * pi * rate * t)
  y_wet(t) = x(t - d(t))
  output = (1 - mix) * x(t) + mix * y_wet(t)

Differentiability
-----------------
Time-varying continuous fractional delay is achieved using PyTorch's native
`torch.nn.functional.grid_sample`, ensuring full autograd support w.r.t. `rate`,
`depth`, `delay`, and `mix`.

Parameters (normalized [0, 1] -> physical)
------------------------------------------
- rate  : 0.1 – 10.0 Hz (LFO modulation frequency)
- depth : 0.0 – 1.0 (modulation depth)
- delay : 1.0 – 50.0 ms (base delay time)
- mix   : 0.0 – 1.0 (wet/dry blend)
"""

from __future__ import annotations

import logging
import math
import torch
import torch.nn.functional as F

from dapn.base_effect import BaseEffect

logger = logging.getLogger(__name__)


class Chorus(BaseEffect):
    """Differentiable multi-voice Chorus effect."""

    effect_name = "Chorus"

    def forward(
        self,
        audio: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply chorus modulation to audio.

        Parameters
        ----------
        audio : torch.Tensor
            Shape (B, C, T) or (C, T), float32.
        parameters : dict[str, torch.Tensor]
            Keys: 'rate', 'depth', 'delay', 'mix' in [0, 1].

        Returns
        -------
        torch.Tensor
            Chorus-processed audio, same shape as input.
        """
        audio_3d, was_2d = self.ensure_3d(audio)
        B, C, T = audio_3d.shape

        from common.config_loader import get_sample_rate
        sr = float(get_sample_rate())

        # 1. Map parameters to physical values
        rate_hz = self.map_to_physical("rate", parameters["rate"])
        depth = self.map_to_physical("depth", parameters["depth"])
        delay_ms = self.map_to_physical("delay", parameters["delay"])
        mix = self.map_to_physical("mix", parameters["mix"])

        # 2. Base delay and modulation depth in samples
        base_delay_samples = delay_ms * sr / 1000.0
        # Modulation swing up to 8 ms
        max_mod_samples = 8.0 * sr / 1000.0

        # 3. LFO time vector and continuous delay buffer
        t = torch.arange(T, dtype=audio_3d.dtype, device=audio_3d.device)
        lfo = torch.sin(2.0 * math.pi * rate_hz * t / sr)
        delay_samples_t = base_delay_samples + depth * max_mod_samples * lfo

        # Read position at each sample t: r(t) = t - delay(t)
        read_pos = t - delay_samples_t

        # 4. Continuous fractional sampling via F.grid_sample
        # Normalize read_pos to grid coordinates [-1.0, 1.0]
        norm_x = 2.0 * read_pos / max(T - 1, 1) - 1.0
        grid = torch.stack([norm_x, torch.zeros_like(norm_x)], dim=-1)  # (T, 2)
        grid_4d = grid.view(1, 1, T, 2).expand(B * C, -1, -1, -1)

        audio_flat_4d = audio_3d.view(B * C, 1, 1, T)
        wet_flat = F.grid_sample(
            audio_flat_4d,
            grid_4d,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        wet = wet_flat.view(B, C, T)

        # 5. Wet/Dry mix
        out = (1.0 - mix) * audio_3d + mix * wet

        if was_2d:
            out = out.squeeze(0)
        return out
