"""
dapn/effects/distortion.py
==========================
Differentiable waveshaping distortion effect with tone filtering.

DSP model
---------
1. Input pre-gain amplification controlled by `drive`.
2. Smooth soft-clipping saturation via tanh waveshaper.
3. Differentiable tone low-pass filter (FIR impulse response formulation).
4. Output makeup gain and wet/dry mix.

Parameters (normalized [0, 1] -> physical)
------------------------------------------
- drive       : 0.0 – 1.0 (dimensionless)
- tone        : 0.0 – 1.0 (dimensionless, 0 = dark/warm, 1 = bright)
- mix         : 0.0 – 1.0 (dimensionless, wet/dry blend)
- output_gain : -12.0 – +6.0 dB
"""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

from dapn.base_effect import BaseEffect

logger = logging.getLogger(__name__)

_TONE_TAPS = 64
_EPS = 1e-8


class Distortion(BaseEffect):
    """Differentiable waveshaping distortion module."""

    effect_name = "Distortion"

    def forward(
        self,
        audio: torch.Tensor,
        parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply distortion to audio.

        Parameters
        ----------
        audio : torch.Tensor
            Shape (B, C, T) or (C, T), float32.
        parameters : dict[str, torch.Tensor]
            Keys: 'drive', 'tone', 'mix', 'output_gain' in [0, 1].

        Returns
        -------
        torch.Tensor
            Distorted audio, same shape as input.
        """
        audio_3d, was_2d = self.ensure_3d(audio)
        B, C, T = audio_3d.shape

        # 1. Map to physical
        drive = self.map_to_physical("drive", parameters["drive"])
        tone = self.map_to_physical("tone", parameters["tone"])
        mix = self.map_to_physical("mix", parameters["mix"])
        output_gain_db = self.map_to_physical("output_gain", parameters["output_gain"])

        # 2. Input pre-gain amplification (drive: 0 -> 1x, 1 -> 25x gain)
        pre_gain = 1.0 + 24.0 * drive
        driven_audio = audio_3d * pre_gain

        # 3. Soft-clipping saturation via tanh waveshaper
        saturated = torch.tanh(driven_audio)

        # 4. Differentiable tone filter
        # Alpha controls low-pass smoothing:
        # tone=0 -> alpha=0.1 (dark/filtered), tone=1 -> alpha=0.99 (bright/bypass)
        alpha = 0.10 + 0.89 * tone
        n = torch.arange(_TONE_TAPS, dtype=audio_3d.dtype, device=audio_3d.device)
        decay = torch.pow((1.0 - alpha).clamp(min=1e-4, max=0.999), n)
        kernel = alpha * decay
        kernel = kernel / (kernel.sum() + _EPS)  # Normalized FIR filter
        kernel_3d = kernel.view(1, 1, -1)

        # Causal grouped conv1d
        pad_len = _TONE_TAPS - 1
        audio_flat = saturated.view(B * C, 1, T)
        padded = F.pad(audio_flat, (pad_len, 0))
        filtered_flat = F.conv1d(padded, kernel_3d)
        filtered = filtered_flat.view(B, C, T)

        # 5. Output makeup gain
        makeup_linear = torch.pow(torch.tensor(10.0, dtype=audio_3d.dtype, device=audio_3d.device), output_gain_db / 20.0)
        wet = filtered * makeup_linear

        # 6. Wet/Dry mix
        out = (1.0 - mix) * audio_3d + mix * wet

        if was_2d:
            out = out.squeeze(0)
        return out
