"""
common/audio_types.py
=====================
Canonical type aliases and dataclasses for audio tensors used
throughout the DAPN pipeline.

Design goals
------------
- Single source of truth for tensor dimension semantics.
- Avoids scattering shape comments across modules.
- No runtime overhead beyond plain Python type hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

# ---------------------------------------------------------------------------
# Scalar aliases — document *what* a float means, not just its Python type
# ---------------------------------------------------------------------------

NormalizedParam = float  # value in [0, 1]  — the ML-facing representation
PhysicalParam   = float  # value in physical DSP units

SampleRate      = int    # Hz
NumChannels     = int    # 1 = mono, 2 = stereo
NumSamples      = int    # T — time dimension

# ---------------------------------------------------------------------------
# Audio tensor shape contract
# ---------------------------------------------------------------------------
# All DAPN tensors must conform to one of:
#
#   (B, C, T)  — batched:   batch × channels × time-samples
#   (C, T)     — unbatched: channels × time-samples
#
# The pipeline works with both.  Individual effects receive (B, C, T) tensors;
# the FXChain automatically adds / removes the batch dimension as needed.
# ---------------------------------------------------------------------------

AudioTensor = torch.Tensor  # shape: (B, C, T) or (C, T), dtype=float32

# Valid dtype string understood by torch.get_default_dtype()
DTypeName = Literal["float32", "float64"]


@dataclass(frozen=True)
class AudioMeta:
    """Metadata describing an audio tensor — *not* the tensor itself.

    Kept separate from the tensor so metadata can be passed around
    cheaply without carrying audio data.
    """

    sample_rate: SampleRate
    channels: NumChannels
    num_samples: NumSamples
    dtype: DTypeName = "float32"

    @property
    def duration_seconds(self) -> float:
        """Duration of the audio in seconds."""
        return self.num_samples / self.sample_rate

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, sample_rate: SampleRate) -> "AudioMeta":
        """Create AudioMeta by inspecting a (C, T) or (B, C, T) tensor."""
        if tensor.ndim == 2:  # (C, T)
            channels, num_samples = tensor.shape
        elif tensor.ndim == 3:  # (B, C, T)
            _, channels, num_samples = tensor.shape
        else:
            raise ValueError(
                f"Expected a 2-D (C, T) or 3-D (B, C, T) tensor, got shape {tuple(tensor.shape)}"
            )
        dtype_map: dict[torch.dtype, DTypeName] = {
            torch.float32: "float32",
            torch.float64: "float64",
        }
        dtype_name: DTypeName = dtype_map.get(tensor.dtype, "float32")
        return cls(
            sample_rate=sample_rate,
            channels=channels,
            num_samples=num_samples,
            dtype=dtype_name,
        )


@dataclass(frozen=True)
class EffectParameterSpec:
    """Describes a single effect parameter as loaded from effects_config.yaml."""

    name: str
    normalized_min: float   # always 0.0 for current MVP
    normalized_max: float   # always 1.0 for current MVP
    physical_min: float
    physical_max: float
    unit: str
    mapping_type: Literal["linear", "log"]  # only "linear" in MVP
    required: bool
