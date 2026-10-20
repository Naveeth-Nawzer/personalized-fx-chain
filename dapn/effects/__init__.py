"""dapn/effects/__init__.py — effect registry."""
from __future__ import annotations

from dapn.effects.eq         import EQ
from dapn.effects.compressor import Compressor
from dapn.effects.delay      import Delay
from dapn.effects.reverb     import Reverb
from dapn.effects.distortion import Distortion
from dapn.effects.chorus     import Chorus
from dapn.effects.flanger    import Flanger

# Registry: canonical name → class
EFFECT_REGISTRY: dict[str, type] = {
    "EQ":         EQ,
    "Compressor": Compressor,
    "Delay":      Delay,
    "Reverb":     Reverb,
    "Distortion": Distortion,
    "Chorus":     Chorus,
    "Flanger":    Flanger,
}

__all__ = [
    "EQ",
    "Compressor",
    "Delay",
    "Reverb",
    "Distortion",
    "Chorus",
    "Flanger",
    "EFFECT_REGISTRY",
]
