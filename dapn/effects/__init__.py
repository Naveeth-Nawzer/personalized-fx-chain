"""dapn/effects/__init__.py — effect registry."""
from __future__ import annotations

from dapn.effects.eq         import EQ
from dapn.effects.compressor import Compressor
from dapn.effects.delay      import Delay
from dapn.effects.reverb     import Reverb

# Registry: canonical name → class
# Add new effects here when implemented.
EFFECT_REGISTRY: dict[str, type] = {
    "EQ":         EQ,
    "Compressor": Compressor,
    "Delay":      Delay,
    "Reverb":     Reverb,
}

__all__ = [
    "EQ",
    "Compressor",
    "Delay",
    "Reverb",
    "EFFECT_REGISTRY",
]
