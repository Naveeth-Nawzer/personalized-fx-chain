"""Audio effects behind a backend-independent interface.

    BaseEffect
      ├── EQ          (scipy: EQEffect          | pedalboard: PedalboardEQEffect)
      ├── Compressor  (scipy: CompressorEffect  | pedalboard: PedalboardCompressorEffect)
      └── Reverb      (scipy: ReverbEffect      | pedalboard: PedalboardReverbEffect)

Use :func:`create_effect` to build effects and :class:`EffectChain` to run them.
"""

from src.effects.base import BackendUnavailableError, BaseEffect, EffectError
from src.effects.chain import EffectChain, EffectStep
from src.effects.parameters import ParameterSampler, ParameterSpec
from src.effects.registry import BACKENDS, CANONICAL_ORDER, canonical_order, create_effect

__all__ = [
    "BACKENDS",
    "CANONICAL_ORDER",
    "BackendUnavailableError",
    "BaseEffect",
    "EffectChain",
    "EffectError",
    "EffectStep",
    "ParameterSampler",
    "ParameterSpec",
    "canonical_order",
    "create_effect",
]
