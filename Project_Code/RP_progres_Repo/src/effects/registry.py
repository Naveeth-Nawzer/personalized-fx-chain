"""Effect registry: canonical names, canonical order and backend factory."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from src.effects.base import BaseEffect, EffectError
from src.effects.compressor import CompressorEffect
from src.effects.eq import EQEffect
from src.effects.pedalboard_backend import (
    PedalboardCompressorEffect,
    PedalboardEQEffect,
    PedalboardReverbEffect,
)
from src.effects.reverb import ReverbEffect

#: Canonical internal processing order (Phase 1). Also the label order.
CANONICAL_ORDER: tuple[str, ...] = ("EQ", "Compressor", "Reverb")

BACKENDS: dict[str, dict[str, type[BaseEffect]]] = {
    "scipy": {"EQ": EQEffect, "Compressor": CompressorEffect, "Reverb": ReverbEffect},
    "pedalboard": {"EQ": PedalboardEQEffect, "Compressor": PedalboardCompressorEffect,
                   "Reverb": PedalboardReverbEffect},
}


def create_effect(name: str, backend: str = "scipy", options: Mapping[str, Any] | None = None) -> BaseEffect:
    """Instantiate an effect.

    Args:
        name: Canonical effect name (``"EQ"``, ``"Compressor"``, ``"Reverb"``).
        backend: ``"scipy"`` or ``"pedalboard"``.
        options: Backend options (currently only used by the scipy reverb).
    """
    if backend not in BACKENDS:
        raise EffectError(f"Unknown DSP backend '{backend}' (known: {sorted(BACKENDS)})")
    classes = BACKENDS[backend]
    if name not in classes:
        raise EffectError(f"Unknown effect '{name}' (known: {list(CANONICAL_ORDER)})")
    cls = classes[name]
    if cls is ReverbEffect:
        return ReverbEffect(options)
    return cls()


def canonical_order(names: Iterable[str]) -> list[str]:
    """Sort effect names into the canonical processing order (EQ -> Compressor -> Reverb).

    Raises:
        EffectError: on unknown or duplicated names.
    """
    names = list(names)
    unknown = [n for n in names if n not in CANONICAL_ORDER]
    if unknown:
        raise EffectError(f"Unknown effect(s) {unknown} (known: {list(CANONICAL_ORDER)})")
    if len(set(names)) != len(names):
        raise EffectError(f"Duplicate effects in chain: {names}")
    return sorted(names, key=CANONICAL_ORDER.index)
