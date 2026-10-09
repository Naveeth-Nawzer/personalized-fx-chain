"""Sequential effect chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.effects.base import BaseEffect


@dataclass(frozen=True)
class EffectStep:
    """One effect plus the physical parameter values it is run with."""

    effect: BaseEffect
    parameters: Mapping[str, float] = field(default_factory=dict)


class EffectChain:
    """Applies effects one after another, in the order given.

    The chain itself is order-agnostic so it can be reused for future
    order-prediction experiments; Phase 1 always builds it in the canonical
    order (see :func:`src.effects.registry.canonical_order`).
    """

    def __init__(self, steps: Sequence[EffectStep]) -> None:
        if not steps:
            raise ValueError("An effect chain needs at least one effect")
        self.steps: tuple[EffectStep, ...] = tuple(steps)

    @property
    def effect_names(self) -> list[str]:
        return [step.effect.name for step in self.steps]

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Run every step in order and return the final output."""
        out = audio
        for step in self.steps:
            out = step.effect.process(out, sample_rate, step.parameters)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Canonical FX-chain dict (same layout as the team's DAPN schema, physical units)."""
        return {"chain": [{"effect": s.effect.name, "parameters": dict(s.parameters)} for s in self.steps]}
