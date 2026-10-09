"""Multi-label effect targets and effect combinations.

The Phase 1 target is a fixed 3-dim binary vector ``[EQ, Compressor, Reverb]``.
It describes only which effects are present: not their order, not their
parameters and never the source instrument/category.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

from src.effects.registry import CANONICAL_ORDER, canonical_order

#: Label dimensions, in order.
LABEL_NAMES: tuple[str, ...] = CANONICAL_ORDER
#: CSV column for each label dimension.
LABEL_COLUMNS: dict[str, str] = {"EQ": "eq_label", "Compressor": "compressor_label", "Reverb": "reverb_label"}


def effect_combinations(enabled: Iterable[str] = CANONICAL_ORDER) -> list[tuple[str, ...]]:
    """All non-empty subsets of the enabled effects, each in canonical order.

    For the three Phase 1 effects this is, in order:
    EQ, Compressor, Reverb, EQ+Compressor, EQ+Reverb, Compressor+Reverb, EQ+Compressor+Reverb.
    """
    effects = canonical_order(set(enabled))
    return [combo for r in range(1, len(effects) + 1) for combo in combinations(effects, r)]


def label_vector(effect_chain: Sequence[str]) -> list[int]:
    """Binary ``[EQ, Compressor, Reverb]`` vector for a set of effects."""
    canonical_order(effect_chain)  # validates names / duplicates
    present = set(effect_chain)
    return [int(name in present) for name in LABEL_NAMES]


def label_columns(effect_chain: Sequence[str]) -> dict[str, int]:
    """``{"eq_label": .., "compressor_label": .., "reverb_label": ..}`` for a set of effects."""
    vector = label_vector(effect_chain)
    return {LABEL_COLUMNS[name]: value for name, value in zip(LABEL_NAMES, vector)}


def combination_name(effect_chain: Sequence[str]) -> str:
    """Human-readable combination key, e.g. ``"EQ+Reverb"``."""
    return "+".join(canonical_order(effect_chain))
