"""Reproducible, leakage-free train/validation/test splitting.

Splitting happens on SONGS (groups), never on generated samples:
every stem of a song, and all seven effect versions of each selected stem,
end up in the same split.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.utils.config import SPLITS


def allocate_counts(total: int, ratios: Mapping[str, float], min_one: bool = True) -> dict[str, int]:
    """Split an integer ``total`` by ``ratios`` using largest-remainder rounding.

    Args:
        total: Number of items to distribute.
        ratios: ``{split: ratio}``; ratios should sum to 1.
        min_one: Give every split with a positive ratio at least one item
            when ``total`` allows it (taken from the largest split).

    Returns:
        ``{split: count}`` summing exactly to ``total``, in ``SPLITS`` order.
    """
    names = [s for s in SPLITS if s in ratios] + [s for s in ratios if s not in SPLITS]
    raw = {s: total * float(ratios[s]) for s in names}
    counts = {s: int(np.floor(raw[s])) for s in names}
    remainder = total - sum(counts.values())
    # Largest fractional part first; ties broken by split order.
    for s in sorted(names, key=lambda s: (-(raw[s] - counts[s]), names.index(s)))[:remainder]:
        counts[s] += 1

    positive = [s for s in names if ratios[s] > 0]
    if min_one and total >= len(positive):
        for s in positive:
            if counts[s] == 0:
                donor = max(names, key=lambda d: counts[d])
                counts[donor] -= 1
                counts[s] += 1
    return counts


def split_groups(groups: Iterable[str], ratios: Mapping[str, float], rng: np.random.Generator) -> dict[str, str]:
    """Assign each group (song) to a split.

    Groups are sorted, shuffled with ``rng`` and then cut into consecutive
    blocks sized by :func:`allocate_counts`.

    Returns:
        ``{group: split}``.
    """
    unique = sorted(set(groups))
    order = [unique[i] for i in rng.permutation(len(unique))]
    counts = allocate_counts(len(unique), ratios)
    assignment: dict[str, str] = {}
    start = 0
    for split, count in counts.items():
        for group in order[start : start + count]:
            assignment[group] = split
        start += count
    return assignment


def find_leakage(items: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    """Return keys that appear in more than one split.

    Args:
        items: ``(key, split)`` pairs, e.g. ``(source_song, split)``.

    Returns:
        ``{key: sorted list of splits}`` for every leaking key (empty if none).
    """
    splits_by_key: dict[str, set[str]] = defaultdict(set)
    for key, split in items:
        splits_by_key[key].add(split)
    return {k: sorted(v) for k, v in splits_by_key.items() if len(v) > 1}
