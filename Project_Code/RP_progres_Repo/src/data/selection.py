"""Reproducible, category-stratified selection of source files per split.

Procedure (all randomness comes from the supplied ``rng``):

1. Songs are already assigned to splits (:func:`src.data.splitting.split_groups`).
2. Each split gets a file quota (e.g. 70/15/15 of 100).
3. Within a split, the quota is divided evenly across categories. Categories
   are visited round-robin; within a category, candidates are taken in a
   seeded random order, preferring songs used least so far (diversity).
4. Every candidate is checked by ``validator`` (readable, long enough, not
   silent). Rejected files are recorded with the reason.
5. Category shortfalls are filled from other categories (logged).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from src.data.discovery import AudioFileInfo
from src.data.splitting import allocate_counts
from src.utils.config import SPLITS

logger = logging.getLogger(__name__)

#: Returns ``None`` if the file is usable, otherwise a human-readable rejection reason.
Validator = Callable[[AudioFileInfo], "str | None"]


class SelectionError(RuntimeError):
    """Raised when not enough valid source files can be selected."""


@dataclass(frozen=True)
class SelectedSource:
    """A source file chosen for generation."""

    source_id: str
    source_index: int
    file: AudioFileInfo
    split: str


@dataclass(frozen=True)
class SkippedFile:
    rel_path: str
    song: str
    category: str
    reason: str


@dataclass
class SelectionResult:
    selected: list[SelectedSource]
    skipped: list[SkippedFile]
    warnings: list[str] = field(default_factory=list)


def allocate_category_quotas(quota: int, categories: Sequence[str], rng: np.random.Generator) -> dict[str, int]:
    """Divide ``quota`` as evenly as possible over ``categories``; leftovers go to random categories."""
    cats = sorted(categories)
    if not cats:
        return {}
    base, extra = divmod(quota, len(cats))
    quotas = {c: base for c in cats}
    for idx in rng.permutation(len(cats))[:extra]:
        quotas[cats[idx]] += 1
    return quotas


def select_sources(
    files: Sequence[AudioFileInfo],
    song_splits: Mapping[str, str],
    num_files: int,
    ratios: Mapping[str, float],
    rng: np.random.Generator,
    validator: Validator,
    stratify: bool = True,
) -> SelectionResult:
    """Select ``num_files`` valid source files, split by song and stratified by category.

    Raises:
        SelectionError: if fewer than ``num_files`` valid files are available.
    """
    split_quotas = allocate_counts(num_files, ratios)
    skipped: list[SkippedFile] = []
    warnings: list[str] = []
    chosen: dict[str, list[AudioFileInfo]] = {}

    for split in SPLITS:
        quota = split_quotas.get(split, 0)
        candidates = [f for f in files if song_splits.get(f.song) == split]
        picked = _select_in_split(candidates, quota, rng, validator, stratify, skipped, warnings, split)
        if len(picked) < quota:
            warnings.append(f"[{split}] only {len(picked)} valid file(s) available for a quota of {quota}")
        chosen[split] = picked

    total = sum(len(v) for v in chosen.values())
    if total < num_files:
        detail = ", ".join(f"{s}={len(chosen[s])}/{split_quotas.get(s, 0)}" for s in SPLITS)
        raise SelectionError(
            f"Only {total} valid source file(s) could be selected but {num_files} were requested ({detail}). "
            f"{len(skipped)} candidate(s) were rejected. Reduce --num-source-files, relax the preprocessing "
            "settings (duration / silence threshold), or add more audio."
        )

    selected: list[SelectedSource] = []
    for split in SPLITS:
        for info in sorted(chosen[split], key=lambda f: f.rel_path):
            idx = len(selected)
            selected.append(SelectedSource(f"src_{idx + 1:04d}", idx, info, split))
    for message in warnings:
        logger.warning(message)
    return SelectionResult(selected, skipped, warnings)


def _select_in_split(
    candidates: Sequence[AudioFileInfo],
    quota: int,
    rng: np.random.Generator,
    validator: Validator,
    stratify: bool,
    skipped: list[SkippedFile],
    warnings: list[str],
    split: str,
) -> list[AudioFileInfo]:
    if quota <= 0 or not candidates:
        return []

    def key(f: AudioFileInfo) -> str:
        return f.category if stratify else "__all__"

    queues: dict[str, list[AudioFileInfo]] = {}
    for f in candidates:
        queues.setdefault(key(f), []).append(f)
    for cat, queue in queues.items():
        queues[cat] = [queue[i] for i in rng.permutation(len(queue))]

    cat_quotas = allocate_category_quotas(quota, list(queues), rng)
    picked: list[AudioFileInfo] = []
    per_cat: Counter[str] = Counter()
    song_use: Counter[str] = Counter()

    def take_valid(cat: str) -> AudioFileInfo | None:
        queue = queues[cat]
        while queue:
            # Prefer the least-used song; ties keep the seeded queue order.
            pos = min(range(len(queue)), key=lambda i: (song_use[queue[i].song], i))
            candidate = queue.pop(pos)
            try:
                reason = validator(candidate)
            except Exception as exc:  # noqa: BLE001 — record and continue with the next file
                reason = f"validation error: {exc}"
            if reason is None:
                return candidate
            logger.info("Skipping %s: %s", candidate.rel_path, reason)
            skipped.append(SkippedFile(candidate.rel_path, candidate.song, candidate.category, reason))
        return None

    def accept(cat: str, f: AudioFileInfo) -> None:
        picked.append(f)
        per_cat[cat] += 1
        song_use[f.song] += 1

    # Pass 1: fill category quotas round-robin.
    active = sorted(c for c in cat_quotas if cat_quotas[c] > 0)
    while active:
        for cat in list(active):
            if per_cat[cat] >= cat_quotas[cat]:
                active.remove(cat)
                continue
            f = take_valid(cat)
            if f is None:
                active.remove(cat)
                continue
            accept(cat, f)

    # Pass 2: fill any shortfall from the least-represented categories that still have files.
    shortfall = quota - len(picked)
    if shortfall > 0 and stratify:
        warnings.append(
            f"[{split}] category quotas could not be met ({dict(per_cat)} vs {cat_quotas}); "
            f"filling {shortfall} slot(s) from other categories"
        )
    while len(picked) < quota:
        options = sorted((c for c in queues if queues[c]), key=lambda c: (per_cat[c], c))
        if not options:
            break
        for cat in options:
            f = take_valid(cat)
            if f is not None:
                accept(cat, f)
                break
    return picked
