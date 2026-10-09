import numpy as np
import pytest

from src.data.splitting import allocate_counts, find_leakage, split_groups

RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


@pytest.mark.parametrize(
    "total, expected",
    [(100, {"train": 70, "validation": 15, "test": 15}),
     (3, {"train": 1, "validation": 1, "test": 1}),
     (10, {"train": 7, "validation": 2, "test": 1}),
     (1, {"train": 1, "validation": 0, "test": 0})],
)
def test_allocate_counts(total: int, expected: dict) -> None:
    counts = allocate_counts(total, RATIOS)
    assert sum(counts.values()) == total
    assert counts == expected


def test_split_groups_is_reproducible_and_complete() -> None:
    songs = [f"song_{i:03d}" for i in range(100)]
    a = split_groups(songs, RATIOS, np.random.default_rng(42))
    b = split_groups(list(reversed(songs)), RATIOS, np.random.default_rng(42))
    c = split_groups(songs, RATIOS, np.random.default_rng(7))
    assert a == b  # input order does not matter
    assert a != c  # seed matters
    assert set(a) == set(songs)
    counts = {s: sum(1 for v in a.values() if v == s) for s in RATIOS}
    assert counts == {"train": 70, "validation": 15, "test": 15}


def test_split_groups_duplicates_collapse() -> None:
    groups = ["a", "a", "b", "c", "c", "c"]
    assignment = split_groups(groups, RATIOS, np.random.default_rng(0))
    assert set(assignment) == {"a", "b", "c"}
    assert set(assignment.values()) == {"train", "validation", "test"}


def test_find_leakage() -> None:
    assert find_leakage([("a", "train"), ("a", "train"), ("b", "test")]) == {}
    assert find_leakage([("a", "train"), ("a", "test")]) == {"a": ["test", "train"]}
