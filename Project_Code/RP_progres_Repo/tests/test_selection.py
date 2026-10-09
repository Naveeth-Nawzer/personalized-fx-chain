from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.data.discovery import AudioFileInfo
from src.data.selection import SelectionError, allocate_category_quotas, select_sources
from src.data.splitting import split_groups

RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
CATEGORIES = ("bass", "drums", "mixture", "other", "vocals")


def fake_files(num_songs: int = 100, categories=CATEGORIES) -> list[AudioFileInfo]:
    return [
        AudioFileInfo(Path(f"/x/Song_{i:03d}/{c}.wav"), f"Song_{i:03d}/{c}.wav", f"Song_{i:03d}", c, ".wav")
        for i in range(num_songs) for c in categories
    ]


def always_valid(_: AudioFileInfo) -> None:
    return None


def run(files, n=100, seed=42, validator=always_valid, stratify=True):
    song_splits = split_groups((f.song for f in files), RATIOS, np.random.default_rng(seed))
    return select_sources(files, song_splits, n, RATIOS, np.random.default_rng(seed + 1), validator, stratify)


def test_selects_exact_split_quotas_and_balanced_categories() -> None:
    result = run(fake_files())
    assert len(result.selected) == 100
    per_split = Counter(s.split for s in result.selected)
    assert per_split == {"train": 70, "validation": 15, "test": 15}
    # 70 / 5 = 14 and 15 / 5 = 3 per category, so overall each category gets exactly 20.
    assert Counter(s.file.category for s in result.selected) == {c: 20 for c in CATEGORIES}
    for split in RATIOS:
        cats = Counter(s.file.category for s in result.selected if s.split == split)
        assert max(cats.values()) - min(cats.values()) <= 1


def test_no_song_appears_in_two_splits() -> None:
    result = run(fake_files())
    splits_by_song: dict[str, set] = {}
    for s in result.selected:
        splits_by_song.setdefault(s.file.song, set()).add(s.split)
    assert all(len(v) == 1 for v in splits_by_song.values())


def test_prefers_distinct_songs() -> None:
    result = run(fake_files())
    # 100 songs available for 100 files: song diversity should be high.
    assert len({s.file.song for s in result.selected}) >= 90


def test_reproducible_with_seed() -> None:
    files = fake_files()
    a = [(s.source_id, s.file.rel_path, s.split) for s in run(files, seed=42).selected]
    b = [(s.source_id, s.file.rel_path, s.split) for s in run(files, seed=42).selected]
    c = [(s.source_id, s.file.rel_path, s.split) for s in run(files, seed=1).selected]
    assert a == b
    assert a != c


def test_source_ids_are_sequential_in_split_order() -> None:
    result = run(fake_files(), n=20)
    assert [s.source_id for s in result.selected] == [f"src_{i:04d}" for i in range(1, 21)]
    assert [s.source_index for s in result.selected] == list(range(20))
    order = [s.split for s in result.selected]
    assert order == sorted(order, key=["train", "validation", "test"].index)


def test_invalid_files_are_skipped_with_reason() -> None:
    def reject_vocals(f: AudioFileInfo):
        return "silent" if f.category == "vocals" else None

    result = run(fake_files(), n=50, validator=reject_vocals)
    assert len(result.selected) == 50
    assert all(s.file.category != "vocals" for s in result.selected)
    assert result.skipped and all(s.reason == "silent" for s in result.skipped)
    assert any("category quotas" in w for w in result.warnings)


def test_validator_exceptions_are_recorded() -> None:
    def broken(f: AudioFileInfo):
        if f.category == "bass":
            raise OSError("corrupt header")
        return None

    result = run(fake_files(), n=20, validator=broken)
    assert len(result.selected) == 20
    assert any("corrupt header" in s.reason for s in result.skipped)


def test_not_enough_files_raises() -> None:
    # 2 songs cannot fill three splits: both go to train (5 files each), so only 8 of the
    # 8 train slots are filled and validation/test stay empty.
    with pytest.raises(SelectionError, match=r"Only 8 valid source file.*validation=0/2, test=0/1"):
        run(fake_files(num_songs=2), n=11)


def test_uniform_mode_without_stratification() -> None:
    result = run(fake_files(), n=30, stratify=False)
    assert len(result.selected) == 30


def test_allocate_category_quotas() -> None:
    rng = np.random.default_rng(0)
    assert allocate_category_quotas(10, ["a", "b"], rng) == {"a": 5, "b": 5}
    q = allocate_category_quotas(7, ["a", "b", "c"], rng)
    assert sum(q.values()) == 7 and sorted(q.values()) == [2, 2, 3]
    assert allocate_category_quotas(5, [], rng) == {}
