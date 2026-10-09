import pytest

from src.data.labels import (
    LABEL_NAMES,
    combination_name,
    effect_combinations,
    label_columns,
    label_vector,
)
from src.effects.base import EffectError

EXPECTED = {
    ("EQ",): [1, 0, 0],
    ("Compressor",): [0, 1, 0],
    ("Reverb",): [0, 0, 1],
    ("EQ", "Compressor"): [1, 1, 0],
    ("EQ", "Reverb"): [1, 0, 1],
    ("Compressor", "Reverb"): [0, 1, 1],
    ("EQ", "Compressor", "Reverb"): [1, 1, 1],
}


def test_label_order() -> None:
    assert LABEL_NAMES == ("EQ", "Compressor", "Reverb")


def test_seven_non_empty_combinations_in_canonical_order() -> None:
    combos = effect_combinations()
    assert combos == list(EXPECTED)
    assert len(combos) == 7
    assert all(len(c) > 0 for c in combos)


@pytest.mark.parametrize("chain, expected", list(EXPECTED.items()))
def test_label_vectors(chain, expected) -> None:
    assert label_vector(chain) == expected
    assert list(label_columns(chain).values()) == expected


def test_labels_are_multi_label_not_one_hot() -> None:
    vectors = [label_vector(c) for c in effect_combinations()]
    assert any(sum(v) > 1 for v in vectors)
    assert all(len(v) == 3 for v in vectors)


def test_label_ignores_order_given() -> None:
    # Phase 1 does not distinguish EQ->Compressor from Compressor->EQ.
    assert label_vector(["Compressor", "EQ"]) == label_vector(["EQ", "Compressor"])
    assert combination_name(["Reverb", "EQ"]) == "EQ+Reverb"


def test_subset_of_effects() -> None:
    assert effect_combinations(["Reverb", "EQ"]) == [("EQ",), ("Reverb",), ("EQ", "Reverb")]


def test_invalid_chains_rejected() -> None:
    with pytest.raises(EffectError):
        label_vector(["EQ", "EQ"])
    with pytest.raises(EffectError):
        label_vector(["Distortion"])


def test_label_does_not_depend_on_source_category() -> None:
    # The label function only sees effect names; vocals+EQ and drums+EQ get the same target.
    vocals_eq = {"source_category": "vocals", "effect_chain": ["EQ"]}
    drums_eq = {"source_category": "drums", "effect_chain": ["EQ"]}
    assert label_vector(vocals_eq["effect_chain"]) == label_vector(drums_eq["effect_chain"]) == [1, 0, 0]
