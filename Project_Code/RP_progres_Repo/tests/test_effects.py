import numpy as np
import pytest
from scipy.signal import freqz

from src.effects import (
    CANONICAL_ORDER,
    EffectChain,
    EffectError,
    EffectStep,
    canonical_order,
    create_effect,
)
from src.effects.eq import peaking_biquad
from src.effects.pedalboard_backend import pedalboard_available
from src.utils.audio import residual_db

SR = 16000
PARAMS = {
    "EQ": {"frequency": 1000.0, "gain": 6.0, "q": 1.0},
    "Compressor": {"threshold": -30.0, "ratio": 4.0, "attack": 5.0, "release": 50.0, "makeup_gain": 3.0},
    "Reverb": {"room_size": 0.5, "damping": 0.5, "wet_mix": 0.3, "dry_mix": 0.7, "width": 1.0},
}


@pytest.mark.parametrize("name", CANONICAL_ORDER)
def test_each_effect_changes_signal_and_keeps_shape(name, tone) -> None:
    effect = create_effect(name)
    out = effect.process(tone, SR, PARAMS[name])
    assert out.shape == tone.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert residual_db(tone, out) > -30.0


@pytest.mark.parametrize("name", CANONICAL_ORDER)
def test_effects_are_deterministic(name, tone) -> None:
    effect = create_effect(name)
    a = effect.process(tone, SR, PARAMS[name])
    b = effect.process(tone, SR, PARAMS[name])
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", CANONICAL_ORDER)
def test_stereo_supported(name, tone) -> None:
    stereo = np.concatenate([tone, 0.5 * tone])
    out = create_effect(name).process(stereo, SR, PARAMS[name])
    assert out.shape == stereo.shape


def test_eq_biquad_gain_at_centre_frequency() -> None:
    b, a = peaking_biquad(1000.0, 6.0, 1.0, SR)
    _, h = freqz(b, a, worN=[1000.0], fs=SR)
    assert 20 * np.log10(abs(h[0])) == pytest.approx(6.0, abs=0.01)
    _, h_far = freqz(b, a, worN=[50.0], fs=SR)
    assert abs(20 * np.log10(abs(h_far[0]))) < 0.5


def test_eq_rejects_frequency_above_nyquist(tone) -> None:
    with pytest.raises(EffectError):
        create_effect("EQ").process(tone, SR, {"frequency": 9000.0, "gain": 3.0, "q": 1.0})


def test_compressor_reduces_dynamic_range(tone) -> None:
    params = dict(PARAMS["Compressor"], makeup_gain=0.0)
    out = create_effect("Compressor").process(tone, SR, params)
    assert np.max(np.abs(out)) < np.max(np.abs(tone))
    # Below threshold nothing happens: a very quiet signal passes unchanged.
    quiet = tone * 1e-3
    np.testing.assert_allclose(create_effect("Compressor").process(quiet, SR, params), quiet, atol=1e-7)


def test_reverb_adds_tail_after_impulse() -> None:
    impulse = np.zeros((1, SR), dtype=np.float32)
    impulse[0, 100] = 1.0
    out = create_effect("Reverb").process(impulse, SR, PARAMS["Reverb"])
    assert np.sum(out[0, 2000:] ** 2) > 0  # energy long after the impulse
    longer = create_effect("Reverb").process(impulse, SR, dict(PARAMS["Reverb"], room_size=1.0))
    assert np.sum(longer[0, 8000:] ** 2) > np.sum(out[0, 8000:] ** 2)


def test_missing_or_unknown_parameters_rejected(tone) -> None:
    with pytest.raises(EffectError):
        create_effect("EQ").process(tone, SR, {"frequency": 1000.0, "gain": 3.0})
    with pytest.raises(EffectError):
        create_effect("EQ").process(tone, SR, {**PARAMS["EQ"], "bogus": 1.0})


def test_unknown_effect_or_backend() -> None:
    with pytest.raises(EffectError):
        create_effect("Distortion")
    with pytest.raises(EffectError):
        create_effect("EQ", backend="nope")


def test_canonical_order() -> None:
    assert canonical_order(["Reverb", "EQ", "Compressor"]) == ["EQ", "Compressor", "Reverb"]
    assert canonical_order(["Reverb", "Compressor"]) == ["Compressor", "Reverb"]
    with pytest.raises(EffectError):
        canonical_order(["EQ", "EQ"])


def _step(name: str) -> EffectStep:
    return EffectStep(create_effect(name), PARAMS[name])


def test_chain_applies_effects_sequentially_in_given_order(tone) -> None:
    chain = EffectChain([_step("EQ"), _step("Compressor")])
    expected = _step("Compressor").effect.process(_step("EQ").effect.process(tone, SR, PARAMS["EQ"]), SR,
                                                  PARAMS["Compressor"])
    np.testing.assert_array_equal(chain.process(tone, SR), expected)
    assert chain.effect_names == ["EQ", "Compressor"]


def test_chain_order_matters(tone) -> None:
    eq_then_comp = EffectChain([_step("EQ"), _step("Compressor")]).process(tone, SR)
    comp_then_eq = EffectChain([_step("Compressor"), _step("EQ")]).process(tone, SR)
    assert not np.allclose(eq_then_comp, comp_then_eq)


def test_full_canonical_chain_and_dict(tone) -> None:
    chain = EffectChain([_step(n) for n in canonical_order(["Reverb", "Compressor", "EQ"])])
    assert chain.effect_names == ["EQ", "Compressor", "Reverb"]
    out = chain.process(tone, SR)
    assert out.shape == tone.shape
    as_dict = chain.to_dict()
    assert [s["effect"] for s in as_dict["chain"]] == ["EQ", "Compressor", "Reverb"]
    assert as_dict["chain"][0]["parameters"] == PARAMS["EQ"]


def test_empty_chain_rejected() -> None:
    with pytest.raises(ValueError):
        EffectChain([])


@pytest.mark.skipif(not pedalboard_available(), reason="pedalboard backend not available on this machine")
@pytest.mark.parametrize("name", CANONICAL_ORDER)
def test_pedalboard_backend(name, tone) -> None:
    out = create_effect(name, backend="pedalboard").process(tone, SR, PARAMS[name])
    assert out.shape == tone.shape
    assert residual_db(tone, out) > -40.0
