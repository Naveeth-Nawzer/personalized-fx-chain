"""tests/test_fx_schema.py — test FX chain schema validation."""
import pytest
from common.fx_schema import FXChainValidator, FXChainSpec, EffectStep


@pytest.fixture
def validator():
    return FXChainValidator()


def _eq_step(freq=0.5, gain=0.7, q=0.4):
    return {"effect": "EQ", "parameters": {"frequency": freq, "gain": gain, "q": q}}


def _comp_step():
    return {
        "effect": "Compressor",
        "parameters": {
            "threshold": 0.6, "ratio": 0.3, "attack": 0.2,
            "release": 0.4, "makeup_gain": 0.3,
        },
    }


def _delay_step():
    return {
        "effect": "Delay",
        "parameters": {"delay_time": 0.1, "feedback": 0.3, "wet_mix": 0.5, "dry_mix": 0.7},
    }


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_single_effect_chain(validator):
    spec = validator.validate({"chain": [_eq_step()]})
    assert isinstance(spec, FXChainSpec)
    assert len(spec) == 1
    assert spec.effect_names() == ["EQ"]


def test_multi_effect_chain(validator):
    spec = validator.validate({"chain": [_eq_step(), _comp_step(), _delay_step()]})
    assert spec.effect_names() == ["EQ", "Compressor", "Delay"]


def test_effect_step_parameters_preserved(validator):
    spec = validator.validate({"chain": [_eq_step(freq=0.25, gain=0.5, q=0.8)]})
    params = spec.steps[0].parameters
    assert abs(params["frequency"] - 0.25) < 1e-6
    assert abs(params["gain"] - 0.5)      < 1e-6
    assert abs(params["q"] - 0.8)         < 1e-6


# ---------------------------------------------------------------------------
# Rejection tests
# ---------------------------------------------------------------------------

def test_missing_chain_key_raises(validator):
    with pytest.raises(ValueError, match="'chain' key"):
        validator.validate({"steps": []})


def test_unknown_effect_raises(validator):
    with pytest.raises(ValueError, match="unknown effect"):
        validator.validate({"chain": [{"effect": "Reverberator", "parameters": {}}]})


def test_unknown_parameter_raises(validator):
    bad = {"effect": "EQ", "parameters": {"frequency": 0.5, "gain": 0.5, "q": 0.5, "volume": 0.5}}
    with pytest.raises(ValueError, match="unknown parameter"):
        validator.validate({"chain": [bad]})


def test_parameter_above_1_raises(validator):
    with pytest.raises(ValueError, match="outside normalized range"):
        validator.validate({"chain": [_eq_step(freq=1.5)]})


def test_parameter_below_0_raises(validator):
    with pytest.raises(ValueError, match="outside normalized range"):
        validator.validate({"chain": [_eq_step(gain=-0.1)]})


def test_missing_required_parameter_raises(validator):
    bad = {"effect": "EQ", "parameters": {"frequency": 0.5, "gain": 0.5}}  # missing q
    with pytest.raises(ValueError, match="required parameter 'q' is missing"):
        validator.validate({"chain": [bad]})


def test_stop_in_chain_raises(validator):
    with pytest.raises(ValueError, match="STOP"):
        validator.validate({"chain": [{"effect": "STOP", "parameters": {}}]})


def test_chain_too_long_raises(validator):
    # max is 8, put 9
    chain = [_eq_step() if i == 0 else _comp_step() if i == 1 else _delay_step()
             for i in range(9)]
    # Make all effects unique by using only valid distinct effects
    chain_dict = {
        "chain": [
            {"effect": "EQ",         "parameters": {"frequency": 0.5, "gain": 0.5, "q": 0.5}},
            {"effect": "Compressor", "parameters": {"threshold": 0.5, "ratio": 0.5, "attack": 0.5, "release": 0.5, "makeup_gain": 0.5}},
            {"effect": "Delay",      "parameters": {"delay_time": 0.5, "feedback": 0.5, "wet_mix": 0.5, "dry_mix": 0.5}},
            {"effect": "Reverb",     "parameters": {"room_size": 0.5, "damping": 0.5, "wet_mix": 0.5, "dry_mix": 0.5, "decay": 0.5}},
            {"effect": "Distortion", "parameters": {"drive": 0.5, "tone": 0.5, "mix": 0.5, "output_gain": 0.5}},
            {"effect": "Chorus",     "parameters": {"rate": 0.5, "depth": 0.5, "delay": 0.5, "mix": 0.5}},
            {"effect": "Flanger",    "parameters": {"rate": 0.5, "depth": 0.5, "feedback": 0.5, "mix": 0.5}},
            {"effect": "EQ",         "parameters": {"frequency": 0.5, "gain": 0.5, "q": 0.5}},  # repeat
            {"effect": "Compressor", "parameters": {"threshold": 0.5, "ratio": 0.5, "attack": 0.5, "release": 0.5, "makeup_gain": 0.5}},
        ]
    }
    with pytest.raises(ValueError):
        validator.validate(chain_dict)


def test_repeated_effect_raises(validator):
    with pytest.raises(ValueError, match="repeated"):
        validator.validate({"chain": [_eq_step(), _eq_step()]})


def test_chain_too_short_raises(validator):
    with pytest.raises(ValueError, match="too short"):
        validator.validate({"chain": []})
