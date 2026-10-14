"""tests/test_fx_chain.py — test FXChain processor and DAPN wrapper."""
import pytest
import torch
from common.fx_schema import FXChainValidator
from dapn.fx_chain import FXChain
from dapn.dapn import DAPN


@pytest.fixture
def validator():
    return FXChainValidator()


@pytest.fixture
def dapn():
    return DAPN()


def _make_chain_dict(*effects):
    param_defaults = {
        "EQ":         {"frequency": 0.5, "gain": 0.7, "q": 0.4},
        "Compressor": {"threshold": 0.6, "ratio": 0.3, "attack": 0.2, "release": 0.4, "makeup_gain": 0.3},
        "Delay":      {"delay_time": 0.05, "feedback": 0.3, "wet_mix": 0.5, "dry_mix": 0.5},
        "Reverb":     {"room_size": 0.5, "damping": 0.5, "wet_mix": 0.5, "dry_mix": 0.5, "decay": 0.3},
    }
    return {
        "chain": [
            {"effect": e, "parameters": param_defaults[e]}
            for e in effects
        ]
    }


def _audio(T=4096, B=None):
    torch.manual_seed(42)
    if B is None:
        return torch.randn(1, T)
    return torch.randn(B, 1, T)


# ---------------------------------------------------------------------------
# Chain execution
# ---------------------------------------------------------------------------

def test_fx_chain_single_effect(validator):
    spec = validator.validate(_make_chain_dict("EQ"))
    chain = FXChain(spec)
    audio = _audio()
    out = chain(audio)
    assert out.shape == audio.shape
    assert torch.isfinite(out).all()


def test_fx_chain_multi_effect(validator):
    spec = validator.validate(_make_chain_dict("EQ", "Compressor", "Delay"))
    chain = FXChain(spec)
    audio = _audio()
    out = chain(audio)
    assert out.shape == audio.shape
    assert torch.isfinite(out).all()


def test_dapn_process_happy_path(dapn):
    chain_dict = _make_chain_dict("EQ", "Compressor")
    audio = _audio()
    out = dapn.process(audio, chain_dict)
    assert out.shape == audio.shape
    assert torch.isfinite(out).all()


def test_dapn_forward_is_alias_of_process(dapn):
    chain_dict = _make_chain_dict("EQ")
    audio = _audio()
    out1 = dapn.process(audio, chain_dict)
    out2 = dapn(audio, chain_dict)
    assert torch.allclose(out1, out2)


def test_chain_order_matters(dapn):
    """EQ -> Compressor should produce a different result than Compressor -> EQ."""
    audio = _audio()
    # High gain boost that will trigger compression
    chain_ab = {
        "chain": [
            {"effect": "EQ", "parameters": {"frequency": 0.5, "gain": 0.9, "q": 0.5}},
            {"effect": "Compressor", "parameters": {"threshold": 0.3, "ratio": 0.8,
                                                    "attack": 0.1, "release": 0.3, "makeup_gain": 0.1}},
        ]
    }
    chain_ba = {
        "chain": [
            {"effect": "Compressor", "parameters": {"threshold": 0.3, "ratio": 0.8,
                                                    "attack": 0.1, "release": 0.3, "makeup_gain": 0.1}},
            {"effect": "EQ", "parameters": {"frequency": 0.5, "gain": 0.9, "q": 0.5}},
        ]
    }
    out_ab = dapn.process(audio, chain_ab)
    out_ba = dapn.process(audio, chain_ba)
    # The outputs should not be identical because DSP operations are non-commutative
    assert not torch.allclose(out_ab, out_ba, atol=1e-4)


def test_dapn_rejects_invalid_chain(dapn):
    audio = _audio()
    with pytest.raises(ValueError):
        dapn.process(audio, {"chain": [{"effect": "UnknownFX", "parameters": {}}]})
