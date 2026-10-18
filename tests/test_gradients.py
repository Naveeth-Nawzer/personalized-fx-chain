"""tests/test_gradients.py — gradient propagation through individual and chained effects."""
import pytest
import torch
from common.fx_schema import FXChainValidator
from dapn.dapn import DAPN


@pytest.fixture
def dapn():
    return DAPN()


def _audio(T=22050):
    torch.manual_seed(99)
    return torch.randn(1, T)


# ---------------------------------------------------------------------------
# Individual effect gradients via DAPN param_override
# ---------------------------------------------------------------------------

def test_eq_all_parameters_have_gradients(dapn):
    audio = _audio()
    freq = torch.tensor(0.5, requires_grad=True)
    gain = torch.tensor(0.7, requires_grad=True)
    q    = torch.tensor(0.4, requires_grad=True)

    chain_dict = {
        "chain": [{"effect": "EQ", "parameters": {"frequency": 0.5, "gain": 0.7, "q": 0.4}}]
    }
    override = {0: {"frequency": freq, "gain": gain, "q": q}}

    out = dapn.process(audio, chain_dict, param_override=override)
    loss = out.pow(2).mean()
    loss.backward()

    for name, p in [("frequency", freq), ("gain", gain), ("q", q)]:
        assert p.grad is not None, f"EQ parameter '{name}' grad is None"
        assert torch.isfinite(p.grad), f"EQ parameter '{name}' grad is not finite"


def test_compressor_all_parameters_have_gradients(dapn):
    audio = _audio() * 0.8
    thresh = torch.tensor(0.6, requires_grad=True)
    ratio  = torch.tensor(0.5, requires_grad=True)
    att    = torch.tensor(0.3, requires_grad=True)
    rel    = torch.tensor(0.4, requires_grad=True)
    makeup = torch.tensor(0.5, requires_grad=True)

    chain_dict = {
        "chain": [{
            "effect": "Compressor",
            "parameters": {"threshold": 0.6, "ratio": 0.5, "attack": 0.3, "release": 0.4, "makeup_gain": 0.5},
        }]
    }
    override = {0: {
        "threshold": thresh, "ratio": ratio, "attack": att, "release": rel, "makeup_gain": makeup
    }}

    out = dapn.process(audio, chain_dict, param_override=override)
    loss = out.pow(2).mean()
    loss.backward()

    for name, p in [("threshold", thresh), ("ratio", ratio), ("attack", att), ("release", rel), ("makeup_gain", makeup)]:
        assert p.grad is not None, f"Compressor '{name}' grad is None"
        assert torch.isfinite(p.grad), f"Compressor '{name}' grad is not finite"
        assert p.grad.abs().item() > 0.0, f"Compressor '{name}' grad is zero"


def test_delay_all_parameters_have_gradients(dapn):
    audio = _audio()
    dtime = torch.tensor(0.05, requires_grad=True)
    fb    = torch.tensor(0.4,  requires_grad=True)
    wet   = torch.tensor(0.5,  requires_grad=True)
    dry   = torch.tensor(0.5,  requires_grad=True)

    chain_dict = {
        "chain": [{
            "effect": "Delay",
            "parameters": {"delay_time": 0.05, "feedback": 0.4, "wet_mix": 0.5, "dry_mix": 0.5},
        }]
    }
    override = {0: {"delay_time": dtime, "feedback": fb, "wet_mix": wet, "dry_mix": dry}}

    out = dapn.process(audio, chain_dict, param_override=override)
    loss = out.pow(2).mean()
    loss.backward()

    for name, p in [("delay_time", dtime), ("feedback", fb), ("wet_mix", wet), ("dry_mix", dry)]:
        assert p.grad is not None, f"Delay '{name}' grad is None"
        assert torch.isfinite(p.grad), f"Delay '{name}' grad is not finite"


# ---------------------------------------------------------------------------
# Multi-effect gradient chain: EQ -> Compressor -> Delay
# ---------------------------------------------------------------------------

def test_gradient_propagates_through_entire_chain(dapn):
    """Gradients must flow all the way back from output through 3 chained effects."""
    audio = _audio()

    eq_freq   = torch.tensor(0.4,  requires_grad=True)
    eq_gain   = torch.tensor(0.8,  requires_grad=True)
    eq_q      = torch.tensor(0.5,  requires_grad=True)

    cp_thresh = torch.tensor(0.5,  requires_grad=True)
    cp_ratio  = torch.tensor(0.4,  requires_grad=True)
    cp_att    = torch.tensor(0.2,  requires_grad=True)
    cp_rel    = torch.tensor(0.3,  requires_grad=True)
    cp_makeup = torch.tensor(0.3,  requires_grad=True)

    dl_time   = torch.tensor(0.02, requires_grad=True)
    dl_fb     = torch.tensor(0.3,  requires_grad=True)
    dl_wet    = torch.tensor(0.6,  requires_grad=True)
    dl_dry    = torch.tensor(0.4,  requires_grad=True)

    chain_dict = {
        "chain": [
            {"effect": "EQ",         "parameters": {"frequency": 0.4, "gain": 0.8, "q": 0.5}},
            {"effect": "Compressor", "parameters": {"threshold": 0.5, "ratio": 0.4,
                                                    "attack": 0.2, "release": 0.3, "makeup_gain": 0.3}},
            {"effect": "Delay",      "parameters": {"delay_time": 0.02, "feedback": 0.3,
                                                    "wet_mix": 0.6, "dry_mix": 0.4}},
        ]
    }
    override = {
        0: {"frequency": eq_freq, "gain": eq_gain, "q": eq_q},
        1: {"threshold": cp_thresh, "ratio": cp_ratio, "attack": cp_att, "release": cp_rel, "makeup_gain": cp_makeup},
        2: {"delay_time": dl_time, "feedback": dl_fb, "wet_mix": dl_wet, "dry_mix": dl_dry},
    }

    out = dapn.process(audio, chain_dict, param_override=override)

    # Simple target: match a target audio tensor
    target = audio * 0.5
    loss = (out - target).pow(2).mean()
    loss.backward()

    # Check that EVERY effect in the chain received gradients
    effects_and_params = [
        ("EQ frequency",         eq_freq),
        ("EQ gain",              eq_gain),
        ("Compressor threshold", cp_thresh),
        ("Compressor attack",    cp_att),
        ("Compressor release",   cp_rel),
        ("Compressor makeup",    cp_makeup),
        ("Delay wet_mix",        dl_wet),
        ("Delay dry_mix",        dl_dry),
    ]

    for label, param in effects_and_params:
        assert param.grad is not None, f"Gradient for {label} is None!"
        assert torch.isfinite(param.grad), f"Gradient for {label} is not finite!"
        # In a 3-effect chain with active processing, gradients should be non-zero
        assert param.grad.abs().item() > 0.0, f"Gradient for {label} is exactly 0.0!"
