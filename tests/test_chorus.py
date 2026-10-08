"""tests/test_chorus.py — test Chorus effect."""
import pytest
import torch
from dapn.effects.chorus import Chorus


@pytest.fixture
def chorus():
    return Chorus()


def _audio(T=4096, B=None):
    torch.manual_seed(42)
    if B is None:
        return torch.randn(1, T) * 0.5
    return torch.randn(B, 1, T) * 0.5


def _params(rate=0.5, depth=0.5, delay=0.5, mix=0.5):
    return {
        "rate": torch.tensor(rate),
        "depth": torch.tensor(depth),
        "delay": torch.tensor(delay),
        "mix": torch.tensor(mix),
    }


def test_output_shape_2d(chorus):
    audio = _audio()
    out = chorus(audio, _params())
    assert out.shape == audio.shape


def test_output_shape_3d(chorus):
    audio = _audio(B=2)
    out = chorus(audio, _params())
    assert out.shape == audio.shape


def test_output_is_finite(chorus):
    audio = _audio()
    out = chorus(audio, _params())
    assert torch.isfinite(out).all()


def test_dry_passthrough_when_mix_is_zero(chorus):
    audio = _audio()
    out = chorus(audio, _params(mix=0.0))
    assert torch.allclose(out, audio, atol=1e-5)


def test_chorus_modifies_signal_when_mixed(chorus):
    audio = _audio()
    out = chorus(audio, _params(mix=0.7))
    assert not torch.allclose(out, audio, atol=1e-3)


def test_gradient_rate(chorus):
    audio = _audio()
    p = _params()
    p["rate"] = torch.tensor(0.5, requires_grad=True)
    out = chorus(audio, p)
    out.pow(2).mean().backward()
    assert p["rate"].grad is not None
    assert torch.isfinite(p["rate"].grad)


def test_gradient_depth(chorus):
    audio = _audio()
    p = _params()
    p["depth"] = torch.tensor(0.5, requires_grad=True)
    out = chorus(audio, p)
    out.pow(2).mean().backward()
    assert p["depth"].grad is not None
    assert torch.isfinite(p["depth"].grad)


def test_gradient_delay(chorus):
    audio = _audio()
    p = _params()
    p["delay"] = torch.tensor(0.5, requires_grad=True)
    out = chorus(audio, p)
    out.pow(2).mean().backward()
    assert p["delay"].grad is not None
    assert torch.isfinite(p["delay"].grad)
