"""tests/test_flanger.py — test Flanger effect."""
import pytest
import torch
from dapn.effects.flanger import Flanger


@pytest.fixture
def flanger():
    return Flanger()


def _audio(T=4096, B=None):
    torch.manual_seed(42)
    if B is None:
        return torch.randn(1, T) * 0.5
    return torch.randn(B, 1, T) * 0.5


def _params(rate=0.5, depth=0.5, feedback=0.4, mix=0.5):
    return {
        "rate": torch.tensor(rate),
        "depth": torch.tensor(depth),
        "feedback": torch.tensor(feedback),
        "mix": torch.tensor(mix),
    }


def test_output_shape_2d(flanger):
    audio = _audio()
    out = flanger(audio, _params())
    assert out.shape == audio.shape


def test_output_shape_3d(flanger):
    audio = _audio(B=2)
    out = flanger(audio, _params())
    assert out.shape == audio.shape


def test_output_is_finite(flanger):
    audio = _audio()
    out = flanger(audio, _params())
    assert torch.isfinite(out).all()


def test_dry_passthrough_when_mix_is_zero(flanger):
    audio = _audio()
    out = flanger(audio, _params(mix=0.0))
    assert torch.allclose(out, audio, atol=1e-5)


def test_flanger_modifies_signal_when_mixed(flanger):
    audio = _audio()
    out = flanger(audio, _params(mix=0.6))
    assert not torch.allclose(out, audio, atol=1e-3)


def test_gradient_rate(flanger):
    audio = _audio()
    p = _params()
    p["rate"] = torch.tensor(0.5, requires_grad=True)
    out = flanger(audio, p)
    out.pow(2).mean().backward()
    assert p["rate"].grad is not None
    assert torch.isfinite(p["rate"].grad)


def test_gradient_depth(flanger):
    audio = _audio()
    p = _params()
    p["depth"] = torch.tensor(0.5, requires_grad=True)
    out = flanger(audio, p)
    out.pow(2).mean().backward()
    assert p["depth"].grad is not None
    assert torch.isfinite(p["depth"].grad)


def test_gradient_feedback(flanger):
    audio = _audio()
    p = _params()
    p["feedback"] = torch.tensor(0.4, requires_grad=True)
    out = flanger(audio, p)
    out.pow(2).mean().backward()
    assert p["feedback"].grad is not None
    assert torch.isfinite(p["feedback"].grad)
