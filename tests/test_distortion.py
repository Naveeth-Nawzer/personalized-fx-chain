"""tests/test_distortion.py — test Distortion effect."""
import pytest
import torch
from dapn.effects.distortion import Distortion


@pytest.fixture
def dist():
    return Distortion()


def _audio(T=4096, B=None):
    torch.manual_seed(42)
    if B is None:
        return torch.randn(1, T) * 0.5
    return torch.randn(B, 1, T) * 0.5


def _params(drive=0.5, tone=0.5, mix=0.5, output_gain=0.5):
    return {
        "drive": torch.tensor(drive),
        "tone": torch.tensor(tone),
        "mix": torch.tensor(mix),
        "output_gain": torch.tensor(output_gain),
    }


def test_output_shape_2d(dist):
    audio = _audio()
    out = dist(audio, _params())
    assert out.shape == audio.shape


def test_output_shape_3d(dist):
    audio = _audio(B=2)
    out = dist(audio, _params())
    assert out.shape == audio.shape


def test_output_is_finite(dist):
    audio = _audio()
    out = dist(audio, _params())
    assert torch.isfinite(out).all()


def test_dry_passthrough_when_mix_is_zero(dist):
    audio = _audio()
    out = dist(audio, _params(mix=0.0))
    assert torch.allclose(out, audio, atol=1e-5)


def test_distortion_modifies_signal_when_mixed(dist):
    audio = _audio()
    out = dist(audio, _params(drive=0.8, mix=1.0))
    assert not torch.allclose(out, audio, atol=1e-3)


def test_gradient_drive(dist):
    audio = _audio()
    p = _params()
    p["drive"] = torch.tensor(0.5, requires_grad=True)
    out = dist(audio, p)
    out.pow(2).mean().backward()
    assert p["drive"].grad is not None
    assert torch.isfinite(p["drive"].grad)
    assert p["drive"].grad.abs().item() > 0.0


def test_gradient_tone(dist):
    audio = _audio()
    p = _params()
    p["tone"] = torch.tensor(0.5, requires_grad=True)
    out = dist(audio, p)
    out.pow(2).mean().backward()
    assert p["tone"].grad is not None
    assert torch.isfinite(p["tone"].grad)


def test_gradient_output_gain(dist):
    audio = _audio()
    p = _params()
    p["output_gain"] = torch.tensor(0.5, requires_grad=True)
    out = dist(audio, p)
    out.pow(2).mean().backward()
    assert p["output_gain"].grad is not None
    assert torch.isfinite(p["output_gain"].grad)
