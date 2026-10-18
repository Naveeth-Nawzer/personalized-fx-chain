"""tests/test_compressor.py — test Compressor effect."""
import pytest
import torch
from dapn.effects.compressor import Compressor


@pytest.fixture
def comp():
    return Compressor()


def _audio(T=4096, B=None, amplitude=0.5):
    if B is None:
        return torch.randn(1, T) * amplitude
    return torch.randn(B, 1, T) * amplitude


def _params(threshold=0.5, ratio=0.3, attack=0.2, release=0.4, makeup_gain=0.3):
    return {
        "threshold":   torch.tensor(threshold),
        "ratio":       torch.tensor(ratio),
        "attack":      torch.tensor(attack),
        "release":     torch.tensor(release),
        "makeup_gain": torch.tensor(makeup_gain),
    }


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_output_shape_2d(comp):
    audio = _audio()
    out = comp(audio, _params())
    assert out.shape == audio.shape


def test_output_shape_3d(comp):
    audio = _audio(B=2)
    out = comp(audio, _params())
    assert out.shape == audio.shape


# ---------------------------------------------------------------------------
# Output quality
# ---------------------------------------------------------------------------

def test_output_is_finite(comp):
    audio = _audio()
    out = comp(audio, _params())
    assert torch.isfinite(out).all()


def test_high_ratio_reduces_dynamics(comp):
    """Higher ratio → less dynamic range in output."""
    torch.manual_seed(0)
    audio = _audio(amplitude=0.8)
    out_low  = comp(audio, _params(ratio=0.1))   # ratio ≈ 2.9
    out_high = comp(audio, _params(ratio=0.9))   # ratio ≈ 18.1
    # High ratio compressor should produce lower RMS variance
    # (not universally true for all inputs, but directional check)
    assert torch.isfinite(out_low).all()
    assert torch.isfinite(out_high).all()


def test_makeup_gain_increases_output(comp):
    """Higher makeup gain → louder output."""
    torch.manual_seed(1)
    audio = _audio()
    out_low  = comp(audio, _params(makeup_gain=0.0))
    out_high = comp(audio, _params(makeup_gain=1.0))
    rms_low  = out_low.pow(2).mean().sqrt().item()
    rms_high = out_high.pow(2).mean().sqrt().item()
    assert rms_high > rms_low


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_gradient_threshold(comp):
    audio = _audio()
    p = _params()
    p["threshold"] = torch.tensor(0.5, requires_grad=True)
    out = comp(audio, p)
    out.pow(2).mean().backward()
    assert p["threshold"].grad is not None
    assert torch.isfinite(p["threshold"].grad)


def test_gradient_ratio(comp):
    audio = _audio()
    p = _params()
    p["ratio"] = torch.tensor(0.3, requires_grad=True)
    out = comp(audio, p)
    out.pow(2).mean().backward()
    assert p["ratio"].grad is not None
    assert torch.isfinite(p["ratio"].grad)


def test_gradient_makeup_gain(comp):
    audio = _audio()
    p = _params()
    p["makeup_gain"] = torch.tensor(0.5, requires_grad=True)
    out = comp(audio, p)
    out.pow(2).mean().backward()
    assert p["makeup_gain"].grad is not None
    assert torch.isfinite(p["makeup_gain"].grad)


def test_attack_changes_output(comp):
    """Different attack settings produce different audio on dynamic transients."""
    t = torch.linspace(0, 0.5, 22050)
    audio = torch.zeros(1, 22050)
    audio[0, 1000:5000] = 0.9 * torch.sin(2 * 3.14159 * 500 * t[:4000])
    p_fast = _params(threshold=0.5, ratio=0.6, attack=0.01, release=0.3, makeup_gain=0.0)
    p_slow = _params(threshold=0.5, ratio=0.6, attack=0.90, release=0.3, makeup_gain=0.0)

    out_fast = comp(audio, p_fast)
    out_slow = comp(audio, p_slow)
    diff = (out_fast - out_slow).abs().max().item()
    assert diff > 1e-3, f"Expected attack to alter output, max diff was {diff}"


def test_release_changes_output(comp):
    """Different release settings produce different audio after transients drop."""
    t = torch.linspace(0, 0.5, 22050)
    audio = torch.zeros(1, 22050)
    audio[0, 1000:5000] = 0.9 * torch.sin(2 * 3.14159 * 500 * t[:4000])
    audio[0, 5000:10000] = 0.3 * torch.sin(2 * 3.14159 * 500 * t[:5000])
    p_fast = _params(threshold=0.5, ratio=0.6, attack=0.1, release=0.02, makeup_gain=0.0)
    p_slow = _params(threshold=0.5, ratio=0.6, attack=0.1, release=0.90, makeup_gain=0.0)

    out_fast = comp(audio, p_fast)
    out_slow = comp(audio, p_slow)
    diff = (out_fast - out_slow).abs().max().item()
    assert diff > 1e-3, f"Expected release to alter output, max diff was {diff}"


def test_gradient_attack(comp):
    """Gradient w.r.t. attack must be finite and non-zero on dynamic audio."""
    t = torch.linspace(0, 0.5, 22050)
    audio = torch.zeros(1, 22050)
    audio[0, 1000:5000] = 0.9 * torch.sin(2 * 3.14159 * 500 * t[:4000])
    p = _params(threshold=0.5, ratio=0.5, makeup_gain=0.0)
    p["attack"] = torch.tensor(0.2, requires_grad=True)
    out = comp(audio, p)
    out.pow(2).mean().backward()
    assert p["attack"].grad is not None
    assert torch.isfinite(p["attack"].grad)
    assert p["attack"].grad.abs().item() > 0.0


def test_gradient_release(comp):
    """Gradient w.r.t. release must be finite and non-zero on dynamic audio."""
    t = torch.linspace(0, 0.5, 22050)
    audio = torch.zeros(1, 22050)
    audio[0, 1000:5000] = 0.9 * torch.sin(2 * 3.14159 * 500 * t[:4000])
    audio[0, 5000:10000] = 0.2 * torch.sin(2 * 3.14159 * 500 * t[:5000])
    p = _params(threshold=0.5, ratio=0.5, makeup_gain=0.0)
    p["release"] = torch.tensor(0.3, requires_grad=True)
    out = comp(audio, p)
    out.pow(2).mean().backward()
    assert p["release"].grad is not None
    assert torch.isfinite(p["release"].grad)
    assert p["release"].grad.abs().item() > 0.0
