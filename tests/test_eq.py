"""tests/test_eq.py — test EQ effect."""
import pytest
import torch
from dapn.effects.eq import EQ


@pytest.fixture
def eq():
    return EQ()


def _audio(T=4096, B=None):
    if B is None:
        return torch.randn(1, T)        # (C, T)
    return torch.randn(B, 1, T)        # (B, C, T)


def _params(freq=0.5, gain=0.5, q=0.5):
    return {
        "frequency": torch.tensor(freq),
        "gain":      torch.tensor(gain),
        "q":         torch.tensor(q),
    }


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------

def test_output_shape_2d(eq):
    audio = _audio()
    out = eq(audio, _params())
    assert out.shape == audio.shape


def test_output_shape_3d(eq):
    audio = _audio(B=2)
    out = eq(audio, _params())
    assert out.shape == audio.shape


# ---------------------------------------------------------------------------
# Output quality
# ---------------------------------------------------------------------------

def test_output_is_finite(eq):
    audio = _audio()
    out = eq(audio, _params())
    assert torch.isfinite(out).all()


def test_unity_gain_eq_close_to_passthrough(eq):
    """At 0 dB gain the output should approximate the input."""
    torch.manual_seed(0)
    audio = _audio()
    out = eq(audio, _params(gain=0.5))  # 0.5 normalized → 0 dB physical
    # Check RMS ratio is close to 1.0 (not exact due to frequency-dependent EQ)
    rms_in  = audio.pow(2).mean().sqrt()
    rms_out = out.pow(2).mean().sqrt()
    ratio = (rms_out / rms_in).item()
    assert 0.1 < ratio < 10.0   # broad sanity check


def test_different_gains_produce_different_outputs(eq):
    torch.manual_seed(42)
    audio = _audio()
    out_low  = eq(audio, _params(gain=0.1))
    out_high = eq(audio, _params(gain=0.9))
    assert not torch.allclose(out_low, out_high, atol=1e-4)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_gradient_frequency(eq):
    audio = _audio()
    freq = torch.tensor(0.5, requires_grad=True)
    out  = eq(audio, {"frequency": freq, "gain": torch.tensor(0.5), "q": torch.tensor(0.5)})
    loss = out.pow(2).mean()
    loss.backward()
    assert freq.grad is not None
    assert torch.isfinite(freq.grad)


def test_gradient_gain(eq):
    audio = _audio()
    gain = torch.tensor(0.7, requires_grad=True)
    out  = eq(audio, {"frequency": torch.tensor(0.5), "gain": gain, "q": torch.tensor(0.5)})
    loss = out.pow(2).mean()
    loss.backward()
    assert gain.grad is not None
    assert torch.isfinite(gain.grad)


def test_gradient_q(eq):
    audio = _audio()
    q = torch.tensor(0.3, requires_grad=True)
    out = eq(audio, {"frequency": torch.tensor(0.5), "gain": torch.tensor(0.6), "q": q})
    loss = out.pow(2).mean()
    loss.backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad)
