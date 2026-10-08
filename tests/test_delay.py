"""tests/test_delay.py — test Delay effect."""
import pytest
import torch
from dapn.effects.delay import Delay


@pytest.fixture
def delay():
    return Delay()


def _audio(T=8192, B=None):
    torch.manual_seed(7)
    if B is None:
        return torch.randn(1, T)
    return torch.randn(B, 1, T)


def _params(delay_time=0.05, feedback=0.3, wet_mix=0.5, dry_mix=0.5):
    return {
        "delay_time": torch.tensor(delay_time),
        "feedback":   torch.tensor(feedback),
        "wet_mix":    torch.tensor(wet_mix),
        "dry_mix":    torch.tensor(dry_mix),
    }


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_output_shape_2d(delay):
    audio = _audio()
    out = delay(audio, _params())
    assert out.shape == audio.shape


def test_output_shape_3d(delay):
    audio = _audio(B=2)
    out = delay(audio, _params())
    assert out.shape == audio.shape


# ---------------------------------------------------------------------------
# Output quality
# ---------------------------------------------------------------------------

def test_output_is_finite(delay):
    audio = _audio()
    out = delay(audio, _params())
    assert torch.isfinite(out).all()


def test_dry_only_approximates_input(delay):
    """With wet_mix=0, dry_mix=1, output ≈ input."""
    audio = _audio()
    out = delay(audio, _params(wet_mix=0.0, dry_mix=1.0, feedback=0.0))
    assert torch.allclose(out, audio, atol=1e-5)


def test_wet_only_is_not_equal_to_input(delay):
    """With wet_mix=1, dry_mix=0, output should differ from input."""
    audio = _audio()
    out = delay(audio, _params(wet_mix=1.0, dry_mix=0.0, delay_time=0.1))
    assert not torch.allclose(out, audio, atol=1e-4)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_gradient_delay_time(delay):
    audio = _audio()
    p = _params()
    p["delay_time"] = torch.tensor(0.05, requires_grad=True)
    out = delay(audio, p)
    out.pow(2).mean().backward()
    assert p["delay_time"].grad is not None
    assert torch.isfinite(p["delay_time"].grad)


def test_gradient_wet_mix(delay):
    audio = _audio()
    p = _params()
    p["wet_mix"] = torch.tensor(0.5, requires_grad=True)
    out = delay(audio, p)
    out.pow(2).mean().backward()
    assert p["wet_mix"].grad is not None
    assert torch.isfinite(p["wet_mix"].grad)


def test_gradient_feedback(delay):
    audio = _audio()
    p = _params()
    p["feedback"] = torch.tensor(0.3, requires_grad=True)
    out = delay(audio, p)
    out.pow(2).mean().backward()
    assert p["feedback"].grad is not None
    assert torch.isfinite(p["feedback"].grad)
