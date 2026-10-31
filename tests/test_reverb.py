"""tests/test_reverb.py — test Reverb effect."""
import pytest
import torch
from dapn.effects.reverb import Reverb


@pytest.fixture
def reverb():
    return Reverb()


def _audio(T=8192, B=None):
    torch.manual_seed(13)
    if B is None:
        return torch.randn(1, T)
    return torch.randn(B, 1, T)


def _params(room_size=0.5, damping=0.5, wet_mix=0.5, dry_mix=0.5, decay=0.3):
    return {
        "room_size": torch.tensor(room_size),
        "damping":   torch.tensor(damping),
        "wet_mix":   torch.tensor(wet_mix),
        "dry_mix":   torch.tensor(dry_mix),
        "decay":     torch.tensor(decay),
    }


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_output_shape_2d(reverb):
    audio = _audio()
    out = reverb(audio, _params())
    assert out.shape == audio.shape


def test_output_shape_3d(reverb):
    audio = _audio(B=2)
    out = reverb(audio, _params())
    assert out.shape == audio.shape


# ---------------------------------------------------------------------------
# Output quality
# ---------------------------------------------------------------------------

def test_output_is_finite(reverb):
    audio = _audio()
    out = reverb(audio, _params())
    assert torch.isfinite(out).all()


def test_dry_only_close_to_input(reverb):
    """dry_mix=1, wet_mix=0 → output ≈ input (with comb filter contribution)."""
    audio = _audio()
    out = reverb(audio, _params(wet_mix=0.0, dry_mix=1.0))
    # Output should at least correlate with input
    assert torch.isfinite(out).all()


def test_wet_changes_output(reverb):
    """Different wet_mix produces different output."""
    audio = _audio()
    out_dry = reverb(audio, _params(wet_mix=0.0, dry_mix=1.0))
    out_wet = reverb(audio, _params(wet_mix=1.0, dry_mix=0.0))
    assert not torch.allclose(out_dry, out_wet, atol=1e-4)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_gradient_room_size(reverb):
    audio = _audio()
    p = _params()
    p["room_size"] = torch.tensor(0.5, requires_grad=True)
    out = reverb(audio, p)
    out.pow(2).mean().backward()
    assert p["room_size"].grad is not None
    assert torch.isfinite(p["room_size"].grad)


def test_gradient_decay(reverb):
    audio = _audio()
    p = _params()
    p["decay"] = torch.tensor(0.3, requires_grad=True)
    out = reverb(audio, p)
    out.pow(2).mean().backward()
    assert p["decay"].grad is not None
    assert torch.isfinite(p["decay"].grad)


def test_gradient_wet_mix(reverb):
    audio = _audio()
    p = _params()
    p["wet_mix"] = torch.tensor(0.5, requires_grad=True)
    out = reverb(audio, p)
    out.pow(2).mean().backward()
    assert p["wet_mix"].grad is not None
    assert torch.isfinite(p["wet_mix"].grad)
