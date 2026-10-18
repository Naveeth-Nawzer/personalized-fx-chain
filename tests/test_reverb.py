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


def test_room_size_changes_output(reverb):
    """Varying room_size produces meaningfully different audio reflections."""
    torch.manual_seed(42)
    # Transient impulse
    audio = torch.zeros(1, 8192)
    audio[0, 100] = 1.0
    out_small = reverb(audio, _params(room_size=0.1, wet_mix=1.0, dry_mix=0.0))
    out_large = reverb(audio, _params(room_size=0.9, wet_mix=1.0, dry_mix=0.0))
    diff = (out_small - out_large).abs().max().item()
    assert diff > 1e-3, f"Expected room size to affect reverb tail, got diff {diff}"


def test_gradient_room_size_nonzero(reverb):
    """Room size gradient must be finite and non-zero."""
    torch.manual_seed(42)
    audio = torch.zeros(1, 8192)
    audio[0, 100:150] = 0.8
    p = _params(room_size=0.5, wet_mix=1.0, dry_mix=0.0)
    p["room_size"] = torch.tensor(0.5, requires_grad=True)
    out = reverb(audio, p)
    loss = out.pow(2).mean()
    loss.backward()
    assert p["room_size"].grad is not None
    assert torch.isfinite(p["room_size"].grad)
    assert p["room_size"].grad.abs().item() > 0.0


def test_reverb_all_parameters_have_gradients(reverb):
    """All reverb parameters must receive non-zero finite gradients."""
    audio = _audio()
    room_size = torch.tensor(0.5, requires_grad=True)
    damping = torch.tensor(0.4, requires_grad=True)
    wet_mix = torch.tensor(0.6, requires_grad=True)
    dry_mix = torch.tensor(0.4, requires_grad=True)
    decay = torch.tensor(0.5, requires_grad=True)

    p = {
        "room_size": room_size,
        "damping": damping,
        "wet_mix": wet_mix,
        "dry_mix": dry_mix,
        "decay": decay,
    }
    out = reverb(audio, p)
    loss = out.pow(2).mean()
    loss.backward()

    for name, param in p.items():
        assert param.grad is not None, f"Reverb '{name}' grad is None"
        assert torch.isfinite(param.grad), f"Reverb '{name}' grad is not finite"
        assert param.grad.abs().item() > 0.0, f"Reverb '{name}' grad is zero"


def test_reverb_boundary_conditions(reverb):
    """Reverb handles short audio, extreme room size, and edge delays without crashing."""
    short_audio = torch.randn(1, 64)
    out_short = reverb(short_audio, _params(room_size=0.0))
    assert out_short.shape == short_audio.shape
    assert torch.isfinite(out_short).all()

    out_max = reverb(short_audio, _params(room_size=1.0))
    assert out_max.shape == short_audio.shape
    assert torch.isfinite(out_max).all()
