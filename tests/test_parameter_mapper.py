"""tests/test_parameter_mapper.py — test ParameterMapper."""
import pytest
import torch
from common.audio_types import EffectParameterSpec
from common.parameter_mapper import ParameterMapper, build_mappers_for_effect


def make_spec(name="gain", phys_min=-12.0, phys_max=12.0, mapping="linear"):
    return EffectParameterSpec(
        name=name,
        normalized_min=0.0,
        normalized_max=1.0,
        physical_min=phys_min,
        physical_max=phys_max,
        unit="dB",
        mapping_type=mapping,
        required=True,
    )


@pytest.fixture
def gain_mapper():
    return ParameterMapper(make_spec())


# ---------------------------------------------------------------------------
# Linear mapping correctness
# ---------------------------------------------------------------------------

def test_normalized_0_maps_to_physical_min(gain_mapper):
    t = torch.tensor(0.0)
    out = gain_mapper.to_physical(t)
    assert abs(out.item() - (-12.0)) < 1e-5


def test_normalized_1_maps_to_physical_max(gain_mapper):
    t = torch.tensor(1.0)
    out = gain_mapper.to_physical(t)
    assert abs(out.item() - 12.0) < 1e-5


def test_normalized_0_5_maps_to_midpoint(gain_mapper):
    t = torch.tensor(0.5)
    out = gain_mapper.to_physical(t)
    assert abs(out.item() - 0.0) < 1e-5   # midpoint of [-12, 12]


def test_roundtrip_normalized(gain_mapper):
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = torch.tensor(v)
        physical = gain_mapper.to_physical(t)
        recovered = gain_mapper.to_normalized(physical)
        assert abs(recovered.item() - v) < 1e-5, f"Roundtrip failed for {v}"


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

def test_clamps_above_1(gain_mapper):
    t = torch.tensor(1.5)
    out = gain_mapper.to_physical(t)
    assert abs(out.item() - 12.0) < 1e-4   # clamped to physical_max


def test_clamps_below_0(gain_mapper):
    t = torch.tensor(-0.5)
    out = gain_mapper.to_physical(t)
    assert abs(out.item() - (-12.0)) < 1e-4  # clamped to physical_min


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_gradient_preserved_to_physical(gain_mapper):
    t = torch.tensor(0.5, requires_grad=True)
    out = gain_mapper.to_physical(t)
    out.backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad)


def test_gradient_preserved_to_normalized(gain_mapper):
    t = torch.tensor(0.0, requires_grad=True)
    physical = gain_mapper.to_physical(t)
    norm = gain_mapper.to_normalized(physical)
    norm.backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad)


def test_no_gradient_at_clamp_boundary(gain_mapper):
    # Gradient should be 0 outside valid range (clamped region)
    t = torch.tensor(2.0, requires_grad=True)   # above 1.0
    out = gain_mapper.to_physical(t)
    out.backward()
    assert t.grad is not None
    assert t.grad.item() == 0.0   # clamped → gradient = 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_build_mappers_for_effect():
    specs = {
        "gain": make_spec("gain"),
        "q":    make_spec("q", 0.1, 10.0),
    }
    mappers = build_mappers_for_effect(specs)
    assert set(mappers.keys()) == {"gain", "q"}
    assert isinstance(mappers["gain"], ParameterMapper)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_zero_range_raises():
    bad_spec = make_spec(phys_min=5.0, phys_max=5.0)
    with pytest.raises(ValueError, match="non-zero"):
        ParameterMapper(bad_spec)


def test_unsupported_mapping_type_raises():
    bad_spec = EffectParameterSpec(
        name="x", normalized_min=0.0, normalized_max=1.0,
        physical_min=0.0, physical_max=1.0, unit="", mapping_type="sqrt", required=True
    )
    with pytest.raises(ValueError, match="Unsupported mapping_type"):
        ParameterMapper(bad_spec)
