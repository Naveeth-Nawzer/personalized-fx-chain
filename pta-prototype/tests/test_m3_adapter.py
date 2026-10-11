import numpy as np
import pytest
from pta.core.guidance_builder import (
    build_guidance_message, WIDTH_UNSUPPORTED,
)
from pta.integration.m3_adapter import M3Adapter, MockM3Receiver
from pta.models.guidance import (
    GuidanceMessage, RewardGuidance, ConstraintMask, Z_TASTE_DIM,
)


PREF = np.array([0.0, -0.3, 0.51, 0.15, 0.0, 0.15, 0.0])


# --- Basic acknowledgement ---

def test_mock_acknowledges_preference_update():
    receiver = MockM3Receiver()
    adapter = M3Adapter(receiver)
    msg = build_guidance_message("U001", "S001", "sample_A", "MORE_REVERB", PREF)
    response = adapter.send(msg)
    assert response["status"] == "ACKNOWLEDGED"


def test_mock_closes_on_satisfied():
    receiver = MockM3Receiver()
    adapter = M3Adapter(receiver)
    msg = build_guidance_message("U001", "S001", "sample_A", "SATISFIED", PREF)
    response = adapter.send(msg)
    assert response["status"] == "SESSION_CLOSED"


def test_sample_id_preserved():
    receiver = MockM3Receiver()
    adapter = M3Adapter(receiver)
    msg = build_guidance_message("U001", "S001", "sample_XYZ", "MORE_REVERB", PREF)
    adapter.send(msg)
    assert receiver.received[-1].sample_id == "sample_XYZ"


# --- Message construction ---

def test_all_fields_have_defaults():
    """Verify that GuidanceMessage can be constructed with minimal args."""
    msg = GuidanceMessage(user_id="U001", session_id="S001")
    assert msg.sample_id == ""
    assert msg.z_taste == [0.0] * Z_TASTE_DIM
    assert msg.reward.r_pta == 0.0
    assert msg.constraints.parameter_bounds == {}


def test_invalid_message_rejected():
    # Valid construction, but validation must fail on empty user_id
    msg = GuidanceMessage(user_id="", session_id="S001")
    with pytest.raises(ValueError, match="user_id and session_id required"):
        M3Adapter().send(msg)


# --- Reward semantics (neutral default + pluggable custom fn) ---

def test_reward_is_neutral_for_unconfirmed_commands():
    """r_pta must be 0 with zero confidence until validated."""
    msg = build_guidance_message("U001", "S001", "sample_A", "MORE_REVERB", PREF)
    assert msg.reward.r_pta == 0.0
    assert msg.reward.confidence == 0.0


def test_reward_is_positive_for_satisfied():
    msg = build_guidance_message("U001", "S001", "sample_A", "SATISFIED", PREF)
    assert msg.reward.r_pta == 1.0
    assert msg.reward.confidence == 1.0


def test_custom_reward_fn_override():
    def custom_fn(pref, cmd):
        return -0.5
    msg = build_guidance_message("U001", "S001", "sample_A", "MORE_REVERB", PREF, reward_fn=custom_fn)
    assert msg.reward.r_pta == -0.5
    assert msg.reward.confidence == 1.0


# --- Width explicitly unsupported ---

def test_width_produces_no_constraint():
    prefs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0])  # strong width
    msg = build_guidance_message("U001", "S001", "sample_A", "WIDER", prefs)
    assert "width" in msg.changed_dimensions
    # No constraint generated for width — explicit design choice
    assert "Chorus" not in msg.constraints.parameter_bounds
    assert "Flanger" not in msg.constraints.parameter_bounds
    assert msg.constraints.parameter_bounds == {}


def test_width_documented_in_unsupported_list():
    assert "width" in WIDTH_UNSUPPORTED
    assert isinstance(WIDTH_UNSUPPORTED["width"], str)


# --- Constraint shape validation ---

def test_constraint_bounds_are_two_element_lists():
    prefs = np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0])
    msg = build_guidance_message("U001", "S001", "sample_A", "MORE_REVERB", prefs)
    for effect, params in msg.constraints.parameter_bounds.items():
        for param, bounds in params.items():
            assert isinstance(bounds, list)
            assert len(bounds) == 2
            assert 0.0 <= bounds[0] < bounds[1] <= 1.0


def test_constraint_effects_are_m3_supported():
    from pta.models.guidance import SUPPORTED_EFFECTS
    prefs = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    msg = build_guidance_message("U001", "S001", "sample_A", "MORE_REVERB", prefs)
    for effect in msg.constraints.parameter_bounds.keys():
        assert effect in SUPPORTED_EFFECTS


# --- Vector validation ---

def test_nan_in_z_taste_rejected():
    msg = GuidanceMessage(
        user_id="U001", session_id="S001",
        z_taste=[0.0] * (Z_TASTE_DIM - 1) + [float("nan")],
    )
    with pytest.raises(ValueError):
        msg.validate()


def test_inf_in_z_taste_rejected():
    msg = GuidanceMessage(
        user_id="U001", session_id="S001",
        z_taste=[0.0] * (Z_TASTE_DIM - 1) + [float("inf")],
    )
    with pytest.raises(ValueError):
        msg.validate()


def test_invalid_reward_value_rejected():
    msg = GuidanceMessage(user_id="U001", session_id="S001")
    msg.reward.r_pta = 5.0
    with pytest.raises(ValueError):
        msg.validate()


def test_invalid_confidence_rejected():
    msg = GuidanceMessage(user_id="U001", session_id="S001")
    msg.reward.confidence = 1.5
    with pytest.raises(ValueError):
        msg.validate()


# --- Command ↔ dimension consistency ---

def test_command_maps_to_expected_dimension():
    mapping = {
        "MORE_REVERB": "reverb",
        "LESS_COMPRESSION": "compression",
        "WIDER": "width",
        "BRIGHTER": "brightness",
    }
    for cmd, expected_dim in mapping.items():
        msg = build_guidance_message("U001", "S001", "sample_A", cmd, PREF)
        assert msg.changed_dimensions == [expected_dim], \
            f"{cmd} should map to {expected_dim}"


def test_satisfied_has_no_changed_dimensions():
    msg = build_guidance_message("U001", "S001", "sample_A", "SATISFIED", PREF)
    assert msg.changed_dimensions == []


def test_adapter_records_all_messages():
    receiver = MockM3Receiver()
    adapter = M3Adapter(receiver)
    for cmd in ["MORE_REVERB", "LESS_COMPRESSION", "WIDER", "SATISFIED"]:
        msg = build_guidance_message("U001", "S001", f"sample_{cmd}", cmd, PREF)
        adapter.send(msg)
    assert len(receiver.received) == 4


def test_json_serialization_of_full_message():
    msg = build_guidance_message("U001", "S001", "sample_A", "MORE_REVERB", PREF)
    d = msg.to_dict()
    assert isinstance(d["z_taste"], list)
    assert isinstance(d["reward"], dict)
    assert isinstance(d["constraints"], dict)
    assert d["schema_version"] == "1.0"
