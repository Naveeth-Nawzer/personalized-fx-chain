import numpy as np
import pytest
from pta.core.guidance_builder import (
    build_guidance_message, project_taste_vector,
    compute_reward_guidance, build_constraints,
)
from pta.models.guidance import Z_TASTE_DIM


SAMPLE_PREF = np.array([0.0, -0.3, 0.51, 0.15, 0.0, 0.15, 0.0])


def test_z_taste_projection_shape():
    z = project_taste_vector(SAMPLE_PREF)
    assert len(z) == Z_TASTE_DIM


def test_z_taste_values_in_range():
    z = project_taste_vector(SAMPLE_PREF)
    assert all(-1.0 <= v <= 1.0 for v in z)


def test_z_taste_first_seven_match_preference():
    z = project_taste_vector(SAMPLE_PREF)
    assert z[:7] == SAMPLE_PREF.tolist()


def test_reward_positive_on_satisfied():
    r = compute_reward_guidance(SAMPLE_PREF, "SATISFIED")
    assert r.r_pta == 1.0
    assert r.confidence == 1.0


def test_reward_neutral_by_default_on_change_request():
    r = compute_reward_guidance(SAMPLE_PREF, "MORE_REVERB")
    assert r.r_pta == 0.0
    assert r.confidence == 0.0


def test_reward_custom_fn():
    def custom_fn(pref, cmd):
        return -0.25
    r = compute_reward_guidance(SAMPLE_PREF, "MORE_REVERB", reward_fn=custom_fn)
    assert r.r_pta == -0.25
    assert r.confidence == 1.0


def test_constraints_from_strong_positive_preference():
    prefs = np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0])
    c = build_constraints(prefs)
    assert "Reverb" in c.parameter_bounds
    assert c.parameter_bounds["Reverb"]["wet_mix"] == [0.4, 0.9]


def test_constraints_from_strong_negative_preference():
    prefs = np.array([0.0, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0])
    c = build_constraints(prefs)
    assert "Compressor" in c.parameter_bounds
    assert c.parameter_bounds["Compressor"]["ratio"] == [0.1, 0.3]


def test_no_constraints_for_weak_preference():
    prefs = np.array([0.1, -0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
    c = build_constraints(prefs)
    assert c.parameter_bounds == {}


def test_build_full_message_preference_updated():
    msg = build_guidance_message(
        "U001", "S001", "sample_00482", "MORE_REVERB", SAMPLE_PREF
    )
    assert msg.event_type == "PREFERENCE_UPDATED"
    assert msg.action == "REGENERATE"
    assert msg.sample_id == "sample_00482"
    assert len(msg.z_taste) == Z_TASTE_DIM
    assert msg.reward.r_pta == 0.0
    assert msg.reward.confidence == 0.0
    assert "Reverb" in msg.constraints.parameter_bounds


def test_build_full_message_satisfied():
    msg = build_guidance_message(
        "U001", "S001", "sample_00482", "SATISFIED", SAMPLE_PREF
    )
    assert msg.event_type == "SESSION_SATISFIED"
    assert msg.action == "NONE"
    assert msg.reward.r_pta == 1.0
    assert msg.reward.confidence == 1.0


def test_invalid_z_taste_dimension_rejected():
    from pta.models.guidance import GuidanceMessage
    msg = GuidanceMessage(user_id="U001", session_id="S001",
                          z_taste=[0.0] * 8)   # wrong length
    with pytest.raises(ValueError):
        msg.validate()


def test_serialization_roundtrip():
    msg = build_guidance_message(
        "U001", "S001", "sample_00482", "MORE_REVERB", SAMPLE_PREF
    )
    d = msg.to_dict()
    assert d["schema_version"] == "1.0"
    assert d["sample_id"] == "sample_00482"
    assert isinstance(d["z_taste"], list)
    assert isinstance(d["reward"]["r_pta"], float)
