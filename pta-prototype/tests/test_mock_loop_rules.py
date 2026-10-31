import numpy as np
import pytest

from pta.core.guidance_builder import build_guidance_message
from pta.core.session_manager import SessionManager
from pta.integration.mock_loop_rules import (
    record_regeneration_count,
    should_regenerate,
)
from pta.storage.json_store import JSONPreferenceStore


def test_normal_feedback_with_regenerate_action_regenerates():
    message = {
        "event_type": "PREFERENCE_UPDATED",
        "feedback_command": "MORE_REVERB",
        "action": "REGENERATE",
    }
    assert should_regenerate(message) is True


def test_satisfied_command_never_regenerates_even_if_action_is_wrong():
    message = {
        "event_type": "PREFERENCE_UPDATED",
        "feedback_command": "SATISFIED",
        "action": "REGENERATE",
    }
    assert should_regenerate(message) is False


def test_satisfied_event_never_regenerates():
    message = {
        "event_type": "SESSION_SATISFIED",
        "feedback_command": "SATISFIED",
        "action": "NONE",
    }
    assert should_regenerate(message) is False


def test_none_action_does_not_regenerate():
    assert should_regenerate({"feedback_command": "MORE_DELAY", "action": "NONE"}) is False


def test_object_messages_are_supported():
    class Message:
        event_type = "PREFERENCE_UPDATED"
        feedback_command = "WIDER"
        action = "REGENERATE"

    assert should_regenerate(Message()) is True


def test_count_increments_only_for_actual_generation():
    message = {"feedback_command": "MORE_REVERB", "action": "REGENERATE"}
    assert record_regeneration_count(2, message, generated=True) == 3
    assert record_regeneration_count(2, message, generated=False) == 2


def test_terminal_message_cannot_be_counted_as_generated():
    message = {"feedback_command": "SATISFIED", "action": "REGENERATE"}
    with pytest.raises(ValueError):
        record_regeneration_count(4, message, generated=True)


def test_negative_count_rejected():
    with pytest.raises(ValueError):
        record_regeneration_count(-1, {"action": "NONE"}, generated=False)


def test_regression_vector_is_updated_before_guidance_is_built(tmp_path):
    store = JSONPreferenceStore(base_dir=str(tmp_path))
    mgr = SessionManager("U_TEST", "electric_guitar", store=store)
    
    # 1. Starts neutral
    assert np.allclose(mgr.get_session_vector(), 0.0)
    
    # 2. Submit feedback
    mgr.submit_feedback("MORE_REVERB")
    v1 = mgr.get_session_vector()
    assert v1[2] == 0.3  # reverb is index 2
    
    # 3. Guidance incorporates the updated vector
    guidance = build_guidance_message(
        "U_TEST", mgr.session.session_id, "sample_prev_01", "MORE_REVERB", v1
    )
    assert guidance.z_taste[2] == 0.3
    assert "reverb" in guidance.changed_dimensions


def test_regression_feedback_retains_intended_sample_id():
    prefs = np.zeros(7)
    guidance = build_guidance_message(
        "U001", "S001", "sample_expected_42", "MORE_DELAY", prefs
    )
    assert guidance.sample_id == "sample_expected_42"
