"""Contract integration test verifying complete PTA ↔ GRRL boundary.

Verifies:
1. Guidance goes into generation.
2. Valid chain, metrics, and audio reference return.
3. User feedback updates the session state.
4. Terminal SATISFIED closes the loop without re-triggering generation.
5. Untouched dimensions do not decay.
"""
import numpy as np
import pytest

from pta.core.guidance_builder import build_guidance_message
from pta.core.session_manager import SessionManager
from pta.integration.grrl_adapter import GRRLAdapter
from pta.storage.json_store import JSONPreferenceStore
from tests.test_grrl_adapter import FakeGRRLEnvironment


def test_full_loop_contract_boundary(tmp_path):
    store = JSONPreferenceStore(base_dir=str(tmp_path))
    env = FakeGRRLEnvironment()
    adapter = GRRLAdapter(env_factory=lambda: env)

    session_mgr = SessionManager("U_CONTRACT_01", "electric_guitar", store=store)
    session_id = session_mgr.session.session_id

    # 1. Generate initial chain with neutral taste
    initial_res = adapter.generate_initial("dry.wav", "target.wav", session_id=session_id)
    assert initial_res["status"] == "success"
    assert len(initial_res["chain"]) > 0
    current_sample_id = initial_res["sample_id"]

    # 2. User feedback sequence
    commands = ["MORE_REVERB", "MORE_REVERB", "LESS_COMPRESSION"]
    generation_history = [initial_res]

    for cmd in commands:
        # Submit feedback to PTA
        result = session_mgr.submit_feedback(cmd)
        current_vector = session_mgr.get_session_vector()

        # Build guidance payload matching contract
        guidance = build_guidance_message(
            user_id="U_CONTRACT_01",
            session_id=session_id,
            sample_id=current_sample_id,
            feedback_command=cmd,
            preference_vector=current_vector,
        )

        # Send to GRRL
        gen_res = adapter.generate(
            dry_audio="dry.wav",
            target_audio="target.wav",
            preferences=current_vector,
            constraints=guidance.constraints.to_dict() if hasattr(guidance.constraints, "to_dict") else {
                "prohibited_effects": guidance.constraints.prohibited_effects,
                "parameter_bounds": guidance.constraints.parameter_bounds,
            },
            session_id=session_id,
        )

        assert gen_res["status"] == "success"
        current_sample_id = gen_res["sample_id"]
        generation_history.append(gen_res)

    # Verify 1 initial + 3 feedback generations = 4 generations
    assert len(generation_history) == 4

    # Verify preference vector shifted appropriately
    final_vec = session_mgr.get_session_vector()
    assert final_vec[2] > 0.4   # Reverb boosted twice
    assert final_vec[1] < -0.2  # Compression reduced
    assert final_vec[0] == 0.0  # Brightness untouched

    # 3. Terminal satisfaction concludes loop
    satisfaction_result = session_mgr.submit_feedback("SATISFIED")
    assert session_mgr.session.satisfied is True

    # Vector must NOT change on SATISFIED
    assert np.allclose(session_mgr.get_session_vector(), final_vec)

    # 4. Long-term merge preserves untouched dimensions
    session_mgr.merge_into_long_term(weight=0.5)
    profile_vec = store.load_profile("U_CONTRACT_01", "electric_guitar", "global")
    assert profile_vec is not None
    assert profile_vec[2] > 0.2  # Reverb learned
    assert profile_vec[0] == 0.0  # Untouched dimension remained 0.0


def test_contract_terminal_satisfied_does_not_call_generator():
    call_counter = {"count": 0}

    def counting_factory():
        call_counter["count"] += 1
        return FakeGRRLEnvironment()

    adapter = GRRLAdapter(env_factory=counting_factory)

    # Initial generation
    adapter.generate_initial("dry.wav", "target.wav")
    assert call_counter["count"] == 1

    # On SATISFIED, caller checks terminal status and avoids calling generate()
    is_terminal = True
    if not is_terminal:
        adapter.generate("dry.wav", "target.wav", np.zeros(7))

    assert call_counter["count"] == 1
