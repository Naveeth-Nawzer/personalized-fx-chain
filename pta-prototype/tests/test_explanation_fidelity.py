"""Fidelity and evidence-grounding tests for DecisionTrace and ExplanationEngine.

Verifies that explanations are strictly faithful to recorded system states,
actual effect decisions, parameter deltas, and DAPN rendering outcomes,
preventing hallucinated or unjustified acoustic claims.
"""
import pytest
from pta.core.decision_trace import DecisionTrace
from pta.core.explanation_engine import (
    explain_preference_update,
    explain_generation,
    generate_explanation,
)
from pta.integration.grrl_adapter import GRRLAdapter


def test_preference_explanation_matches_record():
    """Verify that explain_preference_update reports exact record values and evidence tag."""
    result = explain_preference_update(
        command="MORE_REVERB",
        dimension="reverb",
        before=0.0,
        after=0.3,
        event_id="evt_001",
    )
    assert result["type"] == "preference_update"
    assert result["command"] == "MORE_REVERB"
    assert result["dimension"] == "reverb"
    assert result["before"] == 0.0
    assert result["after"] == 0.3
    assert abs(result["delta"] - 0.3) < 1e-6
    assert result["event_id"] == "evt_001"
    assert result["evidence"] == "preference_update_record"
    assert "reverb preference changed from 0.00 to 0.30" in result["text"]


def test_failed_rendering_does_not_claim_audio_outcome():
    """Verify that failed DAPN renders do not claim acoustic or perceptual changes."""
    trace = DecisionTrace(
        session_id="session-001",
        generation_id="generation-001",
        rendering_status="failed",
    )
    result = explain_generation(trace)
    assert result["status"] == "failed"
    assert "cannot yet be explained" in result["text"]
    assert result["evidence"] == "generation_status"


def test_not_run_rendering_does_not_claim_audio_outcome():
    """Verify that unrendered traces do not make audio claims."""
    trace = DecisionTrace(
        session_id="session-001",
        generation_id="generation-002",
        rendering_status="not_run",
    )
    result = explain_generation(trace)
    assert result["status"] == "not_run"
    assert "has not been rendered yet" in result["text"]
    assert result["evidence"] == "generation_status"


def test_generation_explains_actual_effect_chain():
    """Verify explanation cites actual selected effects in order."""
    trace = DecisionTrace(
        session_id="session-001",
        generation_id="generation-003",
        rendering_status="success",
        selected_chain=[
            {"effect": "EQ", "params": {"gain": 0.6}},
            {"effect": "Compressor", "params": {"threshold": 0.4}},
            {"effect": "Reverb", "params": {"room_size": 0.8}},
        ],
    )
    result = explain_generation(trace)
    assert result["status"] == "success"
    assert result["selected_chain"] == ["EQ", "Compressor", "Reverb"]
    assert "Generated chain: EQ -> Compressor -> Reverb." in result["text"]


def test_parameter_level_explanation_matches_actual_diff():
    """Verify parameter diff between previous and current chains is accurately reported."""
    trace = DecisionTrace(
        session_id="session-001",
        generation_id="generation-004",
        rendering_status="success",
        previous_chain=[
            {"effect": "Reverb", "params": {"room_size": 0.50, "wet_mix": 0.25}},
            {"effect": "EQ", "params": {"gain": 0.50}},
        ],
        selected_chain=[
            {"effect": "Reverb", "params": {"room_size": 0.75, "wet_mix": 0.35}},
            {"effect": "EQ", "params": {"gain": 0.50}},
        ],
    )
    diffs = trace.get_parameter_diff()
    assert len(diffs) == 3
    # Check room_size
    rs_diff = next(d for d in diffs if d["parameter"] == "room_size")
    assert rs_diff["before"] == 0.50
    assert rs_diff["after"] == 0.75
    assert abs(rs_diff["delta"] - 0.25) < 1e-6

    result = explain_generation(trace)
    assert "Reverb.room_size: 0.50 -> 0.75" in result["text"]
    assert "Reverb.wet_mix: 0.25 -> 0.35" in result["text"]


def test_effect_prohibited_by_hard_constraint():
    """Verify that prohibited effect constraints are reported as enforced or violated."""
    trace = DecisionTrace(
        session_id="session-001",
        generation_id="generation-005",
        rendering_status="success",
        guidance_sent={
            "constraints": {"prohibited_effects": ["Distortion", "Flanger"]}
        },
        selected_chain=[
            {"effect": "EQ", "params": {}},
            {"effect": "Reverb", "params": {}},
        ],
    )
    result = explain_generation(trace)
    assert "Prohibited effects enforced: ['Distortion', 'Flanger']." in result["text"]
    assert "WARNING" not in result["text"]


def test_effect_prohibited_violation_flagged():
    """Verify that if agent erroneously selected a prohibited effect, a warning is raised."""
    trace = DecisionTrace(
        session_id="session-001",
        generation_id="generation-006",
        rendering_status="success",
        guidance_sent={
            "constraints": {"prohibited_effects": ["Distortion"]}
        },
        selected_chain=[
            {"effect": "Distortion", "params": {}},
        ],
    )
    result = explain_generation(trace)
    assert "WARNING: Prohibited effects violated by agent: ['Distortion']." in result["text"]


def test_unsupported_width_request_fidelity():
    """Verify mono rendering constraint for width requests is truthfully surfaced."""
    trace = DecisionTrace(
        session_id="session-001",
        generation_id="generation-007",
        feedback_command="WIDER",
        rendering_status="success",
        unsupported_requests=["Mono rendering pipeline: spatial widening unrendered"],
        selected_chain=[{"effect": "EQ", "params": {}}],
    )
    result = explain_generation(trace)
    assert "active rendering pipeline is mono" in result["text"]
    assert "Unsupported pipeline capabilities noted" in result["text"]


def test_successful_render_without_audio_measurements_avoids_unfounded_claims():
    """Verify that if audio rendered but no objective metrics exist, no acoustic claims are made."""
    trace = DecisionTrace(
        session_id="session-001",
        generation_id="generation-008",
        rendering_status="success",
        audio_ref="audio/gen_008.wav",
        audio_metrics={},
        selected_chain=[{"effect": "EQ", "params": {}}],
    )
    result = explain_generation(trace)
    assert "Audio rendered successfully, but objective acoustic measurements are not available" in result["text"]
    # Ensure no fabricated claims like warmer or brighter
    assert "warmer" not in result["text"].lower()
    assert "brighter" not in result["text"].lower()


def test_successful_render_with_audio_measurements_reports_metrics():
    """Verify that when objective metrics exist, they are factually cited."""
    trace = DecisionTrace(
        session_id="session-001",
        generation_id="generation-009",
        rendering_status="success",
        audio_ref="audio/gen_009.wav",
        audio_metrics={"snr_db": 21.4, "thd_pct": 0.02},
        selected_chain=[{"effect": "Compressor", "params": {}}],
    )
    result = explain_generation(trace)
    assert "Measured audio metrics: snr_db=21.40, thd_pct=0.02." in result["text"]


def test_trace_validation_rejects_invalid_state():
    """Verify DecisionTrace rejects empty IDs or invalid statuses."""
    with pytest.raises(ValueError, match="session_id"):
        DecisionTrace(session_id="", generation_id="g1")

    with pytest.raises(ValueError, match="generation_id"):
        DecisionTrace(session_id="s1", generation_id="")

    with pytest.raises(ValueError, match="Invalid rendering_status"):
        DecisionTrace(session_id="s1", generation_id="g1", rendering_status="unknown_status")

    with pytest.raises(ValueError, match="must be in"):
        DecisionTrace(
            session_id="s1",
            generation_id="g1",
            preference_before={"reverb": 2.5},
        )


def test_satisfied_terminal_explanation():
    """Verify SATISFIED terminal event explanation closes session with no regenerations."""
    trace = DecisionTrace(
        session_id="session-001",
        generation_id="generation-010",
        feedback_command="SATISFIED",
        rendering_status="success",
        selected_chain=[{"effect": "Reverb", "params": {}}],
    )
    result = explain_generation(trace)
    assert "User marked generation as satisfactory." in result["text"]
    assert "no further parameter updates" in result["text"]


def test_adapter_generates_and_links_decision_trace():
    """Verify end-to-end that GRRLAdapter produces an auditable DecisionTrace."""
    class MockEnv:
        def reset(self, **kwargs):
            return [0.0]*32
        def step(self, action):
            return [0.0]*32, 0.9, True, {
                "chain": [{"effect": "EQ", "params": {"gain": 0.7}}],
                "metrics": {"snr": 19.5},
                "audio_ref": "audio/sample_100.wav",
            }

    adapter = GRRLAdapter(env_factory=lambda: MockEnv())
    prefs = {
        "brightness": 0.2,
        "compression": 0.0,
        "reverb": 0.4,
        "delay": 0.0,
        "saturation": 0.0,
        "width": 0.0,
        "warmth": 0.0,
    }
    res = adapter.generate(
        dry_audio="dry.wav",
        target_audio="target.wav",
        preferences=prefs,
        constraints={"prohibited_effects": ["Distortion"], "parameter_bounds": {}},
        session_id="sess_100",
        previous_chain=[{"effect": "EQ", "params": {"gain": 0.5}}],
        feedback_command="BRIGHTER",
        feedback_event_id="evt_100",
    )

    assert res["status"] == "success"
    trace = res["decision_trace"]
    assert isinstance(trace, DecisionTrace)
    assert trace.session_id == "sess_100"
    assert trace.feedback_command == "BRIGHTER"
    assert trace.feedback_event_id == "evt_100"
    assert trace.rendering_status == "success"
    assert trace.audio_metrics["snr"] == 19.5

    # Now verify explanation generated from this trace
    expl = explain_generation(trace)
    assert "Generated chain: EQ." in expl["text"]
    assert "EQ.gain: 0.50 -> 0.70" in expl["text"]
    assert "Prohibited effects enforced: ['Distortion']." in expl["text"]
    assert "snr=19.50" in expl["text"]


def test_parameter_requested_vs_not_applied():
    """Verify that if an effect was requested but not chosen by the agent, explanation is truthful."""
    trace = DecisionTrace(
        session_id="session-001",
        generation_id="generation-011",
        feedback_command="MORE_REVERB",
        rendering_status="success",
        # Agent decided to generate only EQ, not Reverb
        selected_chain=[{"effect": "EQ", "params": {"gain": 0.5}}],
    )
    result = explain_generation(trace)
    assert result["status"] == "success"
    assert result["selected_chain"] == ["EQ"]
    assert "Generated chain: EQ." in result["text"]
    assert "Reverb." not in result["text"]  # Does not claim reverb was adjusted or in chain


def test_feedback_not_yet_applied_to_generation():
    """Verify that feedback recorded before generation cannot be explained as a generation outcome."""
    # When no generation trace exists, only preference update is explainable
    update_expl = explain_preference_update(
        command="MORE_COMPRESSION",
        dimension="compression",
        before=0.0,
        after=0.25,
        event_id="evt_pending",
    )
    assert update_expl["type"] == "preference_update"
    assert update_expl["evidence"] == "preference_update_record"
    assert "compression preference changed from 0.00 to 0.25" in update_expl["text"]
    assert "chain" not in update_expl

