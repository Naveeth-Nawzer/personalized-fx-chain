"""Demonstration of Decision-Trace Grounded Explainable AI (XAI).

Traces the 5 stages of personalization:
  1. User feedback
  2. Level 1 (PTA): Preference vector math update
  3. Guidance & Constraints sent to GRRL (16D taste + action masks)
  4. Level 2 (GRRL): Selected FX chain & parameter diffs
  5. Level 3 (Audio Outcome): Truthful DAPN rendering & acoustic metrics

Also demonstrates fidelity edge cases:
  - Mono-pipeline truthfulness (WIDER)
  - Unmeasured rendering (no fabricated acoustic claims)
  - Failed rendering containment
  - Terminal SATISFIED handling (zero regenerations)
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from pta.core.decision_trace import DecisionTrace
from pta.core.explanation_engine import (
    explain_preference_update,
    explain_generation,
)
from pta.integration.grrl_adapter import GRRLAdapter, project_taste


class MockDAPNGymEnv:
    """Mock Gym environment simulating GRRL agent actions and DAPN audio output."""
    def __init__(self, step_idx=0):
        self.step_idx = step_idx
        self.chains = [
            # Initial chain
            [{"effect": "EQ", "params": {"gain": 0.5, "frequency": 0.5}}],
            # Turn 1: Reverb added with room_size, EQ boosted
            [
                {"effect": "EQ", "params": {"gain": 0.7, "frequency": 0.5}},
                {"effect": "Reverb", "params": {"room_size": 0.75, "wet_mix": 0.40}},
            ],
            # Turn 2: Compression added, Reverb parameters increased
            [
                {"effect": "EQ", "params": {"gain": 0.7, "frequency": 0.5}},
                {"effect": "Compressor", "params": {"threshold": 0.30, "ratio": 0.40}},
                {"effect": "Reverb", "params": {"room_size": 0.85, "wet_mix": 0.50}},
            ],
        ]

    def reset(self, **kwargs):
        return [0.0] * 32

    def step(self, action):
        idx = min(self.step_idx, len(self.chains) - 1)
        return (
            [0.0] * 32,
            0.85,
            True,
            {
                "chain": self.chains[idx],
                "metrics": {"snr_db": 19.4, "crest_factor_db": 13.8},
                "audio_ref": f"outputs/audio_gen_00{idx}.wav",
            },
        )


def run():
    print("=" * 80)
    print("  PTA EXPLAINABLE AI (XAI) -- DECISION-TRACE DEMONSTRATION")
    print("  (5-Stage Traceability | 3 Levels of Explanation | Evidence Grounding)")
    print("=" * 80)

    env_step = [0]
    def env_factory():
        env = MockDAPNGymEnv(step_idx=env_step[0])
        env_step[0] += 1
        return env

    adapter = GRRLAdapter(env_factory=env_factory)

    # -------------------------------------------------------------
    # STAGE 0: Initial State
    # -------------------------------------------------------------
    print("\n[STEP 0: Initial Generation]")
    init_res = adapter.generate_initial("dry_guitar.wav", "target.wav", session_id="SESS_DEMO")
    prev_chain = init_res["chain"]
    print(f"  Initial Sample ID : {init_res['sample_id']}")
    print(f"  Initial Chain     : {[b['effect'] for b in prev_chain]}")

    # -------------------------------------------------------------
    # TURN 1: User says "MORE_REVERB"
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[TURN 1: Feedback -> 'MORE_REVERB']")
    print("-" * 80)

    # 1. User feedback
    cmd1 = "MORE_REVERB"
    before_p1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    after_p1 = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]

    # 2. Level 1 Explanation (PTA Math)
    lvl1 = explain_preference_update(cmd1, "reverb", before=0.0, after=0.3, event_id="evt_001")
    print("\n[LEVEL 1: PTA Preference Update]")
    print(f"  Evidence Type : {lvl1['evidence']}")
    print(f"  Explanation   : \"{lvl1['text']}\"")
    print(f"  Delta         : {lvl1['delta']:+.2f}")

    # 3. Guidance to GRRL
    z_taste1 = project_taste(after_p1)
    constraints1 = {
        "prohibited_effects": ["Distortion"],
        "parameter_bounds": {"Reverb": {"room_size": [0.6, 0.9]}},
    }
    print("\n[GUIDANCE SENT TO GRRL]")
    print(f"  Conditioning z_taste (16D) : {z_taste1.round(2).tolist()}")
    print(f"  Prohibited Effect Masks    : {constraints1['prohibited_effects']}")
    print(f"  Parameter Bounds           : {constraints1['parameter_bounds']}")

    # 4 & 5. GRRL Generation & Level 2/3 Explanation
    gen1 = adapter.generate(
        dry_audio="dry_guitar.wav",
        target_audio="target.wav",
        preferences=after_p1,
        constraints=constraints1,
        session_id="SESS_DEMO",
        previous_chain=prev_chain,
        feedback_command=cmd1,
        feedback_event_id="evt_001",
        preference_before=before_p1,
    )

    trace1 = gen1["decision_trace"]
    lvl2_3 = explain_generation(trace1)

    print("\n[LEVEL 2 & 3: GRRL Decision & Audio Outcome Explanation]")
    print(f"  Selected Chain     : {lvl2_3['selected_chain']}")
    print(f"  Parameter Diffs    : {lvl2_3['parameter_changes']}")
    print(f"  Acoustic Metrics   : {lvl2_3['audio_metrics']}")
    print(f"  Full Explanation   :\n    \"{lvl2_3['text']}\"")

    prev_chain = gen1["chain"]

    # -------------------------------------------------------------
    # TURN 2: User says "WIDER" (Mono Invariant Demonstration)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[TURN 2: Feedback -> 'WIDER' (Truthful Mono Pipeline Invariant)]")
    print("-" * 80)

    cmd2 = "WIDER"
    after_p2 = [0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0]
    lvl1_w = explain_preference_update(cmd2, "width", before=0.0, after=0.3, event_id="evt_002")
    print("\n[LEVEL 1: PTA Preference Update]")
    print(f"  Explanation : \"{lvl1_w['text']}\"")

    gen2 = adapter.generate(
        dry_audio="dry_guitar.wav",
        target_audio="target.wav",
        preferences=after_p2,
        constraints={"prohibited_effects": [], "parameter_bounds": {}},
        session_id="SESS_DEMO",
        previous_chain=prev_chain,
        feedback_command=cmd2,
        feedback_event_id="evt_002",
        preference_before=after_p1,
        unsupported_requests=["Mono rendering pipeline (channels=1): spatial widening unrendered"],
    )

    lvl2_3_w = explain_generation(gen2["decision_trace"])
    print("\n[LEVEL 2 & 3: Truthful Handling of Mono Constraints]")
    print(f"  Full Explanation :\n    \"{lvl2_3_w['text']}\"")

    # -------------------------------------------------------------
    # EDGE CASE 1: Failed Rendering (Guards against acoustic claims)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[EDGE CASE 1: Failed Rendering Diagnostic]")
    print("-" * 80)
    failed_trace = DecisionTrace(
        session_id="SESS_DEMO",
        generation_id="gen_err_01",
        feedback_command="MORE_REVERB",
        rendering_status="failed",
    )
    failed_expl = explain_generation(failed_trace)
    print(f"  Rendering Status : {failed_expl['status']}")
    print(f"  Evidence Type    : {failed_expl['evidence']}")
    print(f"  Explanation      : \"{failed_expl['text']}\"")

    # -------------------------------------------------------------
    # EDGE CASE 2: Unmeasured Rendering (No fabricated claims)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[EDGE CASE 2: Audio Rendered Without Acoustic Measurements]")
    print("-" * 80)
    unmeasured_trace = DecisionTrace(
        session_id="SESS_DEMO",
        generation_id="gen_unmeasured_01",
        feedback_command="MORE_DELAY",
        rendering_status="success",
        selected_chain=[{"effect": "Delay", "params": {"wet_mix": 0.4}}],
        audio_metrics={},  # Empty measurements
    )
    unmeasured_expl = explain_generation(unmeasured_trace)
    print(f"  Acoustic Claims Fabricated? : {'No' if 'measurements are not available' in unmeasured_expl['text'] else 'YES (VIOLATION)'}")
    print(f"  Explanation                 : \"{unmeasured_expl['text']}\"")

    # -------------------------------------------------------------
    # TURN 3: Terminal SATISFIED
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[TURN 3: Feedback -> 'SATISFIED' (Terminal Event)]")
    print("-" * 80)
    sat_trace = DecisionTrace(
        session_id="SESS_DEMO",
        generation_id="gen_final",
        feedback_command="SATISFIED",
        rendering_status="success",
        selected_chain=[{"effect": "EQ"}, {"effect": "Reverb"}],
    )
    sat_expl = explain_generation(sat_trace)
    print(f"  Terminal Action Explanation :\n    \"{sat_expl['text']}\"")
    print("=" * 80)
    print("  XAI DECISION TRACE DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run()
