"""Full PTA loop — all components mocked.

Demonstrates:
    dry audio → M1 features → M3 initial chain → user feedback →
    PTA update → guidance → M3 regeneration → repeat → SATISFIED

⚠️ All components are MOCKED. No real audio is processed.
   No RL policy is trained. No acoustic losses are computed.

This demonstrates that the PTA component behaves correctly and is
contract-ready for real M3 integration.
"""
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from pta.core.session_manager import SessionManager
from pta.core.guidance_builder import build_guidance_message
from pta.integration.m3_adapter import M3Adapter, MockM3Receiver
from pta.integration.mock_m3_generator import MockM3Generator
from pta.integration.mock_m1_features import MockM1Features
from pta.models.guidance import GuidanceMessage
from pta.storage.json_store import JSONPreferenceStore


def run():
    print("=" * 72)
    print("  FULL PTA LOOP -- ALL COMPONENTS MOCKED")
    print("  (No real audio | No RL policy | Contract-validated demo)")
    print("=" * 72)

    store = JSONPreferenceStore(base_dir="data")
    m1 = MockM1Features()
    m3_gen = MockM3Generator()
    receiver = MockM3Receiver()
    adapter = M3Adapter(receiver)

    session_mgr = SessionManager("U001", "electric_guitar", store=store)
    session_id = session_mgr.session.session_id

    # --- Step 1: M1 (mocked) extracts features from dry audio ---
    dry_features = m1.extract("dry_guitar_001", source="dry")
    print(f"\n[M1 MOCK] Dry features:")
    print(f"  spectral_centroid_hz: {dry_features['spectral_centroid_hz']}")
    print(f"  crest_factor_db:      {dry_features['crest_factor_db']}")
    print(f"  rms_dbFS:             {dry_features['rms_dbFS']}")

    # --- Step 2: M3 (mocked) generates initial chain ---
    initial = m3_gen.generate(_empty_guidance("U001", session_id))
    sample_id = initial["sample_id"]
    print(f"\n[M3 MOCK] Initial chain: {initial['metadata']['chain']}")
    print(f"[M3 MOCK] sample_id:     {sample_id}")

    # --- Step 3: User feedback loop ---
    feedback_sequence = [
        "MORE_REVERB",
        "MORE_REVERB",
        "LESS_COMPRESSION",
        "WIDER",
        "SATISFIED",
    ]

    for cmd in feedback_sequence:
        # PTA: update preference state
        result = session_mgr.submit_feedback(cmd)
        prefs = session_mgr.get_session_vector()

        # PTA: build guidance message
        guidance = build_guidance_message(
            "U001", session_id, sample_id, cmd, prefs
        )

        # Adapter: send to M3
        response = adapter.send(guidance)

        print(f"\n[USER] {cmd}")
        print(f"  PTA vector        : {prefs.round(3).tolist()}")
        print(f"  changed_dimensions: {guidance.changed_dimensions}")
        print(f"  r_pta             : {guidance.reward.r_pta:.3f} "
              f"(confidence: {guidance.reward.confidence:.1f})")
        print(f"  constraints       : {list(guidance.constraints.parameter_bounds.keys())}")
        print(f"  M3 response       : {response['status']}")

        if cmd != "SATISFIED":
            # M3: regenerate (mocked)
            regen = m3_gen.generate(guidance)
            sample_id = regen["sample_id"]
            print(f"  new sample_id     : {sample_id}")
            print(f"  new chain         : {regen['metadata']['chain']}")

    # --- Summary ---
    print("\n" + "=" * 72)
    print("LOOP COMPLETE")
    print(f"  Total feedback events  : {len(receiver.received)}")
    print(f"  Total M3 regenerations : {len(m3_gen.samples)}")
    print(f"  Session satisfied      : {session_mgr.session.satisfied}")
    print(f"  Final PTA vector       : {session_mgr.get_session_vector().round(3).tolist()}")

    # Save evidence
    evidence_dir = Path("evidence")
    evidence_dir.mkdir(exist_ok=True)

    with (evidence_dir / "mocked_loop_sessions.json").open("w", encoding="utf-8") as f:
        json.dump(m3_gen.samples, f, indent=2)

    print(f"\n  Evidence saved to evidence/mocked_loop_sessions.json")
    print("=" * 72)


def _empty_guidance(user_id: str, session_id: str) -> GuidanceMessage:
    """Empty guidance for initial generation."""
    return GuidanceMessage(
        user_id=user_id,
        session_id=session_id,
        sample_id="initial",
    )


if __name__ == "__main__":
    run()
