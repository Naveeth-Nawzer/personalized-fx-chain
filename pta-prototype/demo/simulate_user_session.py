"""
Simulate a complete user session:
1. User clicks MORE_REVERB
2. Preference vector updates
3. Explanation is generated
4. Profile is persisted to JSON
5. All events are logged
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pta.core.preference_engine import PreferenceVectorEngine
from pta.core.explanation_engine import generate_explanation
from pta.storage.json_store import JSONPreferenceStore
from pta.models.schemas import VECTOR_LABELS


def run():
    print("=" * 60)
    print("  PTA PROTOTYPE — SIMULATED USER SESSION")
    print("=" * 60)

    user_id = "U001"
    session_id = "S001"
    instrument = "electric_guitar"

    store = JSONPreferenceStore(base_dir="data")
    engine = PreferenceVectorEngine(user_id, alpha=0.3)
    engine.P_u = __import__("numpy").array(store.load_profile(user_id, instrument))

    # Simulate a sequence of feedback clicks
    clicks = ["MORE_REVERB", "MORE_REVERB", "LESS_COMPRESSION", "WIDER", "SATISFIED"]

    for i, command in enumerate(clicks, start=1):
        if command == "SATISFIED":
            print(f"\n[{i}] User clicked SATISFIED — session complete.")
            break

        result = engine.update(command)

        explanation = generate_explanation(
            feedback_command=command,
            preference_before=result["before"],
            preference_after=result["after"],
        )

        # Log feedback event
        store.log_feedback({
            "session_id": session_id,
            "user_id": user_id,
            "iteration": i,
            "command": command,
            "before": result["before"],
            "after": result["after"],
            "delta": result["delta"],
        })

        print(f"\n[{i}] User clicked: {command}")
        print(f"    Preference vector updated:")
        for label, b, a in zip(VECTOR_LABELS, result["before"], result["after"]):
            if abs(b - a) > 1e-6:
                print(f"      {label:12s}: {b:+.3f} -> {a:+.3f}")
        print(f"    Explanation: {explanation['text']}")

    # Persist final profile
    store.save_profile(user_id, engine.get_vector().tolist(), instrument)
    print(f"\nFinal profile saved for {user_id} / {instrument}")
    print(f"Final vector: {engine.get_vector().round(3).tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    run()
