"""
Simulate clean multi-session user runs:
Demonstrates:
  1. Session 1 accumulates feedback and merges into long-term profile.
  2. Session 2 starts completely neutral, applies distinct feedback (delay, width).
  3. Selective dimension merge guarantees untouched dimensions from Session 1
     (e.g., reverb and compression) do NOT decay when Session 2 is merged.
"""
import sys
import shutil
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from pta.core.session_manager import SessionManager
from pta.storage.json_store import JSONPreferenceStore
from pta.models.schemas import VECTOR_LABELS


def run(fresh: bool = False, base_dir: str = "data"):
    data_path = Path(base_dir)
    if fresh and data_path.exists():
        for sub in ["users", "sessions", "feedback"]:
            shutil.rmtree(data_path / sub, ignore_errors=True)
        print(f"[FRESH MODE] Cleared previous data in '{base_dir}'.\n")

    store = JSONPreferenceStore(base_dir=base_dir)

    print("=" * 65)
    print("      PTA v0.2 — VERIFIED MULTI-SESSION DEMONSTRATION")
    print("=" * 65)

    # --- SESSION 1 ---
    print("\n--- SESSION 1 (Initial Track Setup: Electric Guitar) ---")
    s1 = SessionManager("U001", "electric_guitar", store=store)
    print(f"Session 1 initialized neutral: {s1.get_session_vector().round(3).tolist()}")

    s1_clicks = ["MORE_REVERB", "MORE_REVERB", "LESS_COMPRESSION", "SATISFIED"]
    for cmd in s1_clicks:
        if cmd == "SATISFIED":
            res = s1.submit_feedback(cmd)
            print(f"[Terminal] Clicked SATISFIED -> {res['explanation']}")
            break
        result = s1.submit_feedback(cmd)
        print(f"[{result['iteration']}] {cmd:18s} -> {result['explanation']}")

    s1_vec = s1.get_session_vector().round(3).tolist()
    print(f"\nSession 1 final session vector : {s1_vec}")
    # Initial merge with weight=1.0 establishes initial baseline from first session
    merged_s1 = s1.merge_into_long_term(weight=1.0)
    print(f"Long-term profile after S1     : {[round(x, 3) for x in merged_s1]}")

    # --- SESSION 2 ---
    print("\n--- SESSION 2 (Subsequent Session: Fresh Neutral State) ---")
    s2 = SessionManager("U001", "electric_guitar", store=store)
    print(f"Session 2 starts neutral       : {s2.get_session_vector().round(3).tolist()}")

    s2_clicks = ["MORE_DELAY", "WIDER", "SATISFIED"]
    for cmd in s2_clicks:
        if cmd == "SATISFIED":
            res = s2.submit_feedback(cmd)
            print(f"[Terminal] Clicked SATISFIED -> {res['explanation']}")
            break
        result = s2.submit_feedback(cmd)
        print(f"[{result['iteration']}] {cmd:18s} -> {result['explanation']}")

    s2_vec = s2.get_session_vector().round(3).tolist()
    print(f"\nSession 2 final session vector : {s2_vec}")
    merged_s2 = s2.merge_into_long_term(weight=0.5)
    print(f"Long-term profile after S2     : {[round(x, 3) for x in merged_s2]}")

    print("\nSummary of Dimension Evolution:")
    for label, v1, v2 in zip(VECTOR_LABELS, merged_s1, merged_s2):
        print(f"  {label:12s}: After S1={v1:+.3f}  ->  After S2={v2:+.3f}")

    print("=" * 65)


if __name__ == "__main__":
    is_fresh = "--fresh" in sys.argv
    run(fresh=is_fresh, base_dir="data")
