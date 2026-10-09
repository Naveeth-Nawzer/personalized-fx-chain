"""Demo: PTA sends three-mode guidance to mock M3."""
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from pta.core.guidance_builder import build_guidance_message
from pta.integration.m3_adapter import M3Adapter, MockM3Receiver


def run():
    print("=" * 70)
    print("  TASK #8 DEMO -- PTA -> M3 THREE-MODE GUIDANCE INTERFACE")
    print("=" * 70)

    receiver = MockM3Receiver()
    adapter = M3Adapter(receiver)

    preference = np.array([0.0, -0.3, 0.51, 0.15, 0.0, 0.15, 0.0])

    events = [
        ("sample_00482", "MORE_REVERB"),
        ("sample_00483", "LESS_COMPRESSION"),
        ("sample_00484", "WIDER"),
        ("sample_00485", "SATISFIED"),
    ]

    for sample_id, cmd in events:
        msg = build_guidance_message(
            "U001", "sess_20261017_001", sample_id, cmd, preference
        )
        response = adapter.send(msg)

        print(f"\n[{cmd}] sample_id={sample_id}")
        print(f"  event_type  : {msg.event_type}")
        print(f"  action      : {msg.action}")
        print(f"  r_pta       : {msg.reward.r_pta:.3f}")
        print(f"  z_taste dim : {len(msg.z_taste)}")
        print(f"  constraints : {list(msg.constraints.parameter_bounds.keys())}")
        print(f"  response    : {response['status']}")

    print("\n" + "=" * 70)
    print(f"Total messages received: {len(receiver.received)}")
    print(f"Reward history: {[round(r, 3) for r in receiver.reward_history]}")

    # Save one example message as JSON
    evidence_dir = Path("evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    example = build_guidance_message(
        "U001", "sess_20261017_001", "sample_00482", "MORE_REVERB", preference
    )
    with open(evidence_dir / "sample_guidance_message.json", "w") as f:
        json.dump(example.to_dict(), f, indent=2)
    print(f"\nExample message saved to evidence/sample_guidance_message.json")
    print("=" * 70)


if __name__ == "__main__":
    run()
