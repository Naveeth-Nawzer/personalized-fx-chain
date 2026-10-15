"""Mock M3 generator — simulates FX-chain regeneration from guidance.

⚠️ This is a MOCK. It does not:
  - Render real audio
  - Run the PPO policy
  - Compute acoustic losses
  - Update the RL agent

It only produces metadata (sample_id, chain, checksum) sufficient to
exercise the PTA feedback loop during PP1. Replace with real M3 adapter
once Nikshan's PPO loop is ready.
"""
import uuid
from pta.models.guidance import GuidanceMessage


class MockM3Generator:
    """
    Simulates M3 receiving PTA guidance and returning a regenerated chain.

    Contract:
        Input:  GuidanceMessage (PREFERENCE_UPDATED)
        Output: {"status": "REGENERATED", "sample_id": str, "metadata": dict}

    For SESSION_SATISFIED:
        Output: {"status": "NO_REGENERATION", "sample_id": str}
    """

    def __init__(self):
        self.samples = {}  # sample_id → metadata

    def generate_initial(self, user_id: str = "U001", session_id: str = "S001") -> dict:
        """
        Generate initial default chain before any feedback occurs.
        Does not require a fabricated GuidanceMessage.
        """
        if not user_id or not session_id:
            raise ValueError("user_id and session_id required")

        initial_sample_id = f"sample_{uuid.uuid4().hex[:8]}"
        initial_chain = [{"effect": "EQ", "parameters": {"gain": 0.5}}]
        metadata = {
            "sample_id": initial_sample_id,
            "user_id": user_id,
            "session_id": session_id,
            "chain": initial_chain,
            "audio_checksum": f"chk_{uuid.uuid4().hex[:8]}",
            "audio_duration_s": 2.0,
            "sample_rate": 44100,
        }
        self.samples[initial_sample_id] = metadata

        return {
            "status": "INITIAL_GENERATED",
            "sample_id": initial_sample_id,
            "metadata": metadata,
        }

    def generate(self, guidance: GuidanceMessage) -> dict:
        """Simulate M3 producing a new chain from PTA guidance."""
        guidance.validate()

        if guidance.event_type == "SESSION_SATISFIED":
            return {
                "status": "NO_REGENERATION",
                "sample_id": guidance.sample_id,
                "reason": "User satisfied — session closed.",
            }

        new_sample_id = f"sample_{uuid.uuid4().hex[:8]}"
        chain = self._synthesize_chain(guidance)

        metadata = {
            "sample_id": new_sample_id,
            "user_id": guidance.user_id,
            "session_id": guidance.session_id,
            "chain": chain,
            "audio_checksum": f"chk_{uuid.uuid4().hex[:8]}",
            "audio_duration_s": 2.0,
            "sample_rate": 44100,
            # NOTE: no audio tensor — metadata-only mock
        }
        self.samples[new_sample_id] = metadata

        return {
            "status": "REGENERATED",
            "sample_id": new_sample_id,
            "metadata": metadata,
        }

    def _synthesize_chain(self, guidance: GuidanceMessage) -> list:
        """
        Build a fake chain that respects guidance.parameter_bounds.

        Deterministic — same guidance → same chain (for reproducibility).
        """
        chain = []
        for effect, params in guidance.constraints.parameter_bounds.items():
            chain.append({
                "effect": effect,
                "parameters": {
                    p: round((bounds[0] + bounds[1]) / 2, 3)
                    for p, bounds in params.items()
                },
            })

        # If no constraints, default to a neutral EQ
        if not chain:
            chain = [{"effect": "EQ", "parameters": {"gain": 0.5}}]

        return chain
