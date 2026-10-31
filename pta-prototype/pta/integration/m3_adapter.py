"""Adapter for sending three-mode guidance to M3."""
from typing import List
from pta.models.guidance import GuidanceMessage


class MockM3Receiver:
    """Simulates M3 accepting the three guidance modes."""
    def __init__(self):
        self.received: List[GuidanceMessage] = []
        self.reward_history: List[float] = []

    def receive(self, message: GuidanceMessage) -> dict:
        message.validate()
        self.received.append(message)

        if message.event_type == "SESSION_SATISFIED":
            return {
                "status": "SESSION_CLOSED",
                "final_sample_id": message.sample_id,
            }

        self.reward_history.append(message.reward.r_pta)

        return {
            "status": "ACKNOWLEDGED",
            "action": message.action,
            "z_taste_dims": len(message.z_taste),
            "r_pta": message.reward.r_pta,
            "constraint_effects": list(message.constraints.parameter_bounds.keys()),
        }


class M3Adapter:
    def __init__(self, receiver=None):
        self.receiver = receiver or MockM3Receiver()

    def send(self, message: GuidanceMessage) -> dict:
        message.validate()
        return self.receiver.receive(message)
