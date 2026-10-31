"""PTA → M3 guidance message schemas. Matches the actual M3 contract."""
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Literal, Optional
from pta.models.schemas import VECTOR_DIM, VECTOR_LABELS

# 7 effects supported by M2/M3 (from the integration PDF)
SUPPORTED_EFFECTS = ["EQ", "Compressor", "Delay", "Reverb",
                     "Distortion", "Chorus", "Flanger"]

VALID_FEEDBACK_COMMANDS = {
    "MORE_REVERB", "LESS_REVERB", "MORE_COMPRESSION", "LESS_COMPRESSION",
    "MORE_DELAY", "LESS_DELAY", "BRIGHTER", "WARMER",
    "MORE_SATURATION", "WIDER", "SATISFIED"
}

# Taste embedding dimension — M3 expects R^16 or R^32
Z_TASTE_DIM = 16


@dataclass
class RewardGuidance:
    """Mode 1 — scalar reward shaping."""
    r_pta: float = 0.0                # in [-1, 1]
    confidence: float = 1.0           # in [0, 1]
    explanation: str = ""


@dataclass
class ConstraintMask:
    """Mode 3 — action masking for the RL policy."""
    prohibited_effects: List[str] = field(default_factory=list)
    parameter_bounds: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    # e.g. {"Reverb": {"wet_mix": [0.3, 0.6]}, "Compressor": {"ratio": [0.1, 0.4]}}


@dataclass
class GuidanceMessage:
    """The full PTA → M3 payload."""
    schema_version: str = "1.0"
    user_id: str = ""
    session_id: str = ""
    sample_id: str = ""               # links feedback to generated audio
    event_type: Literal["PREFERENCE_UPDATED", "SESSION_SATISFIED"] = "PREFERENCE_UPDATED"

    # Feedback context
    feedback_command: str = ""
    changed_dimensions: List[str] = field(default_factory=list)

    # Mode 1: reward guidance
    reward: RewardGuidance = field(default_factory=RewardGuidance)

    # Mode 2: conditioning vector
    z_taste: List[float] = field(default_factory=lambda: [0.0] * Z_TASTE_DIM)

    # Mode 3: constraint masking
    constraints: ConstraintMask = field(default_factory=ConstraintMask)

    # Action
    action: Literal["REGENERATE", "NONE", "SESSION_CLOSED", "NO_REGENERATION"] = "REGENERATE"

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("Unsupported schema_version")
        if not isinstance(self.user_id, str) or not self.user_id.strip() or not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("user_id and session_id required")
        if not isinstance(self.sample_id, str):
            raise ValueError("sample_id must be a string")
        if self.feedback_command and self.feedback_command not in VALID_FEEDBACK_COMMANDS:
            raise ValueError(f"Unknown feedback_command: {self.feedback_command}")
        if self.event_type not in ("PREFERENCE_UPDATED", "SESSION_SATISFIED"):
            raise ValueError(f"Unknown event_type: {self.event_type}")
        if self.event_type == "SESSION_SATISFIED":
            if self.action not in ("NONE", "SESSION_CLOSED", "NO_REGENERATION"):
                raise ValueError("SESSION_SATISFIED requires action=NONE")
            if len(self.changed_dimensions) > 0:
                raise ValueError("SESSION_SATISFIED must not have changed dimensions")
        if self.event_type == "PREFERENCE_UPDATED" and self.action != "REGENERATE":
            raise ValueError("PREFERENCE_UPDATED requires action=REGENERATE")
        if not isinstance(self.changed_dimensions, list):
            raise ValueError("changed_dimensions must be a list")
        for dim in self.changed_dimensions:
            if dim not in VECTOR_LABELS:
                raise ValueError(f"Unknown dimension in changed_dimensions: {dim}")
        if len(self.changed_dimensions) != len(set(self.changed_dimensions)):
            raise ValueError("Duplicate dimensions in changed_dimensions")
        if len(self.z_taste) not in (Z_TASTE_DIM, 32):
            raise ValueError(f"z_taste must have length {Z_TASTE_DIM}")
        for v in self.z_taste:
            if math.isnan(v) or math.isinf(v) or not (-1.0 <= v <= 1.0):
                raise ValueError(f"z_taste out of range: {v}")
        if math.isnan(self.reward.r_pta) or math.isinf(self.reward.r_pta) or not (-1.0 <= self.reward.r_pta <= 1.0):
            raise ValueError(f"r_pta out of range: {self.reward.r_pta}")
        if math.isnan(self.reward.confidence) or math.isinf(self.reward.confidence) or not (0.0 <= self.reward.confidence <= 1.0):
            raise ValueError(f"confidence out of range: {self.reward.confidence}")
        for eff in self.constraints.prohibited_effects:
            if eff not in SUPPORTED_EFFECTS:
                raise ValueError(f"Unknown effect in constraints: {eff}")
        if not isinstance(self.constraints.parameter_bounds, dict):
            raise ValueError("parameter_bounds must be a dict")
        for eff, params in self.constraints.parameter_bounds.items():
            if eff not in SUPPORTED_EFFECTS:
                raise ValueError(f"Unknown effect in parameter_bounds: {eff}")
            if not isinstance(params, dict):
                raise ValueError(f"Parameters for {eff} must be a dict")
            for p_name, bounds in params.items():
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError(f"Bounds for {eff}.{p_name} must be a 2-element list [low, high]")
                low, high = bounds
                if math.isnan(low) or math.isinf(low) or math.isnan(high) or math.isinf(high):
                    raise ValueError(f"Non-finite bound values for {eff}.{p_name}")
                if not (0.0 <= low <= high <= 1.0):
                    raise ValueError(f"Invalid bound range [{low}, {high}] for {eff}.{p_name}; must satisfy 0.0 <= low <= high <= 1.0")

    def to_dict(self) -> dict:
        return asdict(self)
