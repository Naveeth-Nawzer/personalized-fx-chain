"""Decision trace data structures for evidence-grounded Explainable AI (XAI).

Provides a structured, auditable record connecting:
1. User feedback events
2. PTA preference vector updates
3. Three-mode guidance and constraint masks sent to GRRL
4. Actual effects and parameters selected by GRRL
5. Rendered DAPN audio status and measurable acoustic outcomes
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import uuid


VALID_RENDERING_STATUSES = {"not_run", "success", "failed"}


@dataclass
class DecisionTrace:
    """
    A generation-specific record linking user feedback, preference updates,
    guidance sent, GRRL decisions, and DAPN audio rendering outcomes.
    """
    session_id: str
    generation_id: str
    feedback_event_id: Optional[str] = None
    feedback_command: Optional[str] = None
    preference_before: Dict[str, float] = field(default_factory=dict)
    preference_after: Dict[str, float] = field(default_factory=dict)
    guidance_sent: Dict[str, Any] = field(default_factory=dict)
    selected_chain: List[Dict[str, Any]] = field(default_factory=list)
    previous_chain: List[Dict[str, Any]] = field(default_factory=list)
    rendering_status: str = "not_run"  # "not_run", "success", "failed"
    audio_ref: Optional[str] = None
    audio_metrics: Dict[str, float] = field(default_factory=dict)
    unsupported_requests: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate trace fields against schema invariants."""
        if not self.session_id or not isinstance(self.session_id, str):
            raise ValueError("session_id must be a non-empty string")
        if not self.generation_id or not isinstance(self.generation_id, str):
            raise ValueError("generation_id must be a non-empty string")
        if self.rendering_status not in VALID_RENDERING_STATUSES:
            raise ValueError(
                f"Invalid rendering_status: {self.rendering_status}. "
                f"Must be one of {VALID_RENDERING_STATUSES}"
            )
        # Check preferences if provided
        for label, val in self.preference_before.items():
            if not isinstance(val, (int, float)) or not (-1.0 <= float(val) <= 1.0):
                raise ValueError(f"preference_before['{label}'] must be in [-1.0, 1.0]; got {val}")
        for label, val in self.preference_after.items():
            if not isinstance(val, (int, float)) or not (-1.0 <= float(val) <= 1.0):
                raise ValueError(f"preference_after['{label}'] must be in [-1.0, 1.0]; got {val}")

    def is_rendered(self) -> bool:
        """Return True if rendering was confirmed successful."""
        return self.rendering_status == "success"

    def has_audio_measurements(self) -> bool:
        """Return True if objective audio measurements are present."""
        return self.is_rendered() and bool(self.audio_metrics)

    def get_selected_effect_names(self) -> List[str]:
        """Return ordered list of effect names in the selected chain."""
        return [str(e.get("effect", "Unknown")) for e in self.selected_chain]

    def get_previous_effect_names(self) -> List[str]:
        """Return ordered list of effect names in the previous chain."""
        return [str(e.get("effect", "Unknown")) for e in self.previous_chain]

    def get_parameter_diff(self) -> List[Dict[str, Any]]:
        """
        Compare previous_chain and selected_chain parameters for matching effects.

        Returns a list of diff dictionaries:
            {
                "effect": str,
                "parameter": str,
                "before": float,
                "after": float,
                "delta": float,
            }
        """
        diffs = []
        # Index previous chain by effect name
        prev_by_effect: Dict[str, Dict[str, float]] = {}
        for block in self.previous_chain:
            eff = block.get("effect")
            params = block.get("params") or block.get("parameters") or {}
            if eff and isinstance(params, dict):
                prev_by_effect[eff] = {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}

        # Compare with current chain
        for block in self.selected_chain:
            eff = block.get("effect")
            curr_params = block.get("params") or block.get("parameters") or {}
            if not eff or not isinstance(curr_params, dict):
                continue

            prev_params = prev_by_effect.get(eff, {})
            for p_name, p_val in curr_params.items():
                if not isinstance(p_val, (int, float)):
                    continue
                curr_val = float(p_val)
                if p_name in prev_params:
                    prev_val = float(prev_params[p_name])
                    diffs.append({
                        "effect": eff,
                        "parameter": p_name,
                        "before": prev_val,
                        "after": curr_val,
                        "delta": curr_val - prev_val,
                    })

        return diffs

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trace to JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionTrace":
        """Deserialize from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            generation_id=data.get("generation_id", ""),
            feedback_event_id=data.get("feedback_event_id"),
            feedback_command=data.get("feedback_command"),
            preference_before=data.get("preference_before", {}),
            preference_after=data.get("preference_after", {}),
            guidance_sent=data.get("guidance_sent", {}),
            selected_chain=data.get("selected_chain", []),
            previous_chain=data.get("previous_chain", []),
            rendering_status=data.get("rendering_status", "not_run"),
            audio_ref=data.get("audio_ref"),
            audio_metrics=data.get("audio_metrics", {}),
            unsupported_requests=data.get("unsupported_requests", []),
        )
