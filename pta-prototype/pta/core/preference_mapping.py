"""Perceptual-to-DSP guidance mapping specification for PTA → GRRL integration.

Bridges human-facing feedback commands to:
1. 7-dimensional preference vector updates.
2. Structured DSP parameter intents for GRRL/DAPN.
3. Distinction between soft preferences, hard constraints, and terminal actions.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from pta.models.schemas import VECTOR_DIM, VECTOR_LABELS

PREFERENCE_DIMENSIONS = (
    "brightness",
    "compression",
    "reverb",
    "delay",
    "saturation",
    "width",
    "warmth",
)

# Supported DAPN processors
SUPPORTED_EFFECTS = (
    "EQ",
    "Compressor",
    "Delay",
    "Reverb",
    "Distortion",
    "Chorus",
    "Flanger",
)


@dataclass(frozen=True)
class FeedbackIntent:
    """Structured intent representing how user feedback influences PTA and GRRL."""
    command: str
    dimension: Optional[str]
    direction: float                     # +1.0 (more/boost), -1.0 (less/cut), 0.0 (neutral/terminal)
    target_effect: Optional[str]         # DAPN processor target, if directly mapped
    target_parameters: List[str]         # Physical parameters guided by this intent
    guidance_description: str
    is_terminal: bool = False
    is_supported: bool = True            # False if hardware/pipeline cannot render (e.g., width on mono)


# Formalized mapping matrix matching Section 5 of the architecture specification
FEEDBACK_INTENT_CATALOG: Dict[str, FeedbackIntent] = {
    "MORE_REVERB": FeedbackIntent(
        command="MORE_REVERB",
        dimension="reverb",
        direction=1.0,
        target_effect="Reverb",
        target_parameters=["wet_mix", "room_size", "decay"],
        guidance_description="Increase reverb prominence via wet_mix and decay time within validated bounds.",
    ),
    "LESS_REVERB": FeedbackIntent(
        command="LESS_REVERB",
        dimension="reverb",
        direction=-1.0,
        target_effect="Reverb",
        target_parameters=["wet_mix"],
        guidance_description="Reduce reverb contribution by lowering wet_mix ratio.",
    ),
    "MORE_COMPRESSION": FeedbackIntent(
        command="MORE_COMPRESSION",
        dimension="compression",
        direction=1.0,
        target_effect="Compressor",
        target_parameters=["threshold", "ratio", "makeup_gain"],
        guidance_description="Increase dynamic range compression by lowering threshold and increasing ratio.",
    ),
    "LESS_COMPRESSION": FeedbackIntent(
        command="LESS_COMPRESSION",
        dimension="compression",
        direction=-1.0,
        target_effect="Compressor",
        target_parameters=["threshold", "ratio"],
        guidance_description="Reduce compression strength by raising threshold and relaxing ratio.",
    ),
    "MORE_DELAY": FeedbackIntent(
        command="MORE_DELAY",
        dimension="delay",
        direction=1.0,
        target_effect="Delay",
        target_parameters=["wet_mix", "feedback", "delay_time"],
        guidance_description="Increase echo prominence via wet_mix and feedback repeats.",
    ),
    "LESS_DELAY": FeedbackIntent(
        command="LESS_DELAY",
        dimension="delay",
        direction=-1.0,
        target_effect="Delay",
        target_parameters=["wet_mix", "feedback"],
        guidance_description="Reduce audible echo repeats and wet delay mix.",
    ),
    "BRIGHTER": FeedbackIntent(
        command="BRIGHTER",
        dimension="brightness",
        direction=1.0,
        target_effect="EQ",
        target_parameters=["gain", "frequency"],
        guidance_description="Guide parametric EQ toward higher frequency emphasis or high shelf boost.",
    ),
    "WARMER": FeedbackIntent(
        command="WARMER",
        dimension="warmth",
        direction=1.0,
        target_effect="EQ",
        target_parameters=["gain", "frequency"],
        guidance_description="Guide parametric EQ toward low-mid emphasis (250-800 Hz) with gentle high rolloff.",
    ),
    "MORE_SATURATION": FeedbackIntent(
        command="MORE_SATURATION",
        dimension="saturation",
        direction=1.0,
        target_effect="Distortion",
        target_parameters=["drive", "mix"],
        guidance_description="Increase harmonic drive and wet distortion mix within safe headroom limits.",
    ),
    "WIDER": FeedbackIntent(
        command="WIDER",
        dimension="width",
        direction=1.0,
        target_effect=None,
        target_parameters=[],
        guidance_description="Stereo width preference recorded in taste profile; reported as unsupported in mono pipeline.",
        is_supported=False,
    ),
    "SATISFIED": FeedbackIntent(
        command="SATISFIED",
        dimension=None,
        direction=0.0,
        target_effect=None,
        target_parameters=[],
        guidance_description="Session satisfied. Closes feedback loop without altering preference vector or triggering regeneration.",
        is_terminal=True,
    ),
}


def get_feedback_intent(command: str) -> FeedbackIntent:
    """Retrieve structured intent for a feedback command."""
    cmd = command.strip().upper()
    if cmd not in FEEDBACK_INTENT_CATALOG:
        raise ValueError(
            f"Unknown feedback command: '{command}'. "
            f"Expected one of: {list(FEEDBACK_INTENT_CATALOG.keys())}"
        )
    return FEEDBACK_INTENT_CATALOG[cmd]


def validate_preference_dict(preferences: Dict[str, float]) -> np.ndarray:
    """
    Validate and convert a dictionary of preferences into the canonical 7-dim array.

    Enforces:
    - All 7 PREFERENCE_DIMENSIONS are present
    - Values are finite numbers within [-1.0, 1.0]
    """
    if not isinstance(preferences, dict):
        raise TypeError("Preferences must be a dictionary")

    missing = [dim for dim in PREFERENCE_DIMENSIONS if dim not in preferences]
    if missing:
        raise ValueError(f"Missing required preference dimensions: {missing}")

    values = np.asarray(
        [float(preferences[name]) for name in PREFERENCE_DIMENSIONS],
        dtype=np.float32,
    )

    if values.shape != (VECTOR_DIM,):
        raise ValueError(f"Expected exactly {VECTOR_DIM} preference values, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Preferences must be finite numbers (no NaN or Inf)")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError(f"All preference values must be in [-1.0, 1.0]; got {values}")

    return values
