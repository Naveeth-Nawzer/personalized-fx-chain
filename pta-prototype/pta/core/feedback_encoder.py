"""Convert a feedback button click into a directional vector."""
import numpy as np
from pta.models.schemas import VECTOR_DIM


# Feedback taxonomy 
FEEDBACK_CONTROLS = {
    # control_name: (vector_index, direction)
    "MORE_REVERB":       (2, +1),
    "LESS_REVERB":       (2, -1),
    "MORE_COMPRESSION":  (1, +1),
    "LESS_COMPRESSION":  (1, -1),
    "MORE_DELAY":        (3, +1),
    "LESS_DELAY":        (3, -1),
    "BRIGHTER":          (0, +1),
    "WARMER":            (6, +1),   # warmth is its own dimension
    "MORE_SATURATION":   (4, +1),
    "WIDER":             (5, +1),
    "SATISFIED":         None,       # terminal, no vector
}


def encode_feedback(command: str) -> np.ndarray:
    """
    Convert a feedback command into a one-hot directional vector.

    Example:
        "MORE_REVERB" -> [0, 0, +1, 0, 0, 0, 0]
        "LESS_COMPRESSION" -> [0, -1, 0, 0, 0, 0, 0]
    """
    vec = np.zeros(VECTOR_DIM)
    if command not in FEEDBACK_CONTROLS:
        raise ValueError(f"Unknown feedback command: {command}")
    if FEEDBACK_CONTROLS[command] is None:
        return vec  # SATISFIED produces no change
    idx, direction = FEEDBACK_CONTROLS[command]
    vec[idx] = direction
    return vec


def list_available_controls() -> list:
    """Return all valid feedback commands."""
    return [k for k, v in FEEDBACK_CONTROLS.items() if v is not None]
