"""Build PTA → M3 messages in the three modes M3 accepts."""
import numpy as np
from pta.models.guidance import (
    GuidanceMessage, RewardGuidance, ConstraintMask,
    Z_TASTE_DIM, SUPPORTED_EFFECTS,
)
from pta.models.schemas import VECTOR_DIM, VECTOR_LABELS


# Map preference dimension → M2/M3 effect it influences
DIM_TO_EFFECT = {
    "brightness": "EQ",
    "compression": "Compressor",
    "reverb": "Reverb",
    "delay": "Delay",
    "saturation": "Distortion",
    "width": None,       # width handled by stereo tools in DAPN
    "warmth": "EQ",
}

# Map preference dimension → M2 parameters (normalized [0,1] space)
DIM_TO_PARAMS = {
    "brightness":  {"EQ": ["gain"]},
    "compression": {"Compressor": ["threshold", "ratio", "makeup_gain"]},
    "reverb":      {"Reverb": ["wet_mix", "dry_mix", "room_size"]},
    "delay":       {"Delay": ["wet_mix", "feedback", "delay_time"]},
    "saturation":  {"Distortion": ["drive", "mix"]},
    "width":       {},   # no direct single-effect param
    "warmth":      {"EQ": ["gain"]},
}

COMMAND_TO_DIM = {
    "MORE_REVERB": "reverb",
    "LESS_REVERB": "reverb",
    "MORE_COMPRESSION": "compression",
    "LESS_COMPRESSION": "compression",
    "MORE_DELAY": "delay",
    "LESS_DELAY": "delay",
    "BRIGHTER": "brightness",
    "WARMER": "warmth",
    "MORE_SATURATION": "saturation",
    "WIDER": "width",
}


def project_taste_vector(preference_vector: np.ndarray) -> list:
    """
    Project 7-dim preference vector → 16-dim z_taste.

    For PP1: deterministic projection (7-dim + padding with semantic
    cross-products). Post-PP1: replace with a trained encoder.
    """
    if len(preference_vector) != VECTOR_DIM:
        raise ValueError(f"preference_vector must have {VECTOR_DIM} dimensions")

    z = np.zeros(Z_TASTE_DIM)
    # First 7 slots: raw preference values
    z[:VECTOR_DIM] = preference_vector
    # Slots 7-12: pairwise interactions of the most influential dimensions
    #   (compression × reverb), (reverb × delay), (brightness × warmth),
    #   (saturation × warmth), (compression × saturation), (width × reverb)
    z[7]  = preference_vector[1] * preference_vector[2]   # comp × reverb
    z[8]  = preference_vector[2] * preference_vector[3]   # reverb × delay
    z[9]  = preference_vector[0] * preference_vector[6]   # bright × warm
    z[10] = preference_vector[4] * preference_vector[6]   # sat × warm
    z[11] = preference_vector[1] * preference_vector[4]   # comp × sat
    z[12] = preference_vector[5] * preference_vector[2]   # width × reverb
    # Slots 13-15: magnitude aggregates (L1, L2, max abs)
    z[13] = np.sum(np.abs(preference_vector)) / 7.0
    z[14] = np.sqrt(np.sum(preference_vector ** 2)) / np.sqrt(7.0)
    z[15] = np.max(np.abs(preference_vector))

    return np.clip(z, -1.0, 1.0).tolist()


def compute_reward_guidance(preference_vector: np.ndarray,
                            feedback_command: str) -> RewardGuidance:
    """
    Compute scalar R_pta.

    Semantics: how well does the current audio match the user's taste?
      - SATISFIED  → +1.0 (maximum satisfaction)
      - Any change request → negative, magnitude = L1 norm of preference delta
    """
    if feedback_command == "SATISFIED":
        return RewardGuidance(r_pta=+1.0, confidence=1.0,
                              explanation="User satisfied.")

    # Feedback magnitude = how much the user wanted to change
    magnitude = float(np.sum(np.abs(preference_vector))) / 7.0
    r_pta = -magnitude   # negative = current audio not matching taste

    return RewardGuidance(
        r_pta=float(np.clip(r_pta, -1.0, 1.0)),
        confidence=0.8,
        explanation=f"User requested {feedback_command}; "
                    f"preference magnitude {magnitude:.2f}.",
    )


def build_constraints(preference_vector: np.ndarray) -> ConstraintMask:
    """
    Build constraint mask.

    Only adds constraints when preference is strong (|value| > 0.3).
    """
    bounds = {}
    prohibited = []

    for i, dim in enumerate(VECTOR_LABELS):
        p = float(preference_vector[i])
        if abs(p) <= 0.3:
            continue        # weak preference → no constraint

        if p > 0.3:
            # Strong positive preference: keep effect, target upper range
            for effect, params in DIM_TO_PARAMS[dim].items():
                bounds.setdefault(effect, {})
                for param in params:
                    bounds[effect][param] = [0.4, 0.9]

        elif p < -0.3:
            # Strong negative preference: suppress or clamp to lower range
            for effect, params in DIM_TO_PARAMS[dim].items():
                if effect and effect not in prohibited:
                    # For "less" preferences, tighten param range downward
                    bounds.setdefault(effect, {})
                    for param in params:
                        bounds[effect][param] = [0.1, 0.3]

    return ConstraintMask(
        prohibited_effects=prohibited,
        parameter_bounds=bounds,
    )


def build_guidance_message(
    user_id: str,
    session_id: str,
    sample_id: str,
    feedback_command: str,
    preference_vector: np.ndarray,
) -> GuidanceMessage:
    """Build the complete three-mode guidance message."""
    if feedback_command == "SATISFIED":
        msg = GuidanceMessage(
            user_id=user_id, session_id=session_id, sample_id=sample_id,
            event_type="SESSION_SATISFIED",
            feedback_command="SATISFIED",
            reward=RewardGuidance(r_pta=1.0, confidence=1.0,
                                  explanation="User satisfied."),
            z_taste=project_taste_vector(preference_vector),
            constraints=ConstraintMask(),
            action="NONE",
        )
        msg.validate()
        return msg

    reward = compute_reward_guidance(preference_vector, feedback_command)
    constraints = build_constraints(preference_vector)
    z_taste = project_taste_vector(preference_vector)

    changed = [COMMAND_TO_DIM.get(feedback_command, "unknown")]

    msg = GuidanceMessage(
        user_id=user_id, session_id=session_id, sample_id=sample_id,
        event_type="PREFERENCE_UPDATED",
        feedback_command=feedback_command,
        changed_dimensions=changed,
        reward=reward,
        z_taste=z_taste,
        constraints=constraints,
        action="REGENERATE",
    )
    msg.validate()
    return msg
