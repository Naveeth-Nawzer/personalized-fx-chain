"""Template-based, decision-grounded explanations. Grounded in preference updates."""
from typing import Dict, List, Optional, Any

TEMPLATES = {
    "MORE_REVERB": (
        "Preference for more reverb recorded. The ambience preference strength "
        "has increased and is queued to guide the next FX-chain generation."
    ),
    "LESS_REVERB": (
        "Preference for less reverb recorded. The ambience preference has decreased "
        "toward a tighter, drier target for future generation."
    ),
    "MORE_COMPRESSION": (
        "Preference for more compression recorded. Target dynamics profile tightened "
        "to guide subsequent processor parameter bounds."
    ),
    "LESS_COMPRESSION": (
        "Preference for less compression recorded. Target dynamics expanded "
        "toward more natural dynamics."
    ),
    "MORE_DELAY": (
        "Preference for more delay recorded. Echo prominence and feedback repeats "
        "target increased for upcoming chain adjustments."
    ),
    "LESS_DELAY": (
        "Preference for less delay recorded. Delay wet mix and repeat contributions decreased."
    ),
    "BRIGHTER": (
        "Preference for brighter tone recorded. High-frequency balance target "
        "increased for the FX recommendation model."
    ),
    "WARMER": (
        "Preference for warmer tone recorded. Low-mid emphasis increased and "
        "highs softened for upcoming chain adjustments."
    ),
    "MORE_SATURATION": (
        "Preference for more saturation recorded. Harmonic drive preference increased."
    ),
    "WIDER": (
        "Preference for wider stereo image recorded in profile. Note: the active "
        "rendering pipeline is mono (channels=1); spatial widening is unsupported "
        "until stereo DAPN is enabled."
    ),
    "SATISFIED": (
        "User marked generation as satisfactory. Current session closed with no "
        "further parameter updates."
    ),
}


def generate_explanation(
    feedback_command: str,
    preference_before: List[float],
    preference_after: List[float],
    param_changes: Optional[Dict[str, Any]] = None,
    applied_constraints: Optional[Dict[str, Any]] = None,
    generation_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a decision-grounded, truthful explanation from a feedback event and generation outcome.

    Returns a dict so it can be stored in MongoDB / JSON as an auditable document.
    """
    base = TEMPLATES.get(feedback_command, "Preference updated.")

    explanation = {
        "command": feedback_command,
        "text": base,
        "preference_before": preference_before,
        "preference_after": preference_after,
        "preference_delta": [a - b for a, b in zip(preference_after, preference_before)],
    }

    if param_changes:
        explanation["parameter_changes"] = param_changes
        for key, change in param_changes.items():
            explanation["text"] += (
                f" {key}: {change['before']:.2f} -> {change['after']:.2f}."
            )

    if applied_constraints:
        prohibited = applied_constraints.get("prohibited_effects", [])
        bounds = applied_constraints.get("parameter_bounds", {})
        if prohibited:
            explanation["prohibited_effects"] = prohibited
            explanation["text"] += f" Prohibited effects enforced: {prohibited}."
        if bounds:
            explanation["active_parameter_bounds"] = bounds

    if generation_result:
        chain = generation_result.get("chain", [])
        if chain:
            effects = [e.get("effect", "Unknown") for e in chain]
            explanation["generated_chain"] = effects
            explanation["text"] += f" Generated chain: {' -> '.join(effects)}."

    return explanation
