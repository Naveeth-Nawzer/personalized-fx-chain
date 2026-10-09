"""Template-based, decision-grounded explanations. Grounded in preference updates."""

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
        "Preference for wider stereo image recorded. Side-channel weighting target increased."
    ),
}


def generate_explanation(feedback_command: str,
                         preference_before: list,
                         preference_after: list,
                         param_changes: dict = None) -> dict:
    """
    Build an explanation from a feedback event.

    Returns a dict so it can be stored in MongoDB / JSON as a document later.
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
        # Append concrete numbers to make the explanation auditable if real DSP changes occur
        for key, change in param_changes.items():
            explanation["text"] += (
                f" {key}: {change['before']:.2f} -> {change['after']:.2f}."
            )

    return explanation
