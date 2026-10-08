"""Template-based, decision-grounded explanations. No LLM, no hallucination."""

TEMPLATES = {
    "MORE_REVERB": (
        "Reverb increased. Your preference for more ambience was applied "
        "by raising the reverb wet/dry mix and extending the decay tail."
    ),
    "LESS_REVERB": (
        "Reverb reduced. Your preference for a tighter, drier sound was applied "
        "by lowering the wet/dry mix."
    ),
    "MORE_COMPRESSION": (
        "Compression increased. Your preference for a tighter dynamic profile "
        "was applied by lowering the threshold and raising the ratio."
    ),
    "LESS_COMPRESSION": (
        "Compression reduced. Your preference for more natural dynamics was "
        "applied by raising the threshold and lowering the ratio."
    ),
    "BRIGHTER": (
        "Brightness increased. The high-frequency balance was adjusted to "
        "produce a brighter tonal character."
    ),
    "WARMER": (
        "Warmth increased. Low-mid body was reinforced and harsh highs softened."
    ),
    "MORE_SATURATION": (
        "Saturation increased. Harmonic coloration was added to the signal."
    ),
    "WIDER": (
        "Stereo width increased. Side-channel energy was boosted."
    ),
}


def generate_explanation(feedback_command: str,
                         preference_before: list,
                         preference_after: list,
                         param_changes: dict = None) -> dict:
    """
    Build an explanation from a feedback event.

    Returns a dict so it can be stored in MongoDB as a document later.
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
        # Append concrete numbers to make the explanation auditable
        for key, change in param_changes.items():
            explanation["text"] += (
                f" {key}: {change['before']:.2f} -> {change['after']:.2f}."
            )

    return explanation
