"""Template-based and evidence-grounded Explainable AI (XAI) engine.

Supports three distinct levels of explanation:
1. Level 1 (PTA): Why the preference profile changed (before/after, delta, active dimension).
2. Level 2 (GRRL Decision): What effects the agent selected, parameter diffs, and enforced constraints.
3. Level 3 (Audio Outcome): Truthful reporting of DAPN rendering status and verified acoustic measurements
   (never claiming acoustic perceptual shifts without empirical metrics).
"""
from typing import Any, Dict, List, Optional
from pta.core.decision_trace import DecisionTrace


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


def explain_preference_update(
    command: str,
    dimension: str,
    before: float,
    after: float,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Explain why and how a preference dimension was updated following user feedback.

    Evidence type: 'preference_update_record'.
    """
    delta = float(after) - float(before)
    explanation = {
        "type": "preference_update",
        "command": command,
        "dimension": dimension,
        "before": float(before),
        "after": float(after),
        "delta": float(delta),
        "text": (
            f"Your {dimension} preference changed "
            f"from {before:.2f} to {after:.2f} "
            f"after feedback {command}."
        ),
        "evidence": "preference_update_record",
    }
    if event_id:
        explanation["event_id"] = event_id
    return explanation


def explain_generation(trace: DecisionTrace) -> Dict[str, Any]:
    """
    Build a decision-trace grounded explanation of an FX generation and audio outcome.

    Strictly adheres to evidence-based truthfulness:
    - If rendering failed or was not run, does NOT claim an audio outcome.
    - If an effect was prohibited, explicitly reports whether the constraint was enforced.
    - If parameters changed, cites actual parameter diffs from the trace.
    - If audio was rendered without measurements, avoids unfounded acoustic claims.
    """
    if trace.rendering_status == "failed":
        return {
            "type": "generation",
            "status": "failed",
            "session_id": trace.session_id,
            "generation_id": trace.generation_id,
            "text": (
                "The generation was not confirmed as successfully rendered. "
                "Its audio outcome cannot yet be explained."
            ),
            "evidence": "generation_status",
        }

    if trace.rendering_status == "not_run":
        return {
            "type": "generation",
            "status": "not_run",
            "session_id": trace.session_id,
            "generation_id": trace.generation_id,
            "text": (
                "The generation has not been rendered yet. "
                "Audio outcome cannot be verified."
            ),
            "evidence": "generation_status",
        }

    # Case: rendering_status == "success"
    text_parts: List[str] = []

    # 1. Terminal / Satisfaction handling
    if trace.feedback_command == "SATISFIED":
        text_parts.append(
            "User marked generation as satisfactory. Current session closed with no "
            "further parameter updates."
        )

    # 2. Preference and command context
    if trace.feedback_command and trace.feedback_command != "SATISFIED":
        base_tmpl = TEMPLATES.get(trace.feedback_command)
        if base_tmpl:
            text_parts.append(base_tmpl)

    # 3. Selected chain
    selected_effects = trace.get_selected_effect_names()
    if selected_effects:
        text_parts.append(f"Generated chain: {' -> '.join(selected_effects)}.")
    else:
        text_parts.append("Generated chain is empty.")

    # 4. Prohibited effects check
    constraints = trace.guidance_sent.get("constraints", {})
    prohibited = constraints.get("prohibited_effects", [])
    if prohibited:
        enforced = [eff for eff in prohibited if eff not in selected_effects]
        violated = [eff for eff in prohibited if eff in selected_effects]
        if enforced:
            text_parts.append(f"Prohibited effects enforced: {enforced}.")
        if violated:
            text_parts.append(f"WARNING: Prohibited effects violated by agent: {violated}.")

    # 5. Parameter deltas
    param_diffs = trace.get_parameter_diff()
    if param_diffs:
        diff_strs = [
            f"{d['effect']}.{d['parameter']}: {d['before']:.2f} -> {d['after']:.2f}"
            for d in param_diffs
        ]
        text_parts.append(f"Parameter adjustments applied: {', '.join(diff_strs)}.")

    # 6. Unsupported requests (e.g. mono width)
    if trace.unsupported_requests:
        text_parts.append(
            f"Unsupported pipeline capabilities noted: {'; '.join(trace.unsupported_requests)}."
        )

    # 7. Audio measurements outcome
    if trace.has_audio_measurements():
        metric_strs = [f"{k}={v:.2f}" for k, v in trace.audio_metrics.items()]
        text_parts.append(f"Measured audio metrics: {', '.join(metric_strs)}.")
    else:
        text_parts.append(
            "Audio rendered successfully, but objective acoustic measurements "
            "are not available for this sample."
        )

    return {
        "type": "generation",
        "status": "success",
        "session_id": trace.session_id,
        "generation_id": trace.generation_id,
        "feedback_event_id": trace.feedback_event_id,
        "feedback_command": trace.feedback_command,
        "text": " ".join(text_parts),
        "selected_chain": selected_effects,
        "parameter_changes": param_diffs,
        "prohibited_effects": prohibited,
        "audio_ref": trace.audio_ref,
        "audio_metrics": trace.audio_metrics,
        "evidence": "decision_trace",
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
    Build a decision-grounded explanation from a feedback event and generation outcome.

    Maintained for backward compatibility with existing session and adapter workflows.
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
