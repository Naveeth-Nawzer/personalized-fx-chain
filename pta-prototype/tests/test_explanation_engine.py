"""Tests for the decision-grounded explainability engine."""
from pta.core.explanation_engine import generate_explanation


def test_reverb_explanation_contains_expected_text():
    before = [0.0] * 7
    after = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
    exp = generate_explanation("MORE_REVERB", before, after)
    assert exp["command"] == "MORE_REVERB"
    assert "ambience preference strength has increased" in exp["text"]
    assert exp["preference_delta"][2] == 0.3


def test_wider_explanation_truthfully_reports_mono_limitation():
    before = [0.0] * 7
    after = [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0]
    exp = generate_explanation("WIDER", before, after)
    assert "mono (channels=1)" in exp["text"]
    assert "spatial widening is unsupported" in exp["text"]


def test_satisfied_explanation():
    before = [0.0, -0.3, 0.51, 0.0, 0.0, 0.0, 0.0]
    after = [0.0, -0.3, 0.51, 0.0, 0.0, 0.0, 0.0]
    exp = generate_explanation("SATISFIED", before, after)
    assert "User marked generation as satisfactory" in exp["text"]
    assert all(d == 0.0 for d in exp["preference_delta"])


def test_delay_explanations():
    before = [0.0] * 7
    after_more = [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]
    exp_more = generate_explanation("MORE_DELAY", before, after_more)
    assert "Echo prominence and feedback repeats target increased" in exp_more["text"]

    after_less = [0.0, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0]
    exp_less = generate_explanation("LESS_DELAY", before, after_less)
    assert "Delay wet mix and repeat contributions decreased" in exp_less["text"]


def test_explanation_grounded_in_constraints_and_chain():
    before = [0.0] * 7
    after = [0.0, -0.3, 0.5, 0.0, 0.0, 0.0, 0.0]
    constraints = {"prohibited_effects": ["Distortion"]}
    generation = {"chain": [{"effect": "Compressor"}, {"effect": "Reverb"}]}

    exp = generate_explanation(
        feedback_command="MORE_REVERB",
        preference_before=before,
        preference_after=after,
        applied_constraints=constraints,
        generation_result=generation,
    )

    assert "Prohibited effects enforced: ['Distortion']" in exp["text"]
    assert "Generated chain: Compressor -> Reverb" in exp["text"]
    assert exp["generated_chain"] == ["Compressor", "Reverb"]
    assert exp["prohibited_effects"] == ["Distortion"]
