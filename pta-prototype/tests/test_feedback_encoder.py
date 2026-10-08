import numpy as np
from pta.core.feedback_encoder import encode_feedback, FEEDBACK_CONTROLS


def test_more_reverb():
    vec = encode_feedback("MORE_REVERB")
    assert vec[2] == 1.0
    assert np.sum(np.abs(vec)) == 1.0


def test_less_compression():
    vec = encode_feedback("LESS_COMPRESSION")
    assert vec[1] == -1.0


def test_satisfied_is_terminal():
    vec = encode_feedback("SATISFIED")
    assert np.all(vec == 0.0)


def test_unknown_command_raises():
    try:
        encode_feedback("MAKE_IT_SOUND_BETTER")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_all_controls_work():
    for cmd in FEEDBACK_CONTROLS:
        encode_feedback(cmd)   # should not raise
