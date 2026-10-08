import numpy as np
from pta.core.preference_engine import PreferenceVectorEngine


def test_starts_neutral():
    engine = PreferenceVectorEngine("U001")
    assert np.all(engine.get_vector() == 0.0)


def test_single_update():
    engine = PreferenceVectorEngine("U001", alpha=0.3)
    result = engine.update("MORE_REVERB")
    assert result["after"][2] == 0.3     # reverb dimension = +0.3


def test_repeated_update_converges():
    """Two MORE_REVERB clicks should push reverb toward +1."""
    engine = PreferenceVectorEngine("U001", alpha=0.5)
    engine.update("MORE_REVERB")   # 0.5
    engine.update("MORE_REVERB")   # 0.75
    assert engine.get_vector()[2] == 0.75


def test_contradictory_feedback():
    """
    Scenario :
    Iteration 1: MORE_REVERB   -> +0.3
    Iteration 2: MORE_REVERB   -> +0.51
    Iteration 3: LESS_REVERB   -> back down
    """
    engine = PreferenceVectorEngine("U001", alpha=0.3)
    engine.update("MORE_REVERB")
    engine.update("MORE_REVERB")
    engine.update("LESS_REVERB")
    reverb = engine.get_vector()[2]
    assert 0.0 < reverb < 0.3    # net positive but reduced


def test_vector_is_clipped():
    """Preference vector must stay in [-1, +1]."""
    engine = PreferenceVectorEngine("U001", alpha=0.9)
    for _ in range(20):
        engine.update("MORE_REVERB")
    assert engine.get_vector()[2] <= 1.0


def test_reset():
    engine = PreferenceVectorEngine("U001")
    engine.update("MORE_REVERB")
    engine.reset()
    assert np.all(engine.get_vector() == 0.0)

def test_independent_dimensions():
    """Updating one dimension should not decay others."""
    engine = PreferenceVectorEngine("U001", alpha=0.3)
    engine.update("MORE_REVERB")
    reverb_val = engine.get_vector()[2]
    
    engine.update("WIDER")
    assert engine.get_vector()[2] == reverb_val  # Reverb should not have changed
