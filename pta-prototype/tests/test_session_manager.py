import shutil
import pytest
import numpy as np
from pta.core.session_manager import SessionManager
from pta.models.schemas import VECTOR_DIM
from pta.storage.json_store import JSONPreferenceStore

TEST_DIR = "data/_test_session"


def setup_function(_):
    shutil.rmtree(TEST_DIR, ignore_errors=True)


def teardown_function(_):
    shutil.rmtree(TEST_DIR, ignore_errors=True)


# 1. Neutral session
def test_neutral_session():
    mgr = SessionManager("U001")
    assert np.all(mgr.get_session_vector() == 0.0)


# 2. Repeated feedback (with approx)
def test_repeated_reverb_converges():
    mgr = SessionManager("U001", alpha=0.3)
    mgr.submit_feedback("MORE_REVERB")
    mgr.submit_feedback("MORE_REVERB")
    assert mgr.get_session_vector()[2] == pytest.approx(0.51, abs=1e-6)


# 3. Satisfaction doesn't change vector
def test_satisfied_is_terminal_no_vector_change():
    mgr = SessionManager("U001")
    mgr.submit_feedback("MORE_REVERB")
    before = mgr.get_session_vector().copy()
    mgr.submit_feedback("SATISFIED")
    after = mgr.get_session_vector()
    assert np.array_equal(before, after)


# 4. Post-satisfaction feedback raises
def test_post_satisfaction_feedback_raises():
    mgr = SessionManager("U001")
    mgr.submit_feedback("SATISFIED")
    with pytest.raises(RuntimeError):
        mgr.submit_feedback("MORE_REVERB")


# 5. Session persistence
def test_session_persisted():
    store = JSONPreferenceStore(base_dir=TEST_DIR)
    mgr = SessionManager("U001", "electric_guitar", store=store)
    mgr.submit_feedback("MORE_REVERB")
    mgr.submit_feedback("SATISFIED")

    loaded = store.load_session("U001", mgr.session.session_id)
    assert loaded is not None
    assert loaded["satisfied"] is True
    assert loaded["iterations"] == 1     # SATISFIED does not increment


# 6. Long-term merge — no existing profile
def test_merge_when_no_profile_exists():
    store = JSONPreferenceStore(base_dir=TEST_DIR)
    mgr = SessionManager("U001", "electric_guitar", store=store)
    mgr.submit_feedback("MORE_REVERB")
    mgr.submit_feedback("SATISFIED")
    merged = mgr.merge_into_long_term(weight=1.0)
    assert merged[2] == pytest.approx(0.30, abs=1e-6)


# 7. Two-session accumulation with selective dimension preservation
def test_two_sessions_accumulate():
    store = JSONPreferenceStore(base_dir=TEST_DIR)

    s1 = SessionManager("U001", "electric_guitar", store=store)
    s1.submit_feedback("MORE_REVERB")
    s1.submit_feedback("SATISFIED")
    s1.merge_into_long_term(weight=0.5)

    s2 = SessionManager("U001", "electric_guitar", store=store)
    assert np.all(s2.get_session_vector() == 0.0)     # session 2 neutral

    s2.submit_feedback("MORE_COMPRESSION")            # positive compression
    s2.submit_feedback("SATISFIED")
    s2.merge_into_long_term(weight=0.5)

    final = store.load_profile("U001", "electric_guitar")
    assert final[2] > 0        # reverb preserved from session 1 without accidental decay
    assert final[1] > 0        # positive compression


# 8. Duplicate merge prevention
def test_duplicate_merge_rejected():
    store = JSONPreferenceStore(base_dir=TEST_DIR)
    mgr = SessionManager("U001", "electric_guitar", store=store)
    mgr.submit_feedback("MORE_REVERB")
    mgr.submit_feedback("SATISFIED")
    mgr.merge_into_long_term(weight=0.5)
    with pytest.raises(RuntimeError):
        mgr.merge_into_long_term(weight=0.5)


# 9. Invalid merge weight
def test_invalid_merge_weight_rejected():
    store = JSONPreferenceStore(base_dir=TEST_DIR)
    mgr = SessionManager("U001", store=store)
    mgr.submit_feedback("SATISFIED")
    with pytest.raises(ValueError):
        mgr.merge_into_long_term(weight=1.5)
    with pytest.raises(ValueError):
        mgr.merge_into_long_term(weight=-0.1)


# 10. User isolation
def test_user_isolation():
    store = JSONPreferenceStore(base_dir=TEST_DIR)

    u1 = SessionManager("U001", "electric_guitar", store=store)
    u1.submit_feedback("MORE_REVERB")
    u1.submit_feedback("SATISFIED")
    u1.merge_into_long_term(weight=1.0)

    u2 = SessionManager("U002", "electric_guitar", store=store)
    u2.submit_feedback("MORE_COMPRESSION")
    u2.submit_feedback("SATISFIED")
    u2.merge_into_long_term(weight=1.0)

    p1 = store.load_profile("U001", "electric_guitar")
    p2 = store.load_profile("U002", "electric_guitar")
    assert p1[2] > 0 and p1[1] == 0    # U001 only has reverb
    assert p2[1] > 0 and p2[2] == 0    # U002 only has compression


# 11. Untouched dimensions do not decay in long-term merge
def test_untouched_dimensions_do_not_decay():
    store = JSONPreferenceStore(base_dir=TEST_DIR)

    # Session 1: establish reverb preference
    s1 = SessionManager("U001", "electric_guitar", store=store)
    s1.submit_feedback("MORE_REVERB")
    s1.submit_feedback("SATISFIED")
    s1.merge_into_long_term(weight=1.0)  # reverb = 0.30

    reverb_before = store.load_profile("U001", "electric_guitar")[2]

    # Session 2: only expresses delay and width (does not touch reverb)
    s2 = SessionManager("U001", "electric_guitar", store=store)
    s2.submit_feedback("MORE_DELAY")
    s2.submit_feedback("WIDER")
    s2.submit_feedback("SATISFIED")
    s2.merge_into_long_term(weight=0.5)

    final = store.load_profile("U001", "electric_guitar")
    # Reverb must remain EXACTLY unchanged because Session 2 expressed no opinion on reverb
    assert final[2] == pytest.approx(reverb_before, abs=1e-6)
    assert final[3] > 0   # delay updated
    assert final[5] > 0   # width updated
