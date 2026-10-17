"""Unit and integration tests for the PTA + XAI Interactive Terminal CLI."""
import shutil
from pathlib import Path
import pytest
import numpy as np

from pta.cli.display import (
    make_bar,
    format_single_vector_table,
    format_vector_diff_table,
    format_z_taste_table,
    format_xai_card,
    format_session_summary_report,
)
from pta.cli.session_runner import InteractiveSession
from pta.cli.app import PTAInteractiveApp
from pta.core.session_manager import SessionManager
from pta.storage.json_store import JSONPreferenceStore
from pta.models.schemas import VECTOR_DIM, VECTOR_LABELS, DEFAULT_ALPHA

TEST_DIR = "data/_test_cli_interactive"


@pytest.fixture(autouse=True)
def cleanup_test_dir():
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    yield
    shutil.rmtree(TEST_DIR, ignore_errors=True)


# --- 1. Display Formatting Unit Tests ---

def test_make_bar():
    zero_bar = make_bar(0.0)
    assert "|" in zero_bar
    pos_bar = make_bar(0.5)
    assert ">" in pos_bar
    neg_bar = make_bar(-0.5)
    assert "<" in neg_bar
    # Clipping bounds
    clipped_pos = make_bar(2.0)
    assert ">" in clipped_pos
    clipped_neg = make_bar(-2.0)
    assert "<" in clipped_neg


def test_format_single_vector_table():
    vec = [0.1, -0.2, 0.3, 0.0, 0.5, -0.6, 0.7]
    rendered = format_single_vector_table("Test Vector", vec)
    assert "Brightness" in rendered
    assert "Warmth" in rendered
    assert "0.30" in rendered


def test_format_vector_diff_table_new_and_existing():
    before = [0.0] * VECTOR_DIM
    after = [0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0]
    
    # New user diff (no prev profile)
    diff_new = format_vector_diff_table(before, after, active_dim="reverb")
    assert "Reverb" in diff_new
    assert "+0.30" in diff_new

    # Existing user diff (with prev long-term profile)
    prev = [0.0, 0.0, 0.50, 0.0, 0.0, 0.0, 0.0]
    diff_exist = format_vector_diff_table(before, after, prev_long_term=prev, active_dim="reverb")
    assert "Long-Term" in diff_exist
    assert "0.50" in diff_exist


def test_format_z_taste_table():
    z = [0.1] * 16
    rendered = format_z_taste_table(z)
    assert "16-DIMENSIONAL CONDITIONING VECTOR" in rendered
    assert "Comp x Reverb" in rendered
    assert "RMS Energy" in rendered


def test_format_xai_card_standard_and_wider():
    # Standard reverb
    card = format_xai_card(
        command="MORE_REVERB",
        active_dimension="reverb",
        before_val=0.0,
        after_val=0.30,
        explanation_text="Ambience increased.",
        alpha=DEFAULT_ALPHA,
    )
    assert "MORE_REVERB" in card
    assert "Reverb" in card
    assert "alpha=0.30" in card
    assert "Session-local" in card
    assert "NOT EXECUTED" in card

    # WIDER hardware constraint guard
    card_wider = format_xai_card(
        command="WIDER",
        active_dimension="width",
        before_val=0.0,
        after_val=0.30,
        explanation_text="Wider requested.",
    )
    assert "[!] HARDWARE CONSTRAINT: Active pipeline is MONO" in card_wider


def test_format_session_summary_report():
    report = format_session_summary_report(
        user_id="U001",
        session_id="S001",
        instrument="electric_guitar",
        genre="rock",
        start_vector=[0.0] * VECTOR_DIM,
        final_vector=[0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0],
        prev_profile=[0.0, -0.30, 0.0, 0.0, 0.0, 0.0, 0.0],
        events=[{"command": "MORE_REVERB", "active_dimension": "reverb", "delta": [0, 0, 0.30, 0, 0, 0, 0]}],
        merged_long_term=[0.0, -0.30, 0.15, 0.0, 0.0, 0.0, 0.0],
        termination_reason="SATISFIED",
    )
    assert "SESSION COMPLETED" in report
    assert "U001" in report
    assert "electric_guitar" in report
    assert "SATISFIED" in report
    assert "LONG-TERM PROFILE ACCUMULATION" in report


# --- 2. Interactive Session Runner Tests ---

def test_interactive_session_lifecycle():
    store = JSONPreferenceStore(base_dir=TEST_DIR)
    mgr = SessionManager("USER-TEST", "electric_guitar", "rock", store=store)

    # Simulated terminal inputs:
    # 1. More Reverb
    # 2. Less Compression (option 4)
    # 3. View History (option 11)
    # 4. View z_taste (option 12)
    # 5. Wider (option 10)
    # 6. Satisfied (option 13)
    inputs = iter(["1", "4", "11", "12", "10", "13"])
    outputs = []

    def fake_input(prompt=""):
        return next(inputs)

    def fake_print(*args, **kwargs):
        outputs.append(" ".join(str(a) for a in args))

    runner = InteractiveSession(
        session_mgr=mgr,
        prev_profile=None,
        input_fn=fake_input,
        print_fn=fake_print,
    )
    res = runner.run()

    assert res is not None
    assert res["satisfied"] is True
    assert res["events_count"] == 3  # MORE_REVERB, LESS_COMPRESSION, WIDER
    assert mgr.get_session_vector()[2] > 0   # reverb boosted
    assert mgr.get_session_vector()[1] < 0   # compression cut
    assert mgr.get_session_vector()[5] > 0   # width boosted

    # Verify merged profile was persisted to disk
    loaded_profile = store.load_profile("USER-TEST", "electric_guitar", "rock")
    assert loaded_profile[2] == mgr.get_session_vector()[2]
    assert loaded_profile[1] == mgr.get_session_vector()[1]


def test_interactive_session_cancel():
    store = JSONPreferenceStore(base_dir=TEST_DIR)
    mgr = SessionManager("USER-CANCEL", "electric_guitar", "rock", store=store)

    # Choice 0 (Cancel), then 'y' to confirm
    inputs = iter(["0", "y"])
    runner = InteractiveSession(
        session_mgr=mgr,
        input_fn=lambda prompt="": next(inputs),
        print_fn=lambda *a, **k: None,
    )
    res = runner.run()
    assert res is None
    # Ensure profile was not merged
    assert not mgr.session.satisfied


# --- 3. Full App Workflow Tests (New User + Existing User + Multi-Session Zero-Start) ---

def test_app_multi_session_starts_neutral_and_accumulates():
    """
    Test the multi-session architectural invariant:
    Session 1: Create USER-001 -> Starts neutral [0]*7 -> Clicks MORE_REVERB -> SATISFIED.
               Profile saved with Reverb = 0.30.
    Session 2: Relaunch app -> Select USER-001 as existing user.
               CRITICAL INVARIANT: Session 2 session-vector starts at ALL ZEROS [0]*7!
               Clicks MORE_DELAY -> SATISFIED.
               Long-term profile blends Delay and preserves Reverb without decay!
    Session 3: Relaunch app -> Select USER-001 again.
               CRITICAL INVARIANT: Session 3 session-vector starts at ALL ZEROS [0]*7!
    """
    # === SESSION 1 ===
    inputs_s1 = iter([
        "1",               # Main menu: Create new user
        "USER-001",        # User ID
        "electric_guitar", # Instrument
        "rock",            # Genre
        "1",               # Feedback: MORE_REVERB (0.0 -> 0.30)
        "13",              # Feedback: SATISFIED
        "5",               # Main menu: Exit
    ])
    printed_s1 = []

    app1 = PTAInteractiveApp(
        base_dir=TEST_DIR,
        input_fn=lambda p="": next(inputs_s1),
        print_fn=lambda *a, **k: printed_s1.append(" ".join(str(x) for x in a)),
    )
    app1.run()

    # Verify profile after Session 1: Initial baseline established
    store1 = JSONPreferenceStore(base_dir=TEST_DIR)
    p1 = store1.load_profile("USER-001", "electric_guitar", "rock")
    assert p1[2] == pytest.approx(0.30, abs=1e-5)  # Reverb
    assert p1[3] == 0.0                             # Delay untouched

    # === SESSION 2: RELAUNCH APP (Simulating process restart for existing user) ===
    inputs_s2 = iter([
        "2",               # Main menu: Continue as existing user
        "1",               # Select User #1 (USER-001)
        "5",               # Feedback: MORE_DELAY (0.0 -> 0.30 in session vector)
        "13",              # Feedback: SATISFIED
        "5",               # Main menu: Exit
    ])
    printed_s2 = []

    app2 = PTAInteractiveApp(
        base_dir=TEST_DIR,
        input_fn=lambda p="": next(inputs_s2),
        print_fn=lambda *a, **k: printed_s2.append(" ".join(str(x) for x in a)),
    )
    app2.run()

    joined_s2 = "\n".join(printed_s2)
    # Check that Session 2 explicitly announced neutral start alongside saved profile
    assert "Saved Long-Term Profile (Prior Learning Across Sessions)" in joined_s2
    assert "New Session Preference Vector (Starts Neutral at 0.00)" in joined_s2

    # Verify profile after Session 2: Selective dimension preservation
    store2 = JSONPreferenceStore(base_dir=TEST_DIR)
    p2 = store2.load_profile("USER-001", "electric_guitar", "rock")
    # Reverb was untouched in Session 2, so it MUST NOT decay:
    assert p2[2] == pytest.approx(0.30, abs=1e-5)
    # Delay was touched in Session 2 (weight=0.50 -> (1-0.5)*0.0 + 0.5*0.30 = 0.15):
    assert p2[3] == pytest.approx(0.15, abs=1e-5)


def test_app_list_users_and_system_status():
    inputs = iter([
        "3",   # List users (empty)
        "4",   # System status
        "",    # Enter to return from system status
        "5",   # Exit
    ])
    outputs = []
    app = PTAInteractiveApp(
        base_dir=TEST_DIR,
        input_fn=lambda p="": next(inputs),
        print_fn=lambda *a, **k: outputs.append(" ".join(str(x) for x in a)),
    )
    app.run()
    joined = "\n".join(outputs)
    assert "SAVED USER PROFILES REPOSITORY" in joined
    assert "SYSTEM SPECIFICATIONS & XAI GROUNDING INTEGRITY" in joined
    assert "alpha=0.30" in joined
    assert "Every new session starts with all zeros" in joined
