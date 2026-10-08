import shutil
from pathlib import Path
from pta.storage.json_store import JSONPreferenceStore


def test_save_and_load():
    store = JSONPreferenceStore(base_dir="data/_test")
    store.save_profile("U001", [0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0])
    loaded = store.load_profile("U001")
    assert loaded[2] == 0.7
    shutil.rmtree("data/_test", ignore_errors=True)


def test_unknown_user_returns_zeros():
    store = JSONPreferenceStore(base_dir="data/_test")
    loaded = store.load_profile("NOBODY")
    assert loaded == [0.0] * 7
    shutil.rmtree("data/_test", ignore_errors=True)


def test_feedback_logging():
    store = JSONPreferenceStore(base_dir="data/_test")
    store.log_feedback({"session_id": "S1", "command": "MORE_REVERB"})
    store.log_feedback({"session_id": "S1", "command": "LESS_COMPRESSION"})
    history = store.get_feedback_history("S1")
    assert len(history) == 2
    shutil.rmtree("data/_test", ignore_errors=True)
