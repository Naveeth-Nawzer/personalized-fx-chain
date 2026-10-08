"""JSON-file implementation of PreferenceStore (for prototyping)."""
import json
import os
from pathlib import Path
from pta.storage.base import PreferenceStore
from pta.models.schemas import VECTOR_DIM


class JSONPreferenceStore(PreferenceStore):

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        (self.base_dir / "users").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "feedback").mkdir(parents=True, exist_ok=True)

    def _profile_path(self, user_id, instrument, genre):
        return self.base_dir / "users" / f"{user_id}__{instrument}__{genre}.json"

    def save_profile(self, user_id, vector, instrument="global", genre="global"):
        path = self._profile_path(user_id, instrument, genre)
        data = {
            "user_id": user_id,
            "instrument": instrument,
            "genre": genre,
            "vector": vector,
        }
        path.write_text(json.dumps(data, indent=2))

    def load_profile(self, user_id, instrument="global", genre="global"):
        path = self._profile_path(user_id, instrument, genre)
        if not path.exists():
            return [0.0] * VECTOR_DIM
        return json.loads(path.read_text())["vector"]

    def log_feedback(self, event: dict):
        session_id = event.get("session_id", "unknown")
        path = self.base_dir / "feedback" / f"{session_id}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def get_feedback_history(self, session_id: str) -> list:
        path = self.base_dir / "feedback" / f"{session_id}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]
