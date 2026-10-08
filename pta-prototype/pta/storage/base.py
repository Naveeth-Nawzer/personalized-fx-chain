"""Storage interface — swap JSON for MongoDB later without changing core code."""
from abc import ABC, abstractmethod


class PreferenceStore(ABC):
    """Abstract storage for PTA. Implementations: JSON, MongoDB, PostgreSQL."""

    @abstractmethod
    def save_profile(self, user_id: str, vector: list,
                     instrument: str = "global",
                     genre: str = "global") -> None: ...

    @abstractmethod
    def load_profile(self, user_id: str, instrument: str = "global",
                     genre: str = "global") -> list: ...

    @abstractmethod
    def log_feedback(self, event: dict) -> None: ...

    @abstractmethod
    def get_feedback_history(self, session_id: str) -> list: ...
