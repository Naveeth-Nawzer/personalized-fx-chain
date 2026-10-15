"""Data schemas for PTA component. These map directly to MongoDB documents later."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict
import uuid


# The 7-dimensional preference vector
# [brightness, compression, reverb, delay, saturation, width, warmth]
VECTOR_DIM = 7
VECTOR_LABELS = ["brightness", "compression", "reverb", "delay",
                 "saturation", "width", "warmth"]


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format without deprecation."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FeedbackEvent:
    """A single user feedback click."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: str = ""
    timestamp: str = field(default_factory=utc_now_iso)
    feedback_command: str = ""              # e.g., "MORE_REVERB"
    preference_before: List[float] = field(default_factory=list)
    preference_after: List[float] = field(default_factory=list)
    iteration: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionPreference:
    """Derived preference state for a single session."""
    session_id: str = ""
    user_id: str = ""
    instrument: str = ""
    genre: str = ""
    vector: List[float] = field(default_factory=lambda: [0.0] * VECTOR_DIM)
    iterations: int = 0
    satisfied: bool = False
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str = ""
    feedback_commands: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Session:
    """A single production session, now with real lifecycle."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    instrument: str = ""                    # "electric_guitar", "bass", etc.
    genre: str = ""
    started_at: str = field(default_factory=utc_now_iso)
    completed_at: str = ""
    satisfied: bool = False
    iterations: int = 0
    feedback_events: List[dict] = field(default_factory=list)

    # Session-scoped preference state
    session_vector: List[float] = field(default_factory=lambda: [0.0] * VECTOR_DIM)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_preference(self) -> SessionPreference:
        """Convert to a SessionPreference summary."""
        return SessionPreference(
            session_id=self.session_id,
            user_id=self.user_id,
            instrument=self.instrument,
            genre=self.genre,
            vector=self.session_vector.copy(),
            iterations=self.iterations,
            satisfied=self.satisfied,
            started_at=self.started_at,
            ended_at=self.completed_at,
            feedback_commands=[e["command"] for e in self.feedback_events],
        )


@dataclass
class PreferenceProfile:
    """A user's preference vector, scoped by instrument/genre."""
    user_id: str = ""
    instrument: str = "global"
    genre: str = "global"
    vector: List[float] = field(default_factory=lambda: [0.0] * VECTOR_DIM)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)
