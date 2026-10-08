"""Data schemas for PTA component. These map directly to MongoDB documents later."""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict
import uuid


# The 7-dimensional preference vector
# [brightness, compression, reverb, delay, saturation, width, warmth]
VECTOR_DIM = 7
VECTOR_LABELS = ["brightness", "compression", "reverb", "delay",
                 "saturation", "width", "warmth"]


@dataclass
class FeedbackEvent:
    """A single user feedback click."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    feedback_command: str = ""              # e.g., "MORE_REVERB"
    preference_before: List[float] = field(default_factory=list)
    preference_after: List[float] = field(default_factory=list)
    iteration: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Session:
    """A single production session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    instrument: str = ""                    # "electric_guitar", "bass", etc.
    genre: str = ""
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str = ""
    satisfied: bool = False
    iterations: int = 0
    feedback_events: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreferenceProfile:
    """A user's preference vector, scoped by instrument/genre."""
    user_id: str = ""
    instrument: str = "global"
    genre: str = "global"
    vector: List[float] = field(default_factory=lambda: [0.0] * VECTOR_DIM)
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)
