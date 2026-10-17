"""Manage session lifecycle, aggregation, and profile merging."""
from datetime import datetime, timezone
from typing import Optional
import uuid
import numpy as np
from pta.core.preference_engine import PreferenceVectorEngine
from pta.core.explanation_engine import generate_explanation
from pta.models.schemas import Session, SessionPreference, VECTOR_DIM


class SessionManager:
    """
    Owns a single production session.

    Responsibilities:
      - Accumulate feedback into a SESSION-scoped preference vector
      - Associate feedback events with session and generation IDs
      - Merge the session vector into the LONG-TERM profile on completion
      - Selective dimension merge: untouched dimensions in a session do not decay
      - Log every feedback event with per-event timestamps and generation tracking
      - Treat SATISFIED as a terminal action without preference updates
    """

    def __init__(self, user_id: str, instrument: str = "global",
                 genre: str = "global", alpha: float = 0.3,
                 store=None):
        self.session = Session(
            user_id=user_id,
            instrument=instrument,
            genre=genre,
        )
        self.alpha = alpha
        self.store = store
        self.engine = PreferenceVectorEngine(
            user_id=user_id,
            alpha=alpha,
            initial_vector=np.zeros(VECTOR_DIM),   # session starts neutral
        )
        self._merged = False
        self.current_generation_id: Optional[str] = None

    def set_current_generation(self, generation_id: str) -> None:
        """Track the currently active generation ID."""
        self.current_generation_id = generation_id

    def submit_feedback(self, command: str, generation_id: Optional[str] = None) -> dict:
        """
        Process a feedback click inside this session.

        Associates the feedback event with generation_id (if provided or active).
        """
        if self.session.satisfied:
            raise RuntimeError("Session already marked satisfied.")

        gen_id = generation_id or self.current_generation_id

        if command == "SATISFIED":
            return self._handle_satisfied(generation_id=gen_id)

        event_id = str(uuid.uuid4())
        result = self.engine.update(command, event_id=event_id)
        now_ts = datetime.now(timezone.utc).isoformat()

        self.session.iterations += 1
        self.session.session_vector = self.engine.get_vector().tolist()
        event_record = {
            "event_id": event_id,
            "generation_id": gen_id,
            "iteration": self.session.iterations,
            "command": command,
            "active_dimension": result.get("active_dimension"),
            "before": result["before"],
            "after": result["after"],
            "delta": result["delta"],
            "timestamp": now_ts,
        }
        self.session.feedback_events.append(event_record)

        explanation = generate_explanation(
            feedback_command=command,
            preference_before=result["before"],
            preference_after=result["after"],
        )

        if self.store:
            log_entry = {
                "session_id": self.session.session_id,
                "user_id": self.session.user_id,
                **event_record,
            }
            self.store.log_feedback(log_entry)

        return {
            "session_id": self.session.session_id,
            "generation_id": gen_id,
            "event_id": event_id,
            "iteration": self.session.iterations,
            "active_dimension": result.get("active_dimension"),
            "preference_before": result["before"],
            "preference_after": result["after"],
            "explanation": explanation["text"],
        }

    def _handle_satisfied(self, generation_id: Optional[str] = None) -> dict:
        """Terminal action: no preference change, no iteration increment."""
        self.session.satisfied = True
        self.session.completed_at = datetime.now(timezone.utc).isoformat()

        if self.store:
            self.store.save_session(self.session.to_preference())

        return {
            "session_id": self.session.session_id,
            "generation_id": generation_id or self.current_generation_id,
            "iteration": self.session.iterations,
            "preference_before": self.engine.get_vector().tolist(),
            "preference_after": self.engine.get_vector().tolist(),
            "explanation": "Session marked satisfied. No further adjustments applied.",
            "satisfied": True,
        }

    def get_session_preference(self) -> SessionPreference:
        return self.session.to_preference()

    def get_session_vector(self) -> np.ndarray:
        return self.engine.get_vector()

    def merge_into_long_term(self, weight: float = 0.3):
        """
        Merge the session vector into the user's long-term profile.
        Called on 'SATISFIED' or explicit user action.

        weight controls how much this session influences the long-term profile.
        Crucially, only dimensions that received feedback during this session
        are blended; untouched dimensions in this session do NOT decay.
        """
        if not (0.0 <= weight <= 1.0):
            raise ValueError("weight must be in [0, 1]")

        if not self.session.satisfied:
            raise RuntimeError("Cannot merge unfinished session.")

        if getattr(self, "_merged", False):
            raise RuntimeError("Session already merged into long-term profile.")

        if not self.store:
            return None

        long_term = self.store.load_profile(
            self.session.user_id,
            self.session.instrument,
            self.session.genre,
        )

        if not long_term or len(long_term) != VECTOR_DIM:
            long_term = [0.0] * VECTOR_DIM

        lt_vec = np.array(long_term, dtype=float)
        session_vec = self.engine.get_vector()

        # Identify active dimensions in this session
        active_mask = session_vec != 0.0

        merged = lt_vec.copy()
        if np.any(active_mask):
            merged[active_mask] = (1.0 - weight) * lt_vec[active_mask] + weight * session_vec[active_mask]
            merged = np.clip(merged, -1.0, 1.0)

        self.store.save_profile(
            self.session.user_id,
            merged.tolist(),
            self.session.instrument,
            self.session.genre,
        )
        self._merged = True
        return merged.tolist()
