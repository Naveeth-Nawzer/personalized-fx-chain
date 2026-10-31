"""Small, framework-independent rules for a mock PTA -> M3 loop.

These helpers are deliberately narrow. Integrate them into the existing adapter/demo;
they do not replace GuidanceMessage validation or the M3 adapter.
"""

from collections.abc import Mapping
from typing import Any

TERMINAL_COMMANDS = {"SATISFIED"}
TERMINAL_EVENT_TYPES = {"SESSION_SATISFIED", "SATISFIED"}
TERMINAL_ACTIONS = {"NONE", "SESSION_CLOSED", "NO_REGENERATION"}


def _get(message: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a mapping or an object with attributes."""
    if isinstance(message, Mapping):
        return message.get(key, default)
    return getattr(message, key, default)


def should_regenerate(message: Any) -> bool:
    """Return False for terminal satisfaction messages; otherwise honor action.

    This is a guard for a mock/demo loop, not a substitute for validating the
    complete message against the agreed M3 contract.
    """
    command = str(_get(message, "feedback_command", "")).strip().upper()
    event_type = str(_get(message, "event_type", "")).strip().upper()
    action = str(_get(message, "action", "")).strip().upper()

    if command in TERMINAL_COMMANDS or event_type in TERMINAL_EVENT_TYPES:
        return False
    if action in TERMINAL_ACTIONS:
        return False
    return action == "REGENERATE"


def record_regeneration_count(current_count: int, message: Any, generated: bool) -> int:
    """Count an actual generation only when the message permits it and one occurred."""
    if current_count < 0:
        raise ValueError("current_count must be >= 0")
    if generated and not should_regenerate(message):
        raise ValueError("A terminal/non-regeneration message must not generate a sample")
    return current_count + 1 if generated else current_count
