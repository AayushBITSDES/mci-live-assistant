"""ContextManager - in-memory state + derived facts.

Single source of truth for runtime state. Mutated only by event appends
through `record_event()`. Read by every LLM call via the prompt builder.

The state dict and deque are intentionally small. This is not a database;
it is a working memory snapshot. Deeper history lives in the JSONL log.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class RiskWindow:
    """A bounded interval during which a particular risk is being monitored.

    Example: stove turned on -> open RiskWindow{scenario="stove_unattended"}.
    Closed when person returns OR when scenario is resolved by user.
    """
    id: str
    scenario: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "active"          # active | resolved | closed
    associated_events: list[str] = field(default_factory=list)


@dataclass
class ContextState:
    """The full runtime state that gets injected into LLM prompts.

    Every field here is a *derived fact* - a conclusion the system has
    reached from raw events. Keep this small. If a field can be re-derived
    cheaply on demand, it does not belong here.
    """
    last_medication_time: Optional[datetime] = None
    current_activity: Optional[str] = None      # e.g. "making_tea", "watching_tv"
    activity_start: Optional[datetime] = None
    open_risk_windows: list[RiskWindow] = field(default_factory=list)
    last_known_persons: list[str] = field(default_factory=list)
    recent_events: deque = field(default_factory=lambda: deque(maxlen=15))


class ContextManager:
    """Wraps a ContextState and applies events to update derived facts."""

    def __init__(self) -> None:
        self.state = ContextState()

    # --- Event ingestion ---------------------------------------------------

    def record_event(self, event: dict[str, Any]) -> None:
        """Apply an event to the state. Pure function over the state.

        This is called by both live ingestion and replay - identical logic
        guarantees that replaying the JSONL produces the same state as
        live operation did.
        """
        self.state.recent_events.append(event)
        kind = event.get("event_type")
        ts = self._coerce_timestamp(event.get("timestamp"))

        if kind == "medication_taken" and ts is not None:
            self.state.last_medication_time = ts

        elif kind == "activity_started" and ts is not None:
            self.state.current_activity = event.get("activity")
            self.state.activity_start = ts

        elif kind == "activity_ended":
            self.state.current_activity = None
            self.state.activity_start = None

        elif kind == "risk_window_opened":
            window = RiskWindow(
                id=event["window_id"],
                scenario=event.get("scenario", "unknown"),
                start_time=ts or datetime.now(timezone.utc),
            )
            self.state.open_risk_windows.append(window)

        elif kind == "risk_window_closed":
            window_id = event.get("window_id")
            for w in self.state.open_risk_windows:
                if w.id == window_id:
                    w.status = event.get("status", "resolved")
                    w.end_time = ts
            self.state.open_risk_windows = [
                w for w in self.state.open_risk_windows if w.status == "active"
            ]

        elif kind == "persons_detected":
            persons = event.get("persons") or []
            if isinstance(persons, list):
                self.state.last_known_persons = persons

    # --- Helpers -----------------------------------------------------------

    @staticmethod
    def _coerce_timestamp(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def snapshot(self) -> dict[str, Any]:
        """Plain-dict view, useful for prompt builders and tests."""
        return {
            "last_medication_time": self.state.last_medication_time,
            "current_activity": self.state.current_activity,
            "activity_start": self.state.activity_start,
            "open_risk_windows": [w.scenario for w in self.state.open_risk_windows],
            "last_known_persons": list(self.state.last_known_persons),
            "recent_events_count": len(self.state.recent_events),
        }
