"""Rebuild ContextManager state from the JSONL audit trail on startup.

Reads the last `lookback_hours` of events and applies them in order.
Same `record_event` path as live operation - so replay correctness is
guaranteed by the live tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from context.event_log import read_events_since
from context.manager import ContextManager


def replay_recent(
    manager: ContextManager,
    lookback_hours: float = 2.0,
    *,
    events_dir: Path | None = None,
) -> int:
    """Apply the last `lookback_hours` of events into `manager`.

    Returns the number of events replayed.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    count = 0
    for event in read_events_since(cutoff, events_dir=events_dir):
        manager.record_event(event)
        count += 1
    return count
