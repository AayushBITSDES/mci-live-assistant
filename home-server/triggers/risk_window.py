"""RiskWindow lifecycle helpers.

A RiskWindow is a bounded interval during which the system is actively
monitoring for a particular failure mode (stove unattended, abandoned
cooking, medication reminder pending). The dataclass itself lives in
context/manager.py — this module provides creation + transition helpers
that emit the corresponding events for the audit trail.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


def open_risk_window_event(scenario: str, *, now: datetime | None = None) -> dict[str, Any]:
    """Build the event dict that opens a risk window.

    Caller is responsible for both:
      1. appending this to the JSONL log (via context.event_log)
      2. passing it through ContextManager.record_event()
    """
    return {
        "event_type": "risk_window_opened",
        "window_id": str(uuid.uuid4()),
        "scenario": scenario,
        "timestamp": (now or datetime.now(timezone.utc)).isoformat(),
    }


def close_risk_window_event(window_id: str, status: str, *, now: datetime | None = None) -> dict[str, Any]:
    """Build the event dict that closes a risk window.

    `status` is one of: "resolved" (system resolved it), "closed" (user
    explicitly closed via voice command), or any other label the caller
    finds useful.
    """
    return {
        "event_type": "risk_window_closed",
        "window_id": window_id,
        "status": status,
        "timestamp": (now or datetime.now(timezone.utc)).isoformat(),
    }
