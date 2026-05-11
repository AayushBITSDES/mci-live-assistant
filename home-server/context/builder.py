"""Render ContextManager state as a 3-5 line summary for the LLM prompt.

Output is deliberately compact and human-readable. Every LLM call gets
this block prepended so the model has the same situational awareness
the local rules do.
"""
from __future__ import annotations

from datetime import datetime, timezone

from context.manager import ContextManager


def _humanize_age(ts: datetime, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    delta = now - ts
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h {minutes % 60}m ago"
    return ts.isoformat()


def build_summary(manager: ContextManager, now: datetime | None = None) -> str:
    """Return a multi-line plain-text block ready to prepend to the prompt."""
    s = manager.state
    lines: list[str] = ["[Recent Context]"]

    if s.current_activity and s.activity_start:
        lines.append(
            f"- Current activity: {s.current_activity} (started {_humanize_age(s.activity_start, now)})"
        )
    else:
        lines.append("- Current activity: none observed")

    if s.last_medication_time:
        lines.append(f"- Last medication taken: {_humanize_age(s.last_medication_time, now)}")
    else:
        lines.append("- Last medication taken: no record today")

    if s.open_risk_windows:
        scenarios = ", ".join(w.scenario for w in s.open_risk_windows)
        lines.append(f"- Open risk windows: {scenarios}")

    if s.last_known_persons:
        lines.append(f"- People recently in frame: {', '.join(s.last_known_persons)}")

    return "\n".join(lines)
