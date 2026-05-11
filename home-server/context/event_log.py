"""Append-only JSONL event log.

This is the audit trail. Never read during inference - only on server
startup (replay) and from the Streamlit log/feedback pages. Writes are
fire-and-forget so the inference path never blocks on disk.

File layout: storage/events/YYYY-MM-DD.jsonl, one JSON object per line.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.config import settings


def _parse_event_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _path_for(day: date, events_dir: Path | None = None) -> Path:
    base = events_dir if events_dir is not None else settings.events_dir
    return base / f"{day.isoformat()}.jsonl"


def append_event(event: dict[str, Any], *, events_dir: Path | None = None) -> None:
    """Append one event to today's log file.

    Event must already contain a `timestamp` ISO-8601 string. Caller is
    responsible for shape — this layer is intentionally schemaless to
    avoid coupling the audit trail to the in-memory state model.
    """
    base = events_dir if events_dir is not None else settings.events_dir
    base.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    path = _path_for(today, base)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


def read_events_since(cutoff: datetime, *, events_dir: Path | None = None) -> Iterable[dict[str, Any]]:
    """Yield events newer than `cutoff` from today's and yesterday's logs.

    Used by ContextManager.replay() on server startup. Two days are read
    so a server restart at 00:30 still picks up late-night events.
    """
    base = events_dir if events_dir is not None else settings.events_dir
    today = datetime.now(timezone.utc).date()
    yesterday = date.fromordinal(today.toordinal() - 1)

    for day in (yesterday, today):
        path = _path_for(day, base)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = event.get("timestamp")
                if isinstance(ts, str):
                    event_time = _parse_event_time(ts)
                    if event_time is None:
                        continue
                    if event_time >= cutoff:
                        yield event
