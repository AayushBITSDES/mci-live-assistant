"""Build a fresh ContextManager view for Streamlit pages.

Streamlit and FastAPI run in separate processes; rather than build IPC,
both rebuild ContextManager state by replaying the JSONL audit trail.
This module wraps that pattern + reads the recent feedback log + the
embeddings store, so each Streamlit page is one function call away from
"current state."
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from context.manager import ContextManager
from context.replay import replay_recent


def fresh_context(events_dir: Path, *, lookback_hours: float = 24.0) -> ContextManager:
    """Build a ContextManager from the last `lookback_hours` of events.

    Default is 24h (vs FastAPI's 2h) so Streamlit shows a fuller picture
    for daily review. The cost is trivial — even a busy day produces
    well under a few thousand events.
    """
    cm = ContextManager()
    replay_recent(cm, lookback_hours=lookback_hours, events_dir=events_dir)
    return cm


def read_recent_feedback(feedback_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Return the last `limit` entries from feedback/wrong_nudges.jsonl, newest first."""
    path = feedback_dir / "wrong_nudges.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out


def list_known_faces(faces_dir: Path) -> dict[str, int]:
    """Return {name: photo_count} for each subdirectory under storage/faces.

    Subdirectory name = person name. We do not rely on the embeddings
    file because the user may have just dropped photos and not embedded
    them yet.
    """
    if not faces_dir.exists():
        return {}
    out: dict[str, int] = {}
    for entry in faces_dir.iterdir():
        if not entry.is_dir():
            continue
        photos = [
            p for p in entry.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        out[entry.name] = len(photos)
    return out


def read_recent_events(events_dir: Path, limit: int = 50) -> list[dict[str, Any]]:
    """Return the last `limit` events from the JSONL log, newest first."""
    cm = fresh_context(events_dir, lookback_hours=72.0)
    events = list(cm.state.recent_events)
    events.reverse()
    return events[:limit]
