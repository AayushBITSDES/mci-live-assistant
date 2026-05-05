"""Tests for dashboard.state (cross-process state helpers)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from context.event_log import append_event
from dashboard.state import (
    fresh_context,
    list_known_faces,
    read_recent_events,
    read_recent_feedback,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- fresh_context replays the JSONL log --------------------------------

def test_fresh_context_replays_events(tmp_path: Path) -> None:
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    append_event(
        {"event_type": "medication_taken", "timestamp": _now().isoformat()},
        events_dir=events_dir,
    )
    cm = fresh_context(events_dir, lookback_hours=24.0)
    assert cm.state.last_medication_time is not None


def test_fresh_context_empty_when_no_events(tmp_path: Path) -> None:
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    cm = fresh_context(events_dir)
    assert cm.state.last_medication_time is None
    assert cm.state.current_activity is None


# --- read_recent_feedback newest-first ---------------------------------

def test_read_recent_feedback_returns_newest_first(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    log = feedback_dir / "wrong_nudges.jsonl"
    lines = [
        json.dumps({"timestamp": "2026-05-05T10:00:00Z", "nudge_sentence": "first"}),
        json.dumps({"timestamp": "2026-05-05T11:00:00Z", "nudge_sentence": "second"}),
        json.dumps({"timestamp": "2026-05-05T12:00:00Z", "nudge_sentence": "third"}),
    ]
    log.write_text("\n".join(lines) + "\n")

    entries = read_recent_feedback(feedback_dir, limit=10)
    assert [e["nudge_sentence"] for e in entries] == ["third", "second", "first"]


def test_read_recent_feedback_handles_missing_file(tmp_path: Path) -> None:
    assert read_recent_feedback(tmp_path / "feedback") == []


def test_read_recent_feedback_skips_malformed_lines(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    log = feedback_dir / "wrong_nudges.jsonl"
    log.write_text(
        "{not json}\n"
        + json.dumps({"timestamp": "2026-05-05T10:00:00Z", "nudge_sentence": "real"})
        + "\n"
    )
    entries = read_recent_feedback(feedback_dir)
    assert len(entries) == 1
    assert entries[0]["nudge_sentence"] == "real"


def test_read_recent_feedback_respects_limit(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    log = feedback_dir / "wrong_nudges.jsonl"
    log.write_text("\n".join(
        json.dumps({"timestamp": f"2026-05-05T{i:02d}:00:00Z", "n": i})
        for i in range(10)
    ))
    entries = read_recent_feedback(feedback_dir, limit=3)
    assert len(entries) == 3


# --- list_known_faces ---------------------------------------------------

def test_list_known_faces_counts_photos_per_person(tmp_path: Path) -> None:
    faces_dir = tmp_path / "faces"
    (faces_dir / "Anjali").mkdir(parents=True)
    (faces_dir / "Anjali" / "photo_01.jpg").write_bytes(b"fake")
    (faces_dir / "Anjali" / "photo_02.png").write_bytes(b"fake")
    (faces_dir / "Ravi").mkdir(parents=True)
    (faces_dir / "Ravi" / "photo_01.jpeg").write_bytes(b"fake")

    counts = list_known_faces(faces_dir)
    assert counts == {"Anjali": 2, "Ravi": 1}


def test_list_known_faces_ignores_non_image_files(tmp_path: Path) -> None:
    faces_dir = tmp_path / "faces"
    (faces_dir / "Anjali").mkdir(parents=True)
    (faces_dir / "Anjali" / "photo.jpg").write_bytes(b"fake")
    (faces_dir / "Anjali" / "notes.txt").write_text("hello")  # ignored
    counts = list_known_faces(faces_dir)
    assert counts == {"Anjali": 1}


def test_list_known_faces_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    assert list_known_faces(tmp_path / "no-such") == {}


# --- read_recent_events --------------------------------------------------

def test_read_recent_events_newest_first(tmp_path: Path) -> None:
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    base = _now() - timedelta(minutes=10)
    for i in range(5):
        append_event(
            {"event_type": "tick", "timestamp": (base + timedelta(seconds=i)).isoformat(), "i": i},
            events_dir=events_dir,
        )
    out = read_recent_events(events_dir, limit=3)
    assert len(out) == 3
    # Newest first
    assert out[0]["i"] == 4
    assert out[2]["i"] == 2
