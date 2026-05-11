"""Harness tests for ContextManager + event log + replay.

These run without any external services (no API keys, no GPU).
They are the safety net that proves the temporal-memory layer works
identically in live mode and replay mode - which is the whole point
of having an event log alongside in-memory state.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from context.builder import build_summary
from context.event_log import append_event, read_events_since
from context.manager import ContextManager
from context.replay import replay_recent


@pytest.fixture
def tmp_events_dir(tmp_path: Path) -> Path:
    d = tmp_path / "events"
    d.mkdir()
    return d


def _now() -> datetime:
    return datetime.now(timezone.utc)


# -- ContextManager: derived facts update correctly ----------------------

def test_medication_event_updates_last_medication_time() -> None:
    cm = ContextManager()
    ts = _now()
    cm.record_event({"event_type": "medication_taken", "timestamp": ts.isoformat()})
    assert cm.state.last_medication_time is not None
    assert cm.state.last_medication_time.replace(microsecond=0) == ts.replace(microsecond=0)


def test_activity_started_then_ended_clears_state() -> None:
    cm = ContextManager()
    cm.record_event({
        "event_type": "activity_started",
        "timestamp": _now().isoformat(),
        "activity": "making_tea",
    })
    assert cm.state.current_activity == "making_tea"
    assert cm.state.activity_start is not None

    cm.record_event({"event_type": "activity_ended", "timestamp": _now().isoformat()})
    assert cm.state.current_activity is None
    assert cm.state.activity_start is None


def test_risk_window_open_then_close_removes_from_active_list() -> None:
    cm = ContextManager()
    cm.record_event({
        "event_type": "risk_window_opened",
        "window_id": "rw-1",
        "scenario": "stove_unattended",
        "timestamp": _now().isoformat(),
    })
    assert len(cm.state.open_risk_windows) == 1
    assert cm.state.open_risk_windows[0].scenario == "stove_unattended"

    cm.record_event({
        "event_type": "risk_window_closed",
        "window_id": "rw-1",
        "status": "resolved",
        "timestamp": _now().isoformat(),
    })
    assert cm.state.open_risk_windows == []


def test_recent_events_caps_at_15() -> None:
    cm = ContextManager()
    for i in range(20):
        cm.record_event({"event_type": "noop", "timestamp": _now().isoformat(), "i": i})
    assert len(cm.state.recent_events) == 15
    # Oldest 5 evicted; deque preserves insertion order
    assert cm.state.recent_events[0]["i"] == 5
    assert cm.state.recent_events[-1]["i"] == 19


def test_persons_detected_replaces_list() -> None:
    cm = ContextManager()
    cm.record_event({"event_type": "persons_detected", "persons": ["Anjali"], "timestamp": _now().isoformat()})
    cm.record_event({"event_type": "persons_detected", "persons": ["Anjali", "Ravi"], "timestamp": _now().isoformat()})
    assert cm.state.last_known_persons == ["Anjali", "Ravi"]


# -- Event log: append + read round-trip ---------------------------------

def test_append_and_read_round_trip(tmp_events_dir: Path) -> None:
    ts = _now()
    append_event(
        {"event_type": "medication_taken", "timestamp": ts.isoformat()},
        events_dir=tmp_events_dir,
    )
    events = list(read_events_since(ts - timedelta(hours=1), events_dir=tmp_events_dir))
    assert len(events) == 1
    assert events[0]["event_type"] == "medication_taken"


def test_read_filters_by_cutoff(tmp_events_dir: Path) -> None:
    old = _now() - timedelta(hours=5)
    new = _now()
    append_event({"event_type": "old", "timestamp": old.isoformat()}, events_dir=tmp_events_dir)
    append_event({"event_type": "new", "timestamp": new.isoformat()}, events_dir=tmp_events_dir)
    events = list(read_events_since(_now() - timedelta(hours=2), events_dir=tmp_events_dir))
    types = [e["event_type"] for e in events]
    assert "new" in types
    assert "old" not in types


def test_read_accepts_legacy_naive_timestamps(tmp_events_dir: Path) -> None:
    naive_now = datetime.now().replace(microsecond=0)
    append_event({"event_type": "legacy", "timestamp": naive_now.isoformat()}, events_dir=tmp_events_dir)
    events = list(read_events_since(_now() - timedelta(hours=1), events_dir=tmp_events_dir))
    assert [e["event_type"] for e in events] == ["legacy"]


# -- Replay invariant: same events -> same state -------------------------

def test_replay_produces_identical_state_to_live(tmp_events_dir: Path) -> None:
    """Critical invariant: live and replay must converge to the same state.

    This is what justifies treating JSONL as the audit trail and
    ContextManager as the runtime source of truth.
    """
    sequence = [
        {"event_type": "activity_started", "activity": "making_tea", "timestamp": _now().isoformat()},
        {"event_type": "medication_taken", "timestamp": _now().isoformat()},
        {"event_type": "risk_window_opened", "window_id": "rw-1", "scenario": "stove_unattended", "timestamp": _now().isoformat()},
        {"event_type": "persons_detected", "persons": ["Anjali"], "timestamp": _now().isoformat()},
    ]

    # Live path
    live = ContextManager()
    for e in sequence:
        live.record_event(e)
        append_event(e, events_dir=tmp_events_dir)

    # Replay path - fresh manager, replay from disk
    replayed = ContextManager()
    count = replay_recent(replayed, lookback_hours=24, events_dir=tmp_events_dir)
    assert count == len(sequence)

    assert live.snapshot() == replayed.snapshot()


# -- Prompt builder: summary contains expected facts ---------------------

def test_summary_includes_activity_and_medication() -> None:
    cm = ContextManager()
    cm.record_event({
        "event_type": "activity_started",
        "activity": "making_tea",
        "timestamp": (_now() - timedelta(minutes=4)).isoformat(),
    })
    cm.record_event({
        "event_type": "medication_taken",
        "timestamp": (_now() - timedelta(hours=1)).isoformat(),
    })
    summary = build_summary(cm)
    assert "[Recent Context]" in summary
    assert "making_tea" in summary
    assert "medication" in summary.lower()


def test_naive_timestamp_is_normalized_to_utc_in_state() -> None:
    cm = ContextManager()
    naive_now = datetime.now().replace(microsecond=0)
    cm.record_event({"event_type": "activity_started", "activity": "making_tea", "timestamp": naive_now.isoformat()})
    assert cm.state.activity_start is not None
    assert cm.state.activity_start.tzinfo is timezone.utc


def test_summary_when_empty_is_still_informative() -> None:
    cm = ContextManager()
    summary = build_summary(cm)
    assert "no record today" in summary
    assert "none observed" in summary
